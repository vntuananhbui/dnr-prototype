"""Train 4 cấu hình × các seed, lưu model vào artifacts/.

Mỗi run gồm 2 phase (paper Fig.2):
  Phase I  : train encoder + g. divide=true -> L_task + λ_uncor L_uncor + λ_corr L_corr (Eq.5);
             divide=false -> chỉ L_task (cùng kiến trúc, để ablation 2×2 công bằng).
  Phase II : đóng băng encoder, train backbone. refine=true -> L_task + λ1 L_aug-intra + λ2 L_aug-mask (Eq.15);
             refine=false -> chỉ L_task.

Chạy: python train.py [--configs a,b] [--seeds 0,1] [--phase1-epochs N] [--phase2-epochs N] [--frac F]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

import data as data_mod
import metrics
from model import MODS, DnR, divide_losses, pick_device, refine_losses

ART = "artifacts"


# ---------------- tiện ích dữ liệu ----------------

def load_dataset(path="data/dataset.npz") -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def norm_stats(ds: dict) -> dict:
    tr = ds["dialogue_split"][ds["dialogue"]] == 0
    return {m: (ds[m][tr].mean(0), ds[m][tr].std(0) + 1e-6) for m in MODS}


def pad_dialogues(dlgs: list, norm: dict, device="cpu") -> dict:
    """List hội thoại -> tensor đệm: x[m] (B,T,D), y (B,T) với -1 ở phần đệm, lengths (B,)."""
    B = len(dlgs)
    T = max(len(d["y"]) for d in dlgs)
    out = {}
    for m in MODS:
        D = dlgs[0][m].shape[1]
        arr = np.zeros((B, T, D), dtype=np.float32)
        for i, d in enumerate(dlgs):
            arr[i, : len(d["y"])] = (d[m] - norm[m][0]) / norm[m][1]
        out[m] = torch.tensor(arr, device=device)
    y = np.full((B, T), -1, dtype=np.int64)
    for i, d in enumerate(dlgs):
        y[i, : len(d["y"])] = d["y"]
    out["y"] = torch.tensor(y, device=device)
    out["lengths"] = torch.tensor([len(d["y"]) for d in dlgs])
    return out


def batches(n, bs, rng):
    idx = rng.permutation(n)
    for i in range(0, n, bs):
        yield idx[i: i + bs]


def subset(p: dict, idx) -> dict:
    idx_t = torch.as_tensor(idx)
    out = {k: v[idx_t] for k, v in p.items()}
    T = int(out["lengths"].max())
    for k in list(MODS) + ["y"]:
        out[k] = out[k][:, :T]
    return out


@torch.no_grad()
def predict_padded(model: DnR, p: dict):
    model.eval()
    logits, h, z = model({m: p[m] for m in MODS}, p["lengths"])
    valid = p["y"] >= 0
    return logits, valid


def evaluate(model: DnR, p: dict, n_classes: int) -> dict:
    logits, valid = predict_padded(model, p)
    yt = p["y"][valid].cpu().numpy()
    yp = logits[valid].argmax(-1).cpu().numpy()
    return {"acc": metrics.accuracy(yt, yp), "wf1": metrics.weighted_f1(yt, yp, n_classes)}


# ---------------- train một run ----------------

def train_one(cfg: dict, name: str, seed: int, overrides: dict | None = None) -> dict:
    overrides = overrides or {}
    torch.set_num_threads(int(cfg["train"]["threads_per_run"]))
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    device = pick_device()
    flags = cfg["configs"][name]
    tc, dvc, rfc = cfg["train"], cfg["divide"], cfg["refine"]
    C = cfg["data"]["n_classes"]
    e1 = int(overrides.get("phase1_epochs", tc["phase1_epochs"]))
    e2 = int(overrides.get("phase2_epochs", tc["phase2_epochs"]))
    frac = float(overrides.get("frac", 1.0))

    ds = load_dataset()
    norm = norm_stats(ds)
    tr_d = data_mod.dialogues_of(ds, 0)
    tr_d = tr_d[: max(2, int(len(tr_d) * frac))]
    va = pad_dialogues(data_mod.dialogues_of(ds, 1), norm, device)
    tr = pad_dialogues(tr_d, norm, device)
    n_tr = len(tr_d)
    bs = int(tc["batch_size"])

    model = DnR(cfg).to(device)
    t0 = time.time()
    hist = {"phase1": [], "phase2": []}

    # ---- Phase I: Divide ----
    opt = torch.optim.AdamW(model.divide.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    for ep in range(e1):
        model.divide.train()
        tot = {"task": 0.0, "uncor": 0.0, "corr": 0.0}
        for bi in batches(n_tr, bs, rng):
            b = subset(tr, bi)
            valid = b["y"] >= 0
            x = {m: b[m][valid] for m in MODS}
            logits, h = model.divide(x)
            l_task = F.cross_entropy(logits, b["y"][valid])
            loss = l_task
            if flags["divide"]:
                l_unc, l_cor = divide_losses(h, dvc["alpha"])
                loss = loss + dvc["lambda_uncor"] * l_unc + dvc["lambda_corr"] * l_cor
                tot["uncor"] += l_unc.item()
                tot["corr"] += l_cor.item()
            tot["task"] += l_task.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        hist["phase1"].append({k: round(v, 5) for k, v in tot.items()})

    # ---- Phase II: đóng băng Divide, train backbone ----
    for p_ in model.divide.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.AdamW(model.backbone.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    best = {"wf1": -1.0}
    best_state = None
    for ep in range(e2):
        model.backbone.train()
        model.divide.eval()
        tot = {"task": 0.0, "aug_intra": 0.0, "aug_mask": 0.0}
        for bi in batches(n_tr, bs, rng):
            b = subset(tr, bi)
            valid = b["y"] >= 0
            with torch.no_grad():
                h = model.divide.encode({m: b[m] for m in MODS})
            if flags["refine"]:
                logits, l_intra, l_mask = refine_losses(
                    model.backbone, h, b["lengths"], valid, rfc["K"], rfc["aug_sigma"], rfc["temperature"])
                l_task = F.cross_entropy(logits[valid], b["y"][valid])
                loss = l_task + rfc["lambda1"] * l_intra + rfc["lambda2"] * l_mask
                tot["aug_intra"] += l_intra.item()
                tot["aug_mask"] += l_mask.item()
            else:
                logits, _ = model.backbone(h, b["lengths"])
                l_task = F.cross_entropy(logits[valid], b["y"][valid])
                loss = l_task
            tot["task"] += l_task.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = evaluate(model, va, C)
        hist["phase2"].append({**{k: round(x, 5) for k, x in tot.items()}, "val_acc": v["acc"], "val_wf1": v["wf1"]})
        if v["wf1"] > best["wf1"]:
            best = {**v, "epoch": ep + 1}
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    secs = time.time() - t0
    os.makedirs(ART, exist_ok=True)
    path = os.path.join(ART, f"{name}_seed{seed}.pt")
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "norm": {m: (norm[m][0].tolist(), norm[m][1].tolist()) for m in MODS},
        "class_names": cfg["data"]["class_names"],
        "config": cfg, "name": name, "seed": seed,
        "phase1_epochs": e1, "phase2_epochs": e2, "frac": frac,
        "best_val": best, "history": hist, "train_seconds": secs,
    }, path + ".tmp")
    os.replace(path + ".tmp", path)
    return {"name": name, "seed": seed, "path": path, "best_val": best, "train_seconds": round(secs, 2)}


def _job(args):
    return train_one(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--configs", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--phase1-epochs", type=int, default=None)
    ap.add_argument("--phase2-epochs", type=int, default=None)
    ap.add_argument("--frac", type=float, default=None)
    a = ap.parse_args()
    cfg = data_mod.load_config(a.config)
    names = a.configs.split(",") if a.configs else list(cfg["configs"])
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else cfg["train"]["seeds"]
    ov = {k: v for k, v in {"phase1_epochs": a.phase1_epochs, "phase2_epochs": a.phase2_epochs,
                            "frac": a.frac}.items() if v is not None}
    jobs = [(cfg, n, s, ov) for n in names for s in seeds]
    t0 = time.time()
    workers = min(int(cfg["train"]["max_workers"]), len(jobs), os.cpu_count() or 1)
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        results = list(ex.map(_job, jobs))
    for r in results:
        print(f"{r['name']:14s} seed={r['seed']} val_wf1={r['best_val']['wf1']:.4f} "
              f"epoch={r['best_val']['epoch']} {r['train_seconds']}s -> {r['path']}")
    log = {"runs": results, "wall_seconds": round(time.time() - t0, 2), "overrides": ov}
    with open(os.path.join(ART, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"train wall time: {log['wall_seconds']}s")


if __name__ == "__main__":
    main()
