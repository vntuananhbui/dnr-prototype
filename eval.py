"""Đánh giá mọi model trong artifacts/ trên tập test -> results/metrics.json.

Ghi file tạm rồi os.replace; lỗi giữa chừng thì không để lại metrics.json dở (xoá bản cũ trước khi chạy).
Chạy: python eval.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time

import numpy as np
import torch

import data as data_mod
import metrics
from model import MODS, divide_losses
from predict import load_model
from train import ART, load_dataset, norm_stats, pad_dialogues

OUT = os.path.join("results", "metrics.json")
LABEL = "Bản thu nhỏ, dữ liệu mô phỏng, không phải model/số của paper"


@torch.no_grad()
def eval_model(path: str, test_dlgs: list, cfg: dict) -> dict:
    model, ck, norm = load_model(path)
    torch.set_num_threads(4)
    C = cfg["data"]["n_classes"]
    p = pad_dialogues(test_dlgs, norm)
    valid = p["y"] >= 0
    yt = p["y"][valid].numpy()
    x = {m: p[m] for m in MODS}
    h = model.divide.encode(x)
    logits, _ = model.backbone(h, p["lengths"])
    prob = torch.softmax(logits, -1)
    yp = prob.argmax(-1)[valid].numpy()
    res = {
        "acc": metrics.accuracy(yt, yp),
        "wf1": metrics.weighted_f1(yt, yp, C),
        "per_class_f1": metrics.per_class_f1(yt, yp, C).tolist(),
        "confusion": metrics.confusion_matrix(yt, yp, C).tolist(),
        "best_val": ck["best_val"],
        "train_seconds": round(ck["train_seconds"], 2),
        "phase1_epochs": ck["phase1_epochs"],
        "phase2_epochs": ck["phase2_epochs"],
    }
    # Chẩn đoán Divide trên test: các đại lượng của Eq.7 / Eq.8 (không trọng số)
    hv = h[valid]
    l_unc, l_cor = divide_losses(hv, cfg["divide"]["alpha"])
    res["divide_diag"] = {"L_uncor": float(l_unc), "L_corr": float(l_cor)}
    # Đóng góp trung bình từng luồng: xác suất lớp đúng giảm bao nhiêu khi zero luồng (m, s)
    pt = prob.gather(-1, p["y"].clamp(min=0)[..., None])[..., 0]
    occ = np.zeros((3, 3))
    for mi in range(3):
        for si in range(3):
            hz = h.clone()
            hz[..., mi, si, :] = 0.0
            pz = torch.softmax(model.backbone(hz, p["lengths"])[0], -1)
            pzt = pz.gather(-1, p["y"].clamp(min=0)[..., None])[..., 0]
            occ[mi, si] = float((pt - pzt)[valid].mean())
    res["occlusion_true_prob_drop"] = occ.tolist()  # [modality a,v,t][stream U,R,S]
    # Độ bền: tăng độ lệch chuẩn nhiễu của cả 3 modality từ noise_std lên noise_std + e
    rob = {}
    ns = cfg["data"]["noise_std"]
    for e in cfg["eval"]["robustness_noise"]:
        g = torch.Generator().manual_seed(1234)
        extra = float(np.sqrt((ns + e) ** 2 - ns ** 2))
        xn = {}
        for m in MODS:
            raw = p[m] * torch.tensor(norm[m][1]) + torch.tensor(norm[m][0])
            raw = raw + torch.randn(raw.shape, generator=g) * extra
            xn[m] = (raw - torch.tensor(norm[m][0])) / torch.tensor(norm[m][1])
        lg, _ = model.backbone(model.divide.encode(xn), p["lengths"])
        ypn = lg.argmax(-1)[valid].numpy()
        rob[str(e)] = {"acc": metrics.accuracy(yt, ypn), "wf1": metrics.weighted_f1(yt, ypn, C)}
    res["robustness"] = rob
    return res


def verdict(gain_points: float, thr: dict) -> str:
    return "Khớp một phần" if gain_points >= thr["k1_partial_gain"] else "Chưa tái hiện được"


def main(cfg_path="config.yaml"):
    t0 = time.time()
    if os.path.exists(OUT):
        os.remove(OUT)
    cfg = data_mod.load_config(cfg_path)
    ds = load_dataset()
    test = data_mod.dialogues_of(ds, 2)
    runs, summary = {}, {}
    for name in cfg["configs"]:
        runs[name] = {}
        for s in cfg["train"]["seeds"]:
            path = os.path.join(ART, f"{name}_seed{s}.pt")
            runs[name][str(s)] = eval_model(path, test, cfg)
        rs = list(runs[name].values())
        summary[name] = {
            "acc": metrics.mean_std([r["acc"] for r in rs]),
            "wf1": metrics.mean_std([r["wf1"] for r in rs]),
            "robustness_wf1": {e: metrics.mean_std([r["robustness"][e]["wf1"] for r in rs])
                               for e in rs[0]["robustness"]},
            "occlusion_true_prob_drop": np.mean([r["occlusion_true_prob_drop"] for r in rs], 0).tolist(),
            "divide_diag": {k: metrics.mean_std([r["divide_diag"][k] for r in rs]) for k in rs[0]["divide_diag"]},
        }
    base = summary["baseline"]["wf1"]["mean"]
    gains = {}
    for name in cfg["configs"]:
        per_seed = [100 * (runs[name][s]["wf1"] - runs["baseline"][s]["wf1"]) for s in runs[name]]
        gains[name] = {"wf1_points_mean": 100 * (summary[name]["wf1"]["mean"] - base),
                       "wf1_points_per_seed": per_seed,
                       "seeds_better_than_baseline": int(sum(g > 0 for g in per_seed))}
    g = gains["divide_refine"]["wf1_points_mean"]
    train_log = {}
    if os.path.exists(os.path.join(ART, "train_log.json")):
        with open(os.path.join(ART, "train_log.json"), encoding="utf-8") as f:
            train_log = json.load(f)
    out = {
        "meta": {
            "label": LABEL,
            "paper": "Divide and Refine (DnR), arXiv 2601.14274",
            "level": "C",
            "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "n_test_dialogues": len(test),
            "n_test_utterances": int(sum(len(d["y"]) for d in test)),
            "train_wall_seconds": train_log.get("wall_seconds"),
            "eval_seconds": None,
            "torch": torch.__version__,
            "metric_units": "acc và wf1 là tỉ lệ 0-1; gain tính bằng điểm % (×100)",
        },
        "config": cfg,
        "runs": runs,
        "summary": summary,
        "gain_vs_baseline": gains,
        "k1": {
            "claim": "Thêm DnR tăng W-F1 so với backbone gốc (paper Tab.3/4)",
            "check_type": "kiểm thay thế (dữ liệu mô phỏng, backbone late-fusion+BiGRU)",
            "measured": "mean W-F1(divide_refine) - mean W-F1(baseline), điểm %",
            "gain_points": g,
            "threshold_partial": cfg["thresholds"]["k1_partial_gain"],
            "verdict": verdict(g, cfg["thresholds"]),
        },
    }
    out["meta"]["eval_seconds"] = round(time.time() - t0, 2)
    os.makedirs("results", exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    os.replace(tmp, OUT)
    for name in cfg["configs"]:
        s = summary[name]
        print(f"{name:14s} acc={s['acc']['mean']:.4f}±{s['acc']['std']:.4f} "
              f"wf1={s['wf1']['mean']:.4f}±{s['wf1']['std']:.4f} "
              f"Δwf1={gains[name]['wf1_points_mean']:+.2f} điểm")
    print(f"K1: gain={g:+.2f} điểm -> {out['k1']['verdict']} | ghi {OUT} ({out['meta']['eval_seconds']}s)")


if __name__ == "__main__":
    main()
