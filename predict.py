"""Dự đoán một hội thoại bằng model đã lưu trong artifacts/.

Trả cho từng utterance: nhãn, độ tin cậy, xác suất, độ lớn (L2) của U/R/S theo modality,
và đóng góp của từng luồng U/R/S (xác suất lớp dự đoán giảm bao nhiêu khi zero luồng đó).

Chạy: python predict.py --seed 7 [--noise_a 0 --noise_v 0 --noise_t 0] [--model artifacts/divide_refine_seed0.pt]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import data as data_mod
from model import MODS, STREAMS, DnR

ART = "artifacts"
_CACHE: dict = {}


def load_model(path: str):
    """Nạp đúng model + tham số chuẩn hoá đã lưu bởi train.py (có cache)."""
    if path in _CACHE:
        return _CACHE[path]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chưa có model {path}. Chạy ./reproduce.sh trước.")
    torch.set_num_threads(min(4, os.cpu_count() or 1))  # tránh tranh luồng với batch nhỏ
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = DnR(ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    norm = {m: (np.asarray(ck["norm"][m][0], dtype=np.float32), np.asarray(ck["norm"][m][1], dtype=np.float32))
            for m in MODS}
    _CACHE[path] = (model, ck, norm)
    return _CACHE[path]


def _to_tensor(x: dict, norm: dict) -> dict:
    return {m: torch.tensor((np.asarray(x[m], dtype=np.float32) - norm[m][0]) / norm[m][1])[None] for m in MODS}


@torch.no_grad()
def predict_dialogue(path: str, x: dict) -> dict:
    """x: dict m -> mảng (L, D_m) đặc trưng thô của một hội thoại."""
    model, ck, norm = load_model(path)
    xt = _to_tensor(x, norm)
    L = xt["a"].shape[1]
    lengths = torch.tensor([L])
    h = model.divide.encode(xt)                      # (1, L, 3, 3, d)
    logits, _ = model.backbone(h, lengths)
    probs = torch.softmax(logits[0], -1)             # (L, C)
    conf, pred = probs.max(-1)
    energy = h[0].norm(dim=-1)                       # (L, 3, 3)
    contrib = torch.zeros(L, 3, 3)
    for mi in range(3):
        for si in range(3):
            hz = h.clone()
            hz[..., mi, si, :] = 0.0
            pz = torch.softmax(model.backbone(hz, lengths)[0][0], -1)
            contrib[:, mi, si] = conf - pz.gather(-1, pred[:, None])[:, 0]
    names = ck["class_names"]
    return {
        "model": os.path.basename(path),
        "config_name": ck["name"],
        "seed": ck["seed"],
        "pred": pred.tolist(),
        "labels": [names[i] for i in pred.tolist()],
        "confidence": conf.tolist(),
        "probs": probs.tolist(),
        "class_names": names,
        "modalities": list(MODS),
        "streams": list(STREAMS),
        "energy": energy.tolist(),        # [utt][modality][stream]
        "contribution": contrib.tolist(),  # [utt][modality][stream]
    }


def make_input(seed: int, noise: dict | None = None, cfg_path: str = "config.yaml"):
    """Sinh một hội thoại mới (không nằm trong tập train) cho demo."""
    cfg = data_mod.load_config(cfg_path)
    gen = data_mod.Generator(cfg["data"])
    x, y, _ = gen.dialogue(seed, extra_noise=noise)
    return x, y, cfg["data"]["class_names"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--noise_a", type=float, default=0.0)
    ap.add_argument("--noise_v", type=float, default=0.0)
    ap.add_argument("--noise_t", type=float, default=0.0)
    ap.add_argument("--model", default=os.path.join(ART, "divide_refine_seed0.pt"))
    a = ap.parse_args()
    x, y, names = make_input(a.seed, {"a": a.noise_a, "v": a.noise_v, "t": a.noise_t})
    r = predict_dialogue(a.model, x)
    print(f"model={r['model']} | hội thoại seed={a.seed}, {len(y)} utterance")
    print("utt  thật  đoán  tin_cậy  | đóng góp U/R/S cộng theo modality (a, v, t)")
    for i in range(len(y)):
        c = np.asarray(r["contribution"][i]).sum(1)
        print(f"{i:3d}  {names[y[i]]:4s}  {r['labels'][i]:4s}  {r['confidence'][i]:.3f}    "
              + ", ".join(f"{m}={v:+.3f}" for m, v in zip(MODS, c)))
    acc = float(np.mean(np.asarray(r["pred"]) == y))
    print(f"đúng {acc:.3f} trên hội thoại này")


if __name__ == "__main__":
    main()
