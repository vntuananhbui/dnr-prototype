"""Sinh dữ liệu hội thoại mô phỏng cho DnR (bản thu nhỏ).

Mỗi utterance có nhãn c ∈ {0..5} và 3 modality a, v, t. Thông tin nhãn được cài theo 3 kiểu
mà paper DnR muốn tách (§4.1):
  * unique      : mỗi modality mang riêng một phần nhãn
                  a -> c // 3 (2 nhóm), t -> c % 3 (3 nhóm), v -> c % 2 (2 nhóm)
  * redundant   : cả 3 modality cùng mang prototype của lớp c (yếu)
  * synergistic : vector dấu ngẫu nhiên z xuất hiện ở a, và q_c * z ở v; từng modality riêng
                  không lộ gì, chỉ tích a·v mới lộ q_c (kiểu XOR)
cộng nhiễu Gaussian. Nhãn trong hội thoại theo chuỗi Markov có quán tính (stay_prob).

Chạy: python data.py  -> ghi data/dataset.npz và in thống kê.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import yaml

MODS = ("a", "v", "t")
SYNERGY_SIGN = np.array([1, -1, -1, 1, 1, -1], dtype=np.float32)  # q_c, cố định


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Generator:
    """Giữ các ma trận chiếu và prototype cố định theo data.seed."""

    def __init__(self, dcfg: dict):
        self.cfg = dcfg
        rng = np.random.default_rng(dcfg["seed"])
        L = dcfg["latent_dim"]
        C = dcfg["n_classes"]
        dims = dcfg["dims"]
        # prototype ẩn (chuẩn hoá về độ dài sqrt(L))
        def protos(n):
            p = rng.standard_normal((n, L))
            return p / np.linalg.norm(p, axis=1, keepdims=True) * np.sqrt(L)
        self.unique_protos = {"a": protos(2), "t": protos(3), "v": protos(2)}
        self.red_protos = protos(C)
        # ma trận chiếu latent -> modality (mỗi kiểu tín hiệu một ma trận riêng)
        def proj(D):
            return (rng.standard_normal((L, D)) / np.sqrt(L)).astype(np.float32)
        self.P = {m: {k: proj(dims[m]) for k in ("unique", "red", "syn")} for m in MODS}

    @staticmethod
    def unique_index(m: str, c: np.ndarray) -> np.ndarray:
        return {"a": c // 3, "t": c % 3, "v": c % 2}[m]

    def labels(self, rng: np.random.Generator, n: int) -> np.ndarray:
        C = self.cfg["n_classes"]
        y = np.empty(n, dtype=np.int64)
        y[0] = rng.integers(C)
        for i in range(1, n):
            y[i] = y[i - 1] if rng.random() < self.cfg["stay_prob"] else rng.integers(C)
        return y

    def features(self, rng: np.random.Generator, y: np.ndarray, extra_noise=None):
        """Trả dict m -> (n, D_m) float32 và dict thành phần sạch (để minh hoạ)."""
        c = self.cfg
        L = c["latent_dim"]
        n = len(y)
        extra_noise = extra_noise or {}
        z = rng.choice([-1.0, 1.0], size=(n, L))
        syn_lat = {"a": z, "v": z * SYNERGY_SIGN[y][:, None], "t": np.zeros((n, L))}
        out, parts = {}, {}
        for m in MODS:
            P = self.P[m]
            u = self.unique_protos[m][self.unique_index(m, y)] @ P["unique"] * c["unique_strength"]
            r = self.red_protos[y] @ P["red"] * c["redundant_strength"]
            s = syn_lat[m] @ P["syn"] * c["synergy_strength"]
            sigma = c["noise_std"] + float(extra_noise.get(m, 0.0))
            noise = rng.standard_normal((n, c["dims"][m])) * sigma
            out[m] = (u + r + s + noise).astype(np.float32)
            parts[m] = {"unique": u, "redundant": r, "synergy": s}
        return out, parts

    def dialogue(self, seed: int, extra_noise=None, length=None):
        """Một hội thoại mới cho demo/predict, độc lập với tập train."""
        rng = np.random.default_rng(10_000_019 + int(seed))
        n = int(length or rng.integers(self.cfg["min_len"], self.cfg["max_len"] + 1))
        y = self.labels(rng, n)
        x, parts = self.features(rng, y, extra_noise)
        return x, y, parts


def build_dataset(cfg: dict) -> dict:
    d = cfg["data"]
    gen = Generator(d)
    rng = np.random.default_rng(d["seed"] + 1)
    feats = {m: [] for m in MODS}
    labels, dlg = [], []
    for i in range(d["n_dialogues"]):
        n = int(rng.integers(d["min_len"], d["max_len"] + 1))
        y = gen.labels(rng, n)
        x, _ = gen.features(rng, y)
        for m in MODS:
            feats[m].append(x[m])
        labels.append(y)
        dlg.append(np.full(n, i))
    perm = np.random.default_rng(d["seed"] + 2).permutation(d["n_dialogues"])
    n_tr = int(round(d["split"][0] * d["n_dialogues"]))
    n_va = int(round(d["split"][1] * d["n_dialogues"]))
    split = np.empty(d["n_dialogues"], dtype=np.int64)  # 0 train, 1 val, 2 test
    split[perm[:n_tr]] = 0
    split[perm[n_tr:n_tr + n_va]] = 1
    split[perm[n_tr + n_va:]] = 2
    ds = {m: np.concatenate(feats[m]) for m in MODS}
    ds["y"] = np.concatenate(labels)
    ds["dialogue"] = np.concatenate(dlg)
    ds["dialogue_split"] = split
    return ds


def dialogues_of(ds: dict, split_id: int):
    """List các hội thoại (dict a,v,t,y) của một split."""
    ids = np.where(ds["dialogue_split"] == split_id)[0]
    out = []
    for i in ids:
        idx = ds["dialogue"] == i
        out.append({**{m: ds[m][idx] for m in MODS}, "y": ds["y"][idx]})
    return out


def main(cfg_path="config.yaml", out="data/dataset.npz"):
    cfg = load_config(cfg_path)
    ds = build_dataset(cfg)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **ds)
    names = cfg["data"]["class_names"]
    for s, sname in enumerate(["train", "val", "test"]):
        nd = int((ds["dialogue_split"] == s).sum())
        mask = ds["dialogue_split"][ds["dialogue"]] == s
        counts = np.bincount(ds["y"][mask], minlength=len(names))
        print(f"{sname}: {nd} hội thoại, {int(mask.sum())} utterance, theo lớp "
              + ", ".join(f"{n}={c}" for n, c in zip(names, counts)))
    print("dims:", {m: ds[m].shape[1] for m in MODS}, "->", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
