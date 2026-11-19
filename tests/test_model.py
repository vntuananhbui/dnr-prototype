"""Kiểm các khối của phương pháp (không cần artifacts)."""
import torch

import data as data_mod
from model import DnR, aug_bundle, corr, divide_losses, info_nce, mask_bundle, refine_losses


def test_corr_values():
    x = torch.randn(200, 4)
    assert abs(float(corr(x, x)) - 1.0) < 1e-4
    assert abs(float(corr(x, -x)) + 1.0) < 1e-4


def test_mask_and_aug_bundle():
    h = torch.randn(2, 5, 3, 3, 8)
    m = mask_bundle(h, 1)
    assert torch.all(m[..., 0, :, :] == 0) and torch.all(m[..., 2, :, :] == 0)
    assert torch.equal(m[..., 1, :, :], h[..., 1, :, :])
    a = aug_bundle(h, 0.1)
    assert torch.equal(a[..., 0, :], h[..., 0, :]) and torch.equal(a[..., 2, :], h[..., 2, :])
    assert not torch.equal(a[..., 1, :], h[..., 1, :])


def test_info_nce_prefers_matching_pairs():
    a = torch.randn(32, 16)
    assert float(info_nce(a, a, 0.1)) < float(info_nce(a, torch.randn(32, 16), 0.1))


def test_forward_and_losses_shapes():
    cfg = data_mod.load_config()
    model = DnR(cfg)
    dims = cfg["data"]["dims"]
    x = {m: torch.randn(2, 6, dims[m]) for m in "avt"}
    lengths = torch.tensor([6, 4])
    logits, h, z = model(x, lengths)
    assert logits.shape == (2, 6, cfg["data"]["n_classes"]) and h.shape[2:] == (3, 3, cfg["model"]["d"])
    lu, lc = divide_losses(h.reshape(-1, 3, 3, h.shape[-1]), 0.5)
    assert torch.isfinite(lu) and torch.isfinite(lc)
    valid = torch.arange(6)[None] < lengths[:, None]
    lg, li, lm = refine_losses(model.backbone, h.detach(), lengths, valid, 2, 0.1, 0.1)
    assert torch.isfinite(li) and torch.isfinite(lm)


def test_dataset_split_and_shapes():
    cfg = data_mod.load_config()
    cfg["data"]["n_dialogues"] = 20
    ds = data_mod.build_dataset(cfg)
    assert sorted(set(ds["dialogue_split"].tolist())) == [0, 1, 2]
    for m in "avt":
        assert ds[m].shape == (len(ds["y"]), cfg["data"]["dims"][m])
    lens = [len(d["y"]) for s in range(3) for d in data_mod.dialogues_of(ds, s)]
    assert min(lens) >= cfg["data"]["min_len"] and max(lens) <= cfg["data"]["max_len"]
