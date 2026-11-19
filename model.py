"""Model DnR thu nhỏ: Divide (encoder tách U/R/S) + backbone late-fusion BiGRU + loss Refine."""
from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

MODS = ("a", "v", "t")
STREAMS = ("U", "R", "S")


def pick_device() -> torch.device:
    """CUDA -> MPS -> CPU (không viết cứng loại GPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ModalityEncoder(nn.Module):
    """x_m -> h_m = (U, R, S), mỗi luồng d chiều (paper Eq.4)."""

    def __init__(self, in_dim: int, hidden: int, d: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 3 * d)
        )
        self.d = d

    def forward(self, x):
        h = self.net(x)
        return h.view(*h.shape[:-1], 3, self.d)  # (..., 3 luồng U,R,S, d)


class DivideModel(nn.Module):
    """Các encoder theo modality + predictor nhẹ g dùng chung (paper Eq.6)."""

    def __init__(self, dims: dict, hidden: int, d: int, n_classes: int, dropout: float):
        super().__init__()
        self.enc = nn.ModuleDict({m: ModalityEncoder(dims[m], hidden, d, dropout) for m in MODS})
        self.g = nn.Linear(d, n_classes)

    def encode(self, x: dict) -> torch.Tensor:
        """Trả tensor (..., 3 modality, 3 luồng, d)."""
        return torch.stack([self.enc[m](x[m]) for m in MODS], dim=-3)

    def stream_logits(self, h: torch.Tensor) -> torch.Tensor:
        """g áp lên từng luồng: (..., 3, 3, C)."""
        return self.g(h)

    def forward(self, x: dict):
        h = self.encode(x)
        logits = self.stream_logits(h).sum(dim=(-3, -2))  # ŷ = Σ_m g(U)+g(R)+g(S)
        return logits, h


class Backbone(nn.Module):
    """F_prev: late-fusion (concat bundle) + BiGRU trên chuỗi hội thoại -> classifier."""

    def __init__(self, d: int, gru_hidden: int, n_classes: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(9 * d, gru_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.gru = nn.GRU(gru_hidden, gru_hidden, batch_first=True, bidirectional=True)
        self.cls = nn.Linear(2 * gru_hidden, n_classes)

    def forward(self, bundle: torch.Tensor, lengths: torch.Tensor):
        """bundle (B, T, 3, 3, d) -> (logits (B,T,C), Z (B,T,2H))."""
        B, T = bundle.shape[:2]
        x = self.proj(bundle.reshape(B, T, -1))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        z, _ = pad_packed_sequence(out, batch_first=True, total_length=T)
        return self.cls(z), z


# ---------------- loss Divide (paper Eq.7, Eq.8) ----------------

def corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Tương quan Pearson theo từng chiều trên batch, lấy trung bình các chiều.
    suy ra: paper không nêu cách tính corr giữa hai embedding."""
    xc = x - x.mean(0, keepdim=True)
    yc = y - y.mean(0, keepdim=True)
    num = (xc * yc).sum(0)
    den = xc.pow(2).sum(0).sqrt() * yc.pow(2).sum(0).sqrt() + eps
    return (num / den).mean()


def divide_losses(h: torch.Tensor, alpha: float):
    """h: (N, 3 modality, 3 luồng, d). Trả (L_uncor, L_corr)."""
    U, R, S = h[:, :, 0], h[:, :, 1], h[:, :, 2]
    l_uncor = sum(corr(U[:, m], R[:, m]).abs() for m in range(3))
    pairs = list(itertools.combinations(range(3), 2))  # m≠s, cặp không thứ tự (đối xứng)
    l_corr = -sum(corr(R[:, m], R[:, s]) for m, s in pairs)
    l_corr = l_corr - sum(corr(S[:, m], S[:, s]) for m, s in pairs)
    l_corr = l_corr - alpha * sum(corr(S[:, m], U[:, m]) for m in range(3))
    return l_uncor, l_corr


# ---------------- Refine (paper Eq.11-15) ----------------

def mask_bundle(h: torch.Tensor, keep: int) -> torch.Tensor:
    """Giữ modality `keep`, zero các modality khác. h (..., 3, 3, d)."""
    m = torch.zeros(3, device=h.device, dtype=h.dtype)
    m[keep] = 1.0
    return h * m.view(3, 1, 1)


def aug_bundle(h: torch.Tensor, sigma: float, gen: torch.Generator | None = None) -> torch.Tensor:
    """Thay R bằng R + N(0, sigma^2), giữ U, S."""
    noise = torch.randn(h[..., 1, :].shape, generator=gen, device=h.device, dtype=h.dtype) * sigma
    out = h.clone()
    out[..., 1, :] = out[..., 1, :] + noise
    return out


def info_nce(a: torch.Tensor, b: torch.Tensor, tau: float) -> torch.Tensor:
    """InfoNCE đối xứng hai chiều; cặp dương là cùng utterance. a, b: (N, k)."""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.t() / tau
    target = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


def refine_losses(backbone: Backbone, h: torch.Tensor, lengths: torch.Tensor, valid: torch.Tensor,
                  K: int, sigma: float, tau: float):
    """Trả (logits gốc, L_aug_intra, L_aug_mask). h (B,T,3,3,d) đã đóng băng."""
    logits, _ = backbone(h, lengths)
    z_aug = [backbone(aug_bundle(h, sigma), lengths)[1][valid] for _ in range(K)]
    z_mask = [backbone(mask_bundle(h, m), lengths)[1][valid] for m in range(3)]
    l_intra = sum(info_nce(z_aug[k], z_aug[n], tau) for k, n in itertools.combinations(range(K), 2))
    l_mask = sum(info_nce(z_aug[k], z_mask[m], tau) for m in range(3) for k in range(K))
    return logits, l_intra, l_mask


class DnR(nn.Module):
    """Gói cả hai phần để lưu/nạp và dự đoán."""

    def __init__(self, cfg: dict):
        super().__init__()
        dc, mc = cfg["data"], cfg["model"]
        self.divide = DivideModel(dc["dims"], mc["enc_hidden"], mc["d"], dc["n_classes"], mc["dropout"])
        self.backbone = Backbone(mc["d"], mc["gru_hidden"], dc["n_classes"], mc["dropout"])

    def forward(self, x: dict, lengths: torch.Tensor):
        h = self.divide.encode(x)
        logits, z = self.backbone(h, lengths)
        return logits, h, z
