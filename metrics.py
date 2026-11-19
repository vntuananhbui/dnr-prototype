"""Toàn bộ cách tính metric (paper §5.1 dùng accuracy và weighted F1)."""
from __future__ import annotations

import math

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """cm[i, j] = số mẫu nhãn thật i bị đoán là j."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        cm[int(t), int(p)] += 1
    return cm


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def per_class_f1(y_true, y_pred, n_classes: int) -> np.ndarray:
    """F1 từng lớp; lớp có precision+recall = 0 thì F1 = 0."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    pred_pos = cm.sum(0).astype(float)
    true_pos = cm.sum(1).astype(float)
    prec = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    rec = np.divide(tp, true_pos, out=np.zeros_like(tp), where=true_pos > 0)
    denom = prec + rec
    return np.divide(2 * prec * rec, denom, out=np.zeros_like(tp), where=denom > 0)


def weighted_f1(y_true, y_pred, n_classes: int) -> float:
    """Trung bình F1 từng lớp, trọng số = số mẫu thật của lớp."""
    y_true = np.asarray(y_true)
    support = np.bincount(y_true, minlength=n_classes).astype(float)
    if support.sum() == 0:
        return float("nan")
    return float((per_class_f1(y_true, y_pred, n_classes) * support).sum() / support.sum())


def mean_std(values) -> dict:
    """Trung bình và độ lệch chuẩn mẫu (ddof=1; 1 giá trị thì std = 0)."""
    v = [float(x) for x in values]
    n = len(v)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    mu = sum(v) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return {"mean": mu, "std": sd, "n": n}
