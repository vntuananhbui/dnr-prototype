"""So metric với giá trị tính tay."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics  # noqa: E402

# Ví dụ tính tay: 3 lớp, 6 mẫu
Y_TRUE = [0, 0, 1, 1, 1, 2]
Y_PRED = [0, 1, 1, 1, 0, 2]
# lớp 0: tp=1, đoán là 0: 2 mẫu -> P=1/2, có 2 mẫu thật -> R=1/2, F1=1/2
# lớp 1: tp=2, đoán là 1: 3 -> P=2/3, thật 3 -> R=2/3, F1=2/3
# lớp 2: tp=1, P=1, R=1, F1=1
# weighted F1 = (2*1/2 + 3*2/3 + 1*1) / 6 = (1 + 2 + 1)/6 = 4/6


def test_confusion_matrix():
    cm = metrics.confusion_matrix(Y_TRUE, Y_PRED, 3)
    assert cm.tolist() == [[1, 1, 0], [1, 2, 0], [0, 0, 1]]


def test_accuracy():
    assert metrics.accuracy(Y_TRUE, Y_PRED) == 4 / 6


def test_per_class_f1():
    np.testing.assert_allclose(metrics.per_class_f1(Y_TRUE, Y_PRED, 3), [0.5, 2 / 3, 1.0])


def test_weighted_f1():
    assert math.isclose(metrics.weighted_f1(Y_TRUE, Y_PRED, 3), 4 / 6)


def test_class_never_predicted_has_f1_zero():
    # lớp 1 không bao giờ được đoán: F1(1)=0; lớp 0: P=1/2,R=1 -> 2/3; weighted=(1*2/3+1*0)/2=1/3
    assert math.isclose(metrics.weighted_f1([0, 1], [0, 0], 2), 1 / 3)


def test_mean_std():
    r = metrics.mean_std([1.0, 2.0, 3.0])
    assert r["mean"] == 2.0 and math.isclose(r["std"], 1.0) and r["n"] == 3
    assert metrics.mean_std([5.0])["std"] == 0.0
