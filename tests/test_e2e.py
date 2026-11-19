"""End-to-end: predict với model đã lưu, và API của app (khởi động app thật, gọi, tắt).
Cần chạy ./reproduce.sh trước (để có artifacts/)."""
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import pytest
import requests

from predict import make_input, predict_dialogue

MODEL = os.path.join("artifacts", "divide_refine_seed0.pt")
pytestmark = pytest.mark.skipif(not os.path.exists(MODEL), reason="chưa có artifacts/: chạy ./reproduce.sh")


@pytest.mark.parametrize("seed,noise", [(7, {}), (11, {"a": 16.0, "v": 0.0, "t": 16.0})])
def test_predict_runs_on_saved_model(seed, noise):
    x, y, names = make_input(seed, noise)
    r = predict_dialogue(MODEL, x)
    L = len(y)
    assert len(r["pred"]) == L and len(r["labels"]) == L
    assert all(0.0 <= c <= 1.0 for c in r["confidence"])
    assert np.allclose(np.asarray(r["probs"]).sum(1), 1.0, atol=1e-5)
    assert np.asarray(r["energy"]).shape == (L, 3, 3)
    assert np.asarray(r["contribution"]).shape == (L, 3, 3)
    assert set(r["labels"]) <= set(names)


def test_predict_clean_better_than_chance():
    accs = []
    for s in range(5):
        x, y, _ = make_input(100 + s)
        accs.append(np.mean(np.asarray(predict_dialogue(MODEL, x)["pred"]) == y))
    assert np.mean(accs) > 1 / 6 + 0.2


def test_metrics_json_structure():
    with open(os.path.join("results", "metrics.json"), encoding="utf-8") as f:
        m = json.load(f)
    for c in ["baseline", "refine_only", "divide_only", "divide_refine"]:
        assert set(m["summary"][c]["wf1"]) == {"mean", "std", "n"}
        assert len(m["runs"][c]) >= 3
    assert m["k1"]["verdict"] in ("Khớp một phần", "Chưa tái hiện được")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    env = {**os.environ, "PORT": str(port)}
    proc = subprocess.Popen([sys.executable, "app.py"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            requests.get(base + "/api/predict?seed=1", timeout=2)
            break
        except requests.ConnectionError:
            time.sleep(0.2)
    yield base
    proc.terminate()
    proc.wait(timeout=10)


def test_api_structure(server):
    r = requests.get(server + "/api/predict?seed=7&noise_a=0&noise_v=4&noise_t=0", timeout=10)
    assert r.status_code == 200
    j = r.json()
    assert "không phải model/số của paper" in j["label"]
    L = j["input"]["length"]
    for k in ("baseline", "dnr"):
        assert len(j[k]["labels"]) == L and len(j[k]["confidence"]) == L
        assert len(j[k]["contribution"]) == L
    assert j["input"]["noise"] == {"a": 0.0, "v": 4.0, "t": 0.0}


def test_page_shows_example_and_label(server):
    r = requests.get(server + "/?seed=7&noise_a=8&noise_v=0&noise_t=0", timeout=10)
    assert r.status_code == 200
    assert "Bản thu nhỏ, dữ liệu mô phỏng" in r.text and "<svg" in r.text and "DnR" in r.text
    r0 = requests.get(server + "/", timeout=10)
    assert r0.status_code == 200 and "<svg" in r0.text


def test_api_rejects_bad_input(server):
    assert requests.get(server + "/api/predict?seed=x", timeout=10).status_code == 400
    assert requests.get(server + "/api/predict?noise_a=-1", timeout=10).status_code == 400


def test_report_served(server):
    r = requests.get(server + "/report/", timeout=10)
    assert r.status_code == 200 and "fig_compare.png" in r.text
