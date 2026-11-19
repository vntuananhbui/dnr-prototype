"""App demo Flask một trang: sinh hội thoại mô phỏng, dự đoán bằng baseline và DnR đã lưu, vẽ từng bước.

URL: /?seed=7&noise_a=0&noise_v=0&noise_t=0     (trang)
     /api/predict?seed=7&noise_a=0&noise_v=0&noise_t=0   (JSON)
     /report/                                     (trang đánh giá tĩnh)
"""
from __future__ import annotations

import html
import os

import numpy as np
from flask import Flask, abort, jsonify, request, send_from_directory

from predict import make_input, predict_dialogue

ROOT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(ROOT, "artifacts")
MODEL_BASE = os.environ.get("DNR_BASELINE", os.path.join(ART, "baseline_seed0.pt"))
MODEL_DNR = os.environ.get("DNR_MODEL", os.path.join(ART, "divide_refine_seed0.pt"))
LABEL = "Bản thu nhỏ, dữ liệu mô phỏng, không phải model/số của paper"
MAX_NOISE = 32.0

app = Flask(__name__)


def _params():
    try:
        seed = int(request.args.get("seed", 7))
        noise = {m: float(request.args.get(f"noise_{m}", 0.0)) for m in "avt"}
    except ValueError:
        abort(400, "seed phải là số nguyên, noise_* phải là số")
    for m, v in noise.items():
        if not (0.0 <= v <= MAX_NOISE):
            abort(400, f"noise_{m} phải trong [0, {MAX_NOISE}]")
    return seed, noise


def run(seed: int, noise: dict) -> dict:
    x, y, names = make_input(seed, noise, os.path.join(ROOT, "config.yaml"))
    base = predict_dialogue(MODEL_BASE, x)
    dnr = predict_dialogue(MODEL_DNR, x)
    return {
        "label": LABEL,
        "input": {"seed": seed, "noise": noise, "length": len(y), "true_labels": [names[i] for i in y],
                  "true": y.tolist(),
                  "preview": {m: np.round(x[m][:, :24], 2).tolist() for m in "avt"}},
        "baseline": base,
        "dnr": dnr,
        "accuracy": {"baseline": float(np.mean(np.asarray(base["pred"]) == y)),
                     "dnr": float(np.mean(np.asarray(dnr["pred"]) == y))},
    }


@app.get("/api/predict")
def api_predict():
    seed, noise = _params()
    try:
        return jsonify(run(seed, noise))
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503


@app.get("/report/")
@app.get("/report/<path:name>")
def report(name="index.html"):
    return send_from_directory(os.path.join(ROOT, "report"), name)


# ---------------- vẽ SVG phía server (không CDN, không JS) ----------------

def _color(v, vmax, pos=(217, 95, 2), neg=(49, 130, 189)):
    t = 0.0 if vmax <= 0 else max(-1.0, min(1.0, v / vmax))
    c = pos if t >= 0 else neg
    a = abs(t)
    r, g, b = (int(255 - (255 - ch) * a) for ch in c)
    return f"rgb({r},{g},{b})"


def svg_input(prev: dict, true_labels: list) -> str:
    L = len(true_labels)
    cw, ch, gap = 6, 14, 14
    w = 50 + 3 * (24 * cw + gap)
    out = [f'<svg width="{w}" height="{L * ch + 24}" role="img" aria-label="Đặc trưng đầu vào">']
    for j, m in enumerate("avt"):
        x0 = 50 + j * (24 * cw + gap)
        out.append(f'<text x="{x0}" y="12" font-size="11">{m} (24 chiều đầu)</text>')
        for i, row in enumerate(prev[m]):
            for k, v in enumerate(row):
                out.append(f'<rect x="{x0 + k * cw}" y="{18 + i * ch}" width="{cw}" height="{ch - 1}" fill="{_color(v, 20)}"/>')
    for i, lab in enumerate(true_labels):
        out.append(f'<text x="0" y="{18 + i * ch + 11}" font-size="10">{i}: {lab}</text>')
    out.append("</svg>")
    return "".join(out)


def svg_conf(res: dict, true: list, color: str) -> str:
    L = len(true)
    out = [f'<svg width="170" height="{L * 16 + 4}" role="img" aria-label="Độ tin cậy">']
    for i in range(L):
        c = res["confidence"][i]
        ok = res["pred"][i] == true[i]
        out.append(f'<rect x="0" y="{2 + i * 16}" width="{c * 100:.1f}" height="12" fill="{color}"/>')
        out.append(f'<text x="104" y="{12 + i * 16}" font-size="10" fill="{"#222" if ok else "#c00"}">'
                   f'{res["labels"][i]} {c:.2f}{"" if ok else " ✗"}</text>')
    out.append("</svg>")
    return "".join(out)


def svg_contrib(res: dict) -> str:
    C = np.asarray(res["contribution"])  # (L, 3, 3)
    L = C.shape[0]
    vmax = float(np.abs(C).max()) or 1.0
    cw, ch = 26, 16
    out = [f'<svg width="{9 * cw + 10}" height="{L * ch + 30}" role="img" aria-label="Đóng góp U R S">']
    for j, m in enumerate("avt"):
        for k, s in enumerate("URS"):
            out.append(f'<text x="{(j * 3 + k) * cw + 6}" y="12" font-size="10">{s}{m}</text>')
    for i in range(L):
        for j in range(3):
            for k in range(3):
                v = C[i, j, k]
                out.append(f'<rect x="{(j * 3 + k) * cw}" y="{18 + i * ch}" width="{cw - 2}" height="{ch - 2}" '
                           f'fill="{_color(v, vmax)}"><title>{v:+.3f}</title></rect>')
    out.append("</svg>")
    return "".join(out)


def svg_bars(res: dict, key: str, title: str) -> str:
    A = np.asarray(res[key]).mean(0)  # (3, 3)
    vmax = float(np.abs(A).max()) or 1.0
    out = [f'<svg width="300" height="150" role="img" aria-label="{html.escape(title)}">',
           f'<text x="0" y="12" font-size="11">{html.escape(title)}</text>']
    cols = {"U": "#fdae6b", "R": "#6baed6", "S": "#74c476"}
    for j, m in enumerate("avt"):
        for k, s in enumerate("URS"):
            v = A[j, k]
            h = abs(v) / vmax * 90
            x = 10 + j * 95 + k * 26
            y = 115 - h if v >= 0 else 115
            out.append(f'<rect x="{x}" y="{y:.1f}" width="22" height="{h:.1f}" fill="{cols[s]}"/>')
            out.append(f'<text x="{x}" y="128" font-size="9">{s}{m}</text>')
            out.append(f'<text x="{x - 2}" y="{(y - 2) if v >= 0 else (y + h + 10):.1f}" font-size="8">{v:.2f}</text>')
    out.append('<line x1="5" y1="115" x2="295" y2="115" stroke="#333"/></svg>')
    return "".join(out)


PAGE = """<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DnR demo</title><style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:0 auto;padding:16px;color:#222}}
.label{{background:#fff3cd;border:1px solid #e0c060;padding:8px 12px;border-radius:6px;font-weight:600}}
.grid{{display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start}} .card{{border:1px solid #ddd;border-radius:8px;padding:10px}}
form input{{width:70px}} h2{{font-size:17px;margin:18px 0 6px}} .acc{{font-size:15px}} code{{background:#f3f3f3}}
</style></head><body>
<h1>Divide and Refine (DnR) — demo nhận dạng cảm xúc trong hội thoại</h1>
<p class="label" id="level-label">{label}</p>
<details><summary><b>Paper này làm gì</b></summary><p>DnR (arXiv 2601.14274) tách biểu diễn mỗi modality thành U (riêng), R (trùng lặp giữa modality),
S (cộng hưởng) — bước Divide — rồi đóng bƒÉng chúng và huấn luyện backbone với các bundle gốc / che modality / nhiễu hoá R bằng InfoNCE — bước Refine.
App này sinh một hội thoại mô phỏng, chạy model baseline và model DnR đã train (artifacts/), và hiện từng bước.</p></details>
<p><a href="/report/">→ Trang đánh giá (so với paper)</a> · <a href="/api/predict?{qs}">JSON của ví dụ này</a></p>
<form method="get">seed <input name="seed" value="{seed}"> noise_a <input name="noise_a" value="{na}">
noise_v <input name="noise_v" value="{nv}"> noise_t <input name="noise_t" value="{nt}"> <button>Sinh &amp; chạy</button>
<small>noise_* = std nhiễu thêm vào từng modality (0-{maxn}; nhiễu nền là 8)</small></form>
<p class="acc">Hội thoại seed {seed}: {L} utterance. Đúng: baseline <b>{acc_b:.0%}</b>, DnR <b>{acc_d:.0%}</b>.</p>
<div class="grid">
<div class="card"><h2>1. Đầu vào (nhãn thật bên trái)</h2>{svg_in}</div>
<div class="card"><h2>2. Baseline: nhãn, độ tin cậy</h2>{svg_cb}</div>
<div class="card"><h2>3. DnR: nhãn, độ tin cậy</h2>{svg_cd}</div>
</div>
<div class="grid">
<div class="card"><h2>4. DnR — Divide: độ lớn ‖U‖,‖R‖,‖S‖ theo modality</h2>{svg_e}</div>
<div class="card"><h2>5. DnR — đóng góp trung bình mỗi luồng</h2>{svg_cm}
<p><small>Đóng góp = xác suất lớp dự đoán giảm bao nhiêu khi zero luồng đó trước backbone.</small></p></div>
<div class="card"><h2>6. DnR — đóng góp theo từng utterance</h2>{svg_c}<p><small>cam = giúp, xanh = ngược lại</small></p></div>
</div>
<p><small>Model: {mb} và {md} (nạp từ artifacts/, không train lại).</small></p>
</body></html>"""


@app.get("/")
def index():
    seed, noise = _params()
    try:
        r = run(seed, noise)
    except FileNotFoundError as e:
        return f"<p>{html.escape(str(e))}</p>", 503
    true = r["input"]["true"]
    qs = f"seed={seed}&noise_a={noise['a']}&noise_v={noise['v']}&noise_t={noise['t']}"
    return PAGE.format(
        label=LABEL, qs=html.escape(qs), seed=seed, na=noise["a"], nv=noise["v"], nt=noise["t"], maxn=MAX_NOISE,
        L=r["input"]["length"], acc_b=r["accuracy"]["baseline"], acc_d=r["accuracy"]["dnr"],
        svg_in=svg_input(r["input"]["preview"], r["input"]["true_labels"]),
        svg_cb=svg_conf(r["baseline"], true, "#8c8c8c"), svg_cd=svg_conf(r["dnr"], true, "#d95f02"),
        svg_e=svg_bars(r["dnr"], "energy", "trung bình trên các utterance"),
        svg_cm=svg_bars(r["dnr"], "contribution", "trung bình trên các utterance"),
        svg_c=svg_contrib(r["dnr"]), mb=os.path.basename(MODEL_BASE), md=os.path.basename(MODEL_DNR))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"DnR demo: http://127.0.0.1:{port}/?seed=7", flush=True)
    app.run(host="127.0.0.1", port=port, debug=False)
