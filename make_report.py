"""Sinh report/index.html + mỗi hình một PNG riêng (<100KB), không CDN, không base64.

Mọi số "Tái hiện" đọc từ results/metrics.json. Số "Paper" là trích dẫn (Tab.3/Tab.4, §5.3), khác điều kiện.
Chạy: python make_report.py
"""
from __future__ import annotations

import html
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import data as data_mod  # noqa: E402

OUT = "report"
DPI = 90
CONFIGS = ["baseline", "refine_only", "divide_only", "divide_refine"]
CFG_LABEL = {"baseline": "Baseline", "refine_only": "+Refine", "divide_only": "+Divide",
             "divide_refine": "+Divide+Refine (DnR)"}
COLORS = {"baseline": "#8c8c8c", "refine_only": "#6baed6", "divide_only": "#3182bd", "divide_refine": "#d95f02"}
# Trích từ paper (W-F1, setting atv). Không đưa vào metrics.json.
PAPER_GAINS = [
    ("MMGCN, IEMOCAP (Tab.3)", 66.70, 67.96),
    ("DialogueGCN, IEMOCAP (Tab.3)", 66.01, 67.91),
    ("MMGCN, MELD (Tab.4)", 58.78, 59.94),
    ("DialogueGCN, MELD (Tab.4)", 58.90, 59.64),
]
PAPER_DIVIDE_ONLY = ("DialogueGCN+Divide, IEMOCAP (§5.3)", 66.01, 65.46)


def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    size = os.path.getsize(path)
    assert size < 100_000, f"{name} quá lớn: {size} bytes"
    return name


def fig_method():
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.1, 4)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="#333", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=12, color="#333"))

    for i, (m, lab) in enumerate([("a", "audio 100d"), ("v", "video 128d"), ("t", "text 192d")]):
        y = 3.0 - i * 1.1
        box(0.1, y, 1.2, 0.7, lab, "#eeeeee")
        box(1.8, y, 1.3, 0.7, f"Encoder {m}", "#c6dbef")
        box(3.5, y, 1.5, 0.7, f"U{m} | R{m} | S{m}", "#fdd0a2")
        arrow(1.3, y + 0.35, 1.8, y + 0.35)
        arrow(3.1, y + 0.35, 3.5, y + 0.35)
        arrow(5.0, y + 0.35, 5.6, 1.95)
    ax.text(3.0, 3.85, "Phase I — Divide: L_task (Σ g(U,R,S)) + 1.0·L_uncor + 0.5·L_corr", fontsize=9, ha="center")
    box(5.6, 1.4, 1.6, 1.1, "Bundles:\ngốc, mask m,\naug R+N(0,σ²)", "#fee6ce")
    box(7.6, 1.4, 1.2, 1.1, "Backbone\nconcat +\nBiGRU → Z", "#c7e9c0")
    box(9.05, 1.65, 0.85, 0.6, "nhãn", "#eeeeee")
    arrow(7.2, 1.95, 7.6, 1.95)
    arrow(8.8, 1.95, 9.05, 1.95)
    ax.text(7.4, 0.45, "Phase II — Refine (Divide đóng băng):\nL_task + λ1·InfoNCE(Z_aug^k, Z_aug^n) + λ2·InfoNCE(Z_aug, Z_m)",
            fontsize=9, ha="center")
    return savefig(fig, "fig_method.png")


def fig_data(cfg):
    gen = data_mod.Generator(cfg["data"])
    x, y, parts = gen.dialogue(3)
    names = cfg["data"]["class_names"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), gridspec_kw={"width_ratios": [0.5, 3, 3, 3]})
    axes[0].imshow(y[:, None], aspect="auto", cmap="tab10", vmin=0, vmax=9)
    axes[0].set_yticks(range(len(y)))
    axes[0].set_yticklabels([names[i] for i in y], fontsize=7)
    axes[0].set_xticks([])
    axes[0].set_title("nhãn", fontsize=9)
    for ax, m in zip(axes[1:], ["a", "v", "t"]):
        ax.imshow(x[m][:, :40], aspect="auto", cmap="RdBu_r", vmin=-20, vmax=20)
        ax.set_title(f"modality {m}: 40 chiều đầu (có nhiễu)", fontsize=9)
        ax.set_yticks([])
        ax.set_xlabel("chiều đặc trưng", fontsize=8)
    fig.suptitle("Một hội thoại mô phỏng (seed 3): mỗi hàng là 1 utterance", fontsize=10, y=1.06)
    return savefig(fig, "fig_data.png")


def fig_compare(M):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, c in enumerate(CONFIGS):
        s = M["summary"][c]["wf1"]
        ax.bar(i, 100 * s["mean"], yerr=100 * s["std"], color=COLORS[c], capsize=4)
        pts = [100 * r["wf1"] for r in M["runs"][c].values()]
        ax.scatter([i] * len(pts), pts, color="k", s=10, zorder=3)
        ax.text(i, 100 * s["mean"] + 100 * s["std"] + 0.15, f"{100 * s['mean']:.2f}", ha="center", fontsize=9)
    vals = [100 * M["summary"][c]["wf1"]["mean"] for c in CONFIGS]
    ax.set_ylim(min(vals) - 2, max(vals) + 1.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([CFG_LABEL[c] for c in CONFIGS], fontsize=9)
    ax.set_ylabel("Weighted F1 (%) trên test")
    ax.set_title("Tái hiện: W-F1 của 4 cấu hình (cột = trung bình, chấm = từng seed)\n"
                 "trục y không bắt đầu từ 0 để thấy chênh lệch nhỏ", fontsize=9)
    return savefig(fig, "fig_compare.png")


def fig_gain(M):
    rows = [(f"Tái hiện: {CFG_LABEL[c]}", M["gain_vs_baseline"][c]["wf1_points_mean"], COLORS[c])
            for c in ["divide_refine", "divide_only", "refine_only"]]
    rows += [(f"Paper: {n}", b - a, "#636363") for n, a, b in PAPER_GAINS]
    n, a, b = PAPER_DIVIDE_ONLY
    rows.append((f"Paper: {n}", b - a, "#bdbdbd"))
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (lab, v, col) in enumerate(rows):
        ax.barh(i, v, color=col)
        ax.text(v + (0.05 if v >= 0 else -0.05), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    lo = min(r[1] for r in rows)
    hi = max(r[1] for r in rows)
    ax.set_xlim(lo - 0.6, hi + 0.6)
    ax.set_xlabel("Chênh W-F1 so với backbone gốc (điểm %)")
    ax.set_title("K1: DnR có tăng W-F1 không? Tái hiện (dữ liệu mô phỏng) cạnh Paper (IEMOCAP/MELD)", fontsize=9)
    return savefig(fig, "fig_gain.png")


def fig_confusion(M, names):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, c in zip(axes, ["baseline", "divide_refine"]):
        cm = np.sum([np.asarray(r["confusion"]) for r in M["runs"][c].values()], 0).astype(float)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("đoán")
        ax.set_ylabel("thật")
        ax.set_title(f"{CFG_LABEL[c]} (gộp 3 seed, chuẩn hoá theo hàng)", fontsize=9)
    return savefig(fig, "fig_confusion.png")


def fig_robust(M):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for c in CONFIGS:
        r = M["summary"][c]["robustness_wf1"]
        es = sorted(r, key=float)
        ax.errorbar([float(e) for e in es], [100 * r[e]["mean"] for e in es], yerr=[100 * r[e]["std"] for e in es],
                    marker="o", color=COLORS[c], label=CFG_LABEL[c], capsize=3)
    ax.set_xlabel("nhiễu thêm lúc test: std tăng thêm e (cả 3 modality)")
    ax.set_ylabel("W-F1 (%)")
    ax.set_title("Độ bền với nhiễu (phân tích thêm, paper không có hình này)", fontsize=9)
    ax.legend(fontsize=8)
    return savefig(fig, "fig_robust.png")


def fig_occlusion(M):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    vmax = max(np.abs(np.asarray(M["summary"][c]["occlusion_true_prob_drop"])).max()
               for c in ["baseline", "divide_refine"])
    for ax, c in zip(axes, ["baseline", "divide_refine"]):
        o = np.asarray(M["summary"][c]["occlusion_true_prob_drop"])
        ax.imshow(o, cmap="Oranges", vmin=0, vmax=vmax)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{o[i, j]:.3f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["U", "R", "S"])
        ax.set_yticks(range(3))
        ax.set_yticklabels(["audio", "video", "text"])
        ax.set_title(f"{CFG_LABEL[c]}", fontsize=9)
    fig.suptitle("Xác suất lớp đúng giảm bao nhiêu khi zero từng luồng (trung bình test, 3 seed)", fontsize=9, y=1.04)
    return savefig(fig, "fig_occlusion.png")


def pct(x):
    return f"{100 * x:.2f}"


def build_html(M, figs):
    cfg = M["config"]
    names = cfg["data"]["class_names"]
    S = M["summary"]
    esc = html.escape
    rows = "".join(
        f"<tr><td>{CFG_LABEL[c]}</td><td>{pct(S[c]['acc']['mean'])} ± {pct(S[c]['acc']['std'])}</td>"
        f"<td>{pct(S[c]['wf1']['mean'])} ± {pct(S[c]['wf1']['std'])}</td>"
        f"<td>{M['gain_vs_baseline'][c]['wf1_points_mean']:+.2f}</td>"
        f"<td>{', '.join(f'{v:+.2f}' for v in M['gain_vs_baseline'][c]['wf1_points_per_seed'])}</td>"
        f"<td>{M['gain_vs_baseline'][c]['seeds_better_than_baseline']}/{S[c]['wf1']['n']}</td>"
        f"<td><code>summary.{c}.wf1</code></td></tr>" for c in CONFIGS)
    seed_rows = "".join(
        f"<tr><td>{CFG_LABEL[c]}</td><td>{s}</td><td>{pct(r['acc'])}</td><td>{pct(r['wf1'])}</td>"
        f"<td>{r['best_val']['epoch']}</td><td>{r['train_seconds']}</td></tr>"
        for c in CONFIGS for s, r in M["runs"][c].items())
    pcf = "".join(
        f"<tr><td>{CFG_LABEL[c]}</td>" + "".join(
            f"<td>{pct(np.mean([r['per_class_f1'][k] for r in M['runs'][c].values()]))}</td>" for k in range(len(names)))
        + "</tr>" for c in CONFIGS)
    diag = "".join(
        f"<tr><td>{CFG_LABEL[c]}</td><td>{S[c]['divide_diag']['L_uncor']['mean']:.3f}</td>"
        f"<td>{S[c]['divide_diag']['L_corr']['mean']:.3f}</td></tr>" for c in CONFIGS)
    paper_rows = "".join(f"<tr><td>{esc(n)}</td><td>{a:.2f}</td><td>{b:.2f}</td><td>{b - a:+.2f}</td></tr>"
                         for n, a, b in PAPER_GAINS + [PAPER_DIVIDE_ONLY])
    k1 = M["k1"]
    dvo = M["gain_vs_baseline"]["divide_only"]["wf1_points_mean"]
    rfo = M["gain_vs_baseline"]["refine_only"]["wf1_points_mean"]
    diffs = [
        ("Dữ liệu", "IEMOCAP/MELD thật (§5.1, Tab.1-2)", "Mô phỏng: 600 hội thoại, U/R/S cài sẵn + nhiễu", "Không so được số tuyệt đối với paper"),
        ("Đặc trưng", "a100 / v512 / t768 (IEMOCAP)", "a100 / v128 / t192", "Model nhỏ hơn"),
        ("Backbone", "MMGCN, DialogueGCN, MM-DFN, GraphSmile, SDT", "Late-fusion concat + BiGRU", "Không có GNN theo người nói"),
        ("Lịch train", "Phase I/II; không nói rõ baseline train thế nào", "Cả 4 cấu hình đều 2 phase, encoder đóng băng ở phase II (ablation 2×2)", "Baseline yếu hơn baseline end-to-end có thể"),
        ("α, λ1, λ2, K, Aug, τ", "Không nêu", "α=0.5, λ1=λ2=0.1, K=2, Aug = R+N(0,0.1²), τ=0.1", "Chưa dò tham số; có thể chưa tối ưu"),
        ("corr(·,·)", "Không nêu cách tính", "Pearson theo từng chiều trên batch, trung bình các chiều", "Có thể khác hiện thực của tác giả"),
        ("Batch", "Không nêu", "32 hội thoại", "—"),
        ("Epoch", "Divide 30-80, Refine 20-50 (§5.1)", f"{cfg['train']['phase1_epochs']} / {cfg['train']['phase2_epochs']}", "—"),
        ("μ", "Dò trong [0.5, 2.0] nhưng không định nghĩa", "Bỏ (không biết là gì)", "Có thể thiếu một thành phần"),
        ("Seed", "Không nêu", f"{len(cfg['train']['seeds'])} seed", "—"),
        ("Phần cứng", "1 GPU NVIDIA L4", "CPU", "Chỉ ảnh hưởng thời gian"),
    ]
    diff_rows = "".join(f"<tr><td>{esc(a)}</td><td>{esc(b)}</td><td>{esc(c)}</td><td>{esc(d)}</td></tr>"
                        for a, b, c, d in diffs)
    css = """body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:980px;margin:0 auto;padding:16px;color:#222;line-height:1.5}
h1{font-size:24px}h2{font-size:19px;margin-top:32px;border-bottom:1px solid #ddd}img{max-width:100%}
table{border-collapse:collapse;font-size:13px;margin:8px 0;font-variant-numeric:tabular-nums}td,th{border:1px solid #ccc;padding:4px 8px}
th{background:#f3f3f3}.label{background:#fff3cd;border:1px solid #e0c060;padding:8px 12px;border-radius:6px}
.verdict{font-size:18px;font-weight:600}code{background:#f3f3f3;padding:0 3px}"""
    return f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DnR — trang đánh giá bản thu nhỏ</title><style>{css}</style></head><body>
<h1>Divide and Refine (DnR, arXiv 2601.14274) — trang đánh giá</h1>
<p class="label"><b>{esc(M['meta']['label'])}.</b> Mức C. Mọi số "Tái hiện" đọc từ <code>results/metrics.json</code>
(tạo lúc {esc(M['meta']['created'])}); số "Paper" là trích dẫn, khác dữ liệu và backbone.</p>

<h2>1. Paper làm gì</h2>
<p>DnR tách biểu diễn mỗi modality (âm thanh, hình, chữ) thành ba phần: <b>U</b> riêng của modality, <b>R</b> trùng lặp giữa các modality
và <b>S</b> cộng hưởng (chỉ có ích khi kết hợp) — bước <b>Divide</b> (§4.1). Bước <b>Refine</b> (§4.2) đóng băng phần Divide, tạo các bundle
(gốc, che bớt modality, nhiễu hoá R) và dùng InfoNCE để backbone nhận dạng cảm xúc học biểu diễn bền hơn. Paper báo tăng W-F1 trên IEMOCAP và MELD cho nhiều backbone.</p>
<img src="{figs['method']}" alt="Sơ đồ Divide rồi Refine">

<h2>2. Dữ liệu trông thế nào</h2>
<p>Dữ liệu tự sinh (không dùng IEMOCAP/MELD vì cần đăng ký và trích đặc trưng nặng): {cfg['data']['n_dialogues']} hội thoại, mỗi hội thoại
{cfg['data']['min_len']}-{cfg['data']['max_len']} utterance, {cfg['data']['n_classes']} lớp ({', '.join(names)}), chia 70/10/20 theo hội thoại.
Test có {M['meta']['n_test_dialogues']} hội thoại, {M['meta']['n_test_utterances']} utterance. Nhãn được cài theo 3 kiểu: unique (a → nhóm c//3, t → c%3, v → c%2),
redundant (cả 3 modality chứa prototype lớp), synergistic (dấu ngẫu nhiên z ở a và q<sub>c</sub>·z ở v — kiểu XOR), cộng nhiễu Gaussian std {cfg['data']['noise_std']}.</p>
<img src="{figs['data']}" alt="Heatmap đặc trưng một hội thoại">

<h2>3. Kết quả chính</h2>
<img src="{figs['compare']}" alt="So sánh W-F1 bốn cấu hình">
<table><tr><th>Cấu hình</th><th>Acc (%)</th><th>W-F1 (%)</th><th>Δ W-F1 so baseline (điểm)</th><th>Δ từng seed</th><th>Seed hơn baseline</th><th>Khoá</th></tr>{rows}</table>
<img src="{figs['gain']}" alt="Chênh W-F1 tái hiện cạnh paper">
<p>Số Paper (W-F1 %, setting atv; trích, không có trong metrics.json):</p>
<table><tr><th>Backbone, dataset (bảng)</th><th>Gốc</th><th>+DnR / +Divide</th><th>Chênh</th></tr>{paper_rows}</table>

<h2>4. Phân tích lỗi</h2>
<img src="{figs['confusion']}" alt="Ma trận nhầm lẫn">
<p>F1 từng lớp (%, trung bình 3 seed, <code>runs.*.per_class_f1</code>):</p>
<table><tr><th>Cấu hình</th>{''.join(f'<th>{n}</th>' for n in names)}</tr>{pcf}</table>
<img src="{figs['occlusion']}" alt="Đóng góp từng luồng U R S">
<p>Chẩn đoán Divide trên test (<code>summary.*.divide_diag</code>; L_uncor = Σ|corr(U,R)| càng nhỏ càng tách; L_corr càng âm thì R/S giữa modality càng tương quan):</p>
<table><tr><th>Cấu hình</th><th>L_uncor</th><th>L_corr</th></tr>{diag}</table>
<img src="{figs['robust']}" alt="Độ bền với nhiễu">

<h2>5. Bảng số từng seed</h2>
<table><tr><th>Cấu hình</th><th>Seed</th><th>Acc (%)</th><th>W-F1 (%)</th><th>Epoch chọn (val)</th><th>Giây train</th></tr>{seed_rows}</table>
<p>Thời gian train (đồng hồ, chạy song song): {M['meta']['train_wall_seconds']} s (<code>meta.train_wall_seconds</code>).</p>

<h2>6. Khác paper</h2>
<table><tr><th>Mục</th><th>Paper</th><th>Bản này</th><th>Ảnh hưởng dự kiến</th></tr>{diff_rows}</table>

<h2>7. Kết luận theo ngưỡng (chốt trước khi chạy)</h2>
<p>K1 — "{esc(k1['claim'])}". Đại lượng đo: {esc(k1['measured'])}; {esc(k1['check_type'])}, nên tối đa là "Khớp một phần".
Ngưỡng: ≥ {k1['threshold_partial']} điểm → Khớp một phần; nhỏ hơn → Chưa tái hiện được.</p>
<p class="verdict">K1: chênh = {k1['gain_points']:+.2f} điểm (<code>k1.gain_points</code>) → {esc(k1['verdict'])}</p>
<p>Phụ (paper §5.3: Divide đơn lẻ có thể làm giảm): ở đây +Divide {dvo:+.2f} điểm, +Refine {rfo:+.2f} điểm
(<code>gain_vs_baseline.*.wf1_points_mean</code>). Kết quả trên IEMOCAP/MELD thật với backbone GNN: <b>Chưa tái hiện được — cần bộ dữ liệu có đăng ký,
trích đặc trưng và GPU</b>.</p>
</body></html>"""


def main():
    with open(os.path.join("results", "metrics.json"), encoding="utf-8") as f:
        M = json.load(f)
    os.makedirs(OUT, exist_ok=True)
    cfg = M["config"]
    figs = {
        "method": fig_method(), "data": fig_data(cfg), "compare": fig_compare(M), "gain": fig_gain(M),
        "confusion": fig_confusion(M, cfg["data"]["class_names"]), "robust": fig_robust(M), "occlusion": fig_occlusion(M),
    }
    tmp = os.path.join(OUT, "index.html.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(build_html(M, figs))
    os.replace(tmp, os.path.join(OUT, "index.html"))
    for n in sorted(os.listdir(OUT)):
        print(f"{OUT}/{n}: {os.path.getsize(os.path.join(OUT, n))} bytes")


if __name__ == "__main__":
    main()
