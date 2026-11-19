# DnR prototype — Divide and Refine (arXiv 2601.14274), bản thu nhỏ

> **Bản thu nhỏ, dữ liệu mô phỏng, không phải model/số của paper.** Mức C: cài đúng thuật toán Divide → Refine
> ở cấu hình nhỏ, train trên CPU với dữ liệu hội thoại tự sinh.

Paper: *Divide and Refine: Enhancing Multimodal Representation and Explainability for Emotion Recognition in Conversation*,
arXiv 2601.14274 (đã đọc bản đầy đủ HTML).

**Kết quả trung thực:** trên dữ liệu mô phỏng này, khẳng định K1 (DnR tăng W-F1 so với backbone gốc) **chưa tái hiện được**: Divide+Refine thấp hơn baseline 0.37 điểm W-F1 (3 seed), chỉ Divide đơn lẻ tăng +0.41. Chạy `bash reproduce.sh` để sinh `results/metrics.json` và `report/index.html` (hai thư mục này sinh ra khi chạy, không commit trong bản này).

## Chạy

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
bash reproduce.sh        # dữ liệu -> train 4 cấu hình × 3 seed -> results/metrics.json -> report/index.html
bash run_demo.sh         # app: http://127.0.0.1:5001/?seed=7   (đổi cổng: PORT=5005 bash run_demo.sh)
.venv/bin/python -m pytest -q tests
```

`reproduce.sh`/`run_demo.sh` dùng `$PYTHON` nếu có, không thì `.venv/bin/python`, không thì `python3`.
Đo trên máy 48 nhân CPU, không GPU: `reproduce.sh` mất khoảng 107 giây (train chạy 12 run song song, 2 luồng mỗi run;
tổng CPU khoảng 24 phút, nên máy 4 nhân có thể lâu hơn nhiều — giảm `train.max_workers` hoặc epoch trong `config.yaml`).

## Các file

| File | Việc |
|---|---|
| `config.yaml` | Mọi thông số, mỗi dòng ghi nguồn (paper § / suy ra / chưa rõ), ngưỡng khớp chốt trước khi chạy |
| `data.py` | Sinh hội thoại mô phỏng: 6 lớp, a100/v128/t192, tín hiệu unique / redundant / synergistic (XOR a-v) + nhiễu; chia 70/10/20 theo hội thoại |
| `model.py` | Encoder tách U/R/S (Eq.4), g dùng chung (Eq.6), L_uncor (Eq.7), L_corr (Eq.8), bundle gốc/mask/aug (Eq.11), backbone concat+BiGRU (Eq.12), InfoNCE (Eq.13-14) |
| `train.py` | Phase I (Divide hoặc chỉ L_task) → đóng băng → Phase II (Refine hoặc chỉ L_task); lưu `artifacts/<cấu hình>_seed<k>.pt` |
| `metrics.py` | accuracy, weighted F1, F1 từng lớp, ma trận nhầm lẫn, mean±std (test tay: `tests/test_metrics.py`) |
| `predict.py` | Hội thoại → nhãn từng utterance, độ tin cậy, ‖U‖‖R‖‖S‖ theo modality, đóng góp từng luồng |
| `eval.py` | Test set → `results/metrics.json` (ghi file tạm rồi rename) |
| `make_report.py` | `report/index.html` + 7 PNG riêng (<100 KB), không CDN, không base64; số đọc từ metrics.json |
| `app.py` | Flask một trang, SVG vẽ phía server, không cần mạng |

## Địa chỉ của app (mỗi luồng một URL)

- `/?seed=7` — ví dụ mặc định, mở ra là thấy kết quả.
- `/?seed=7&noise_a=12&noise_v=0&noise_t=0` — làm nhiễu audio.
- `/?seed=42&noise_a=0&noise_v=12&noise_t=12` — mẫu khó: nhiễu video và text.
- `/api/predict?seed=7&noise_a=0&noise_v=0&noise_t=0` — JSON (baseline và DnR: nhãn, độ tin cậy, xác suất, năng lượng và đóng góp U/R/S).
- `/report/` — trang đánh giá.

`noise_*` là độ lệch chuẩn nhiễu cộng thêm cho từng modality (0–32; nhiễu nền là 8).

## Kết quả

Sau khi chạy `bash reproduce.sh`, xem `report/index.html` và `results/metrics.json` (mọi số tái hiện nằm ở đó, khoá `summary.<cấu hình>.wf1`,
`gain_vs_baseline`, `k1`). README không chép số tái hiện để tránh lệch với file kết quả.

Số của paper (chỉ để so, khác dữ liệu và backbone; W-F1 %, setting atv):

| Backbone, dataset | Gốc | +DnR | Nguồn |
|---|---|---|---|
| MMGCN, IEMOCAP | 66.70 | 67.96 | Tab.3 |
| DialogueGCN, IEMOCAP | 66.01 | 67.91 | Tab.3 |
| MMGCN, MELD | 58.78 | 59.94 | Tab.4 |
| DialogueGCN, MELD | 58.90 | 59.64 | Tab.4 |
| DialogueGCN + chỉ Divide, IEMOCAP | 66.01 | 65.46 | §5.3 |

## Khác paper

| Mục | Paper | Bản này |
|---|---|---|
| Dữ liệu | IEMOCAP / MELD thật | Mô phỏng 600 hội thoại |
| Đặc trưng | a100 / v512 / t768 | a100 / v128 / t192 |
| Backbone | MMGCN, DialogueGCN, MM-DFN, GraphSmile, SDT | late-fusion concat + BiGRU |
| Lịch train | không nói rõ baseline | cả 4 cấu hình đều 2 phase (ablation 2×2), encoder đóng băng ở phase II |
| α, λ1, λ2, K, Aug, τ, batch | không nêu | 0.5, 0.1, 0.1, 2, R+N(0,0.1²), 0.1, 32 (suy ra) |
| corr(·,·) | không nêu | Pearson theo từng chiều trên batch, trung bình |
| μ | dò trong [0.5, 2] nhưng không định nghĩa | bỏ |
| Phần cứng | 1 GPU L4 | CPU |

## Bản đầy đủ

Cần IEMOCAP (đăng ký với USC SAIL) và MELD, trích đặc trưng OpenSmile/OpenFace/sBERT (IEMOCAP) hoặc wav2vec/MA-Net/DeBERTa (MELD),
cài backbone GNN (MMGCN, DialogueGCN…). `config.yaml` có mục `paper_full` (dims, epoch cận trên). `model.pick_device()` tự dùng CUDA → MPS → CPU.
Ước lượng: một GPU cỡ L4, vài giờ cho mỗi backbone × dataset; chưa chạy ở đây.
