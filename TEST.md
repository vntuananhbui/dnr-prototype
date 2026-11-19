# Lệnh test chung

```bash
# từ thư mục gốc repo, sau ./reproduce.sh (test e2e cần artifacts/ và results/)
PYTHON=<python của venv> ./reproduce.sh
<python của venv> -m pytest --collect-only -q    # phải 19 tests, không lỗi import
<python của venv> -m pytest -q tests
```

Gói: đúng `requirements.txt` (torch bản CPU qua `--extra-index-url https://download.pytorch.org/whl/cpu`). Không cài thêm gói nào.
Không có test bị bỏ qua khi đã chạy `reproduce.sh`; nếu chưa có `artifacts/` thì `tests/test_e2e.py` bị skip (có lý do in ra).
