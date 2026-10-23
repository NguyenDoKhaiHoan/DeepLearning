# RT-DETRv2 + P2 trên VisDrone

Dự án Computer Vision giai đoạn 2: [Traffic MOT dưới che khuất](traffic-mot/README.md).
Ưu tiên hoàn thiện đề tài CV: detection đã chạy thật CPU; Kaggle/standard MOT còn
cần nghiệm thu. Kế hoạch, source và bằng chứng nằm trong `traffic-mot/`.

Notebook inference mới: [01_detection_inference.ipynb](notebooks/kaggle/01_detection_inference.ipynb).
Xem [hướng dẫn clone và chuẩn bị Kaggle](notebooks/kaggle/README.md) trước khi chạy.

Web demo mới **TrafficVision**: chạy `python web_demo/run.py`, mở
**http://127.0.0.1:8000**. Giao diện React có ảnh/video, bộ lọc, xuất kết quả,
dashboard metrics và responsive mobile. Xem [hướng dẫn web demo](web_demo/README.md).

Demo dùng checkpoint E1_p2 cuối cùng trong `artifacts/E1_p2_final/`. Chạy
`python -m streamlit run rtdetrv2_next/demo_app.py`; đường dẫn model đã điền sẵn.
Xem [hướng dẫn demo](docs/DEMO_SETUP.md) để chạy ảnh/video và kiểm tra nguồn artifact.

Notebook và công cụ train RT-DETRv2-R18 + P2 trên Kaggle, so sánh với control E0,
và đánh giá đối tượng nhỏ / che khuất trên VisDrone đã chuyển đổi sang COCO.

## Chạy trên Kaggle

1. Import [`notebooks/rtdetrv2_e1_p2_full_resume.ipynb`](notebooks/rtdetrv2_e1_p2_full_resume.ipynb).
2. Bật GPU và Internet, gắn dataset VisDrone có train/val với `images/` và `annotations/*.txt`.
3. Kiểm tra `DATA_ROOT` ở cell cấu hình. Chạy `RUN_MODE="proxy"` để thử 2 epoch trước.
4. Đặt `RUN_MODE="full"` để train 120 epoch. Mặc định `EXPERIMENTS=["E1_p2"]`,
   batch 2, input 640×640 và AMP. Chạy E0 với cùng mode/batch/seed để đánh giá đóng góp của P2.

Notebook nhúng sẵn các script cần thiết, khóa commit repo RT-DETR chính thức,
và tạo config E0/E1 trước khi train. E1 thêm feature stride 4; encoder attention vẫn trên P5.
Mapping giữ category 1–10, classifier 11 đầu ra và bỏ slot 0 tường minh khi export.

## Kết quả và resume

Output gồm `best.pth`, `last.pth`, config/manifest/log, predictions, AP/APsmall toàn bộ
và riêng phương tiện, Recall-heavy kèm số GT, ảnh lỗi, benchmark và ZIP artifact.
Để nối phiên Kaggle, lưu output, gắn output phiên trước vào Input và điền `RESUME`
trỏ tới `last.pth`; giữ config, manifest, log và best checkpoint đi kèm.
Không resume checkpoint ba level cũ vào P2 bốn level.

Đánh giá là COCO trên VisDrone đã chuyển đổi, không phải kết quả challenge VisDrone chính thức.
FPS là batch 1, FP32, PyTorch eager, chỉ forward model; chỉ đối chiếu trên cùng GPU/protocol.

## Công cụ

- [`prepare_visdrone_kaggle.py`](prepare_visdrone_kaggle.py): chuyển annotation VisDrone sang COCO.
- [`rtdetrv2_next/`](rtdetrv2_next/): tạo config P2, export EMA, đánh giá small/occlusion và kiểm tra.
- [`rtdetrv2_next/README.md`](rtdetrv2_next/README.md): kế hoạch và hướng dẫn bước B–C chi tiết.
- `rtdetrv2_next/build_kaggle_notebook.py`: tái tạo notebook từ các script trong repo.
- `rtdetrv2_next/validate_kaggle_notebook.py --repo <rtdetrv2_pytorch>`: kiểm tra notebook với ảnh giả trên CPU.

Đã kiểm tra smoke, train/validation một epoch cho E0/E1, export EMA, diagnostics,
render ảnh lỗi, benchmark CPU và resume E1 với ảnh giả. Chưa xác nhận train trên GPU Kaggle
hay kết quả AP/FPS trên VisDrone thật. Các số liệu trong `baseline_results.csv` và
`baseline_log.json` là baseline đã lưu, không phải kết quả mới của E1 P2.
