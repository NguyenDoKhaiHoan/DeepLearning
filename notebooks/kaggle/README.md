# Kaggle — hoàn thiện đề tài Computer Vision

Các notebook dùng chung package `traffic_mot`, không chứa bản sao thuật toán.
Tất cả hiện chưa có execution/output Kaggle trong session phát triển này.

| Notebook | Trạng thái |
|---|---|
| 01_detection_inference | detection đã chạy CPU local, chờ GPU Kaggle |
| 02_mot_baselines | preflight/template; tracker/evaluation CLI chưa có |
| 03_occlusion_ablation | preflight/template; cần baseline/OAR/runner |
| 04_final_benchmark | preflight/template; cần validation freeze/final CLI |

Template02–04 dừng với lỗi khi thiếu entrypoint; không dùng ByteTrackLite/MOT-lite
để lấp baseline chuẩn và không tạo metric giả.

## Chạy notebook01

Trong một cell Kaggle, bật Internet rồi lấy source từ GitHub:

```python
!git clone https://github.com/NguyenDoKhaiHoan/DeepLearning.git /kaggle/working/DeepLearning
!git clone https://github.com/lyuwenyu/RT-DETR.git /kaggle/working/DeepLearning/RT-DETR
!git -C /kaggle/working/DeepLearning/RT-DETR checkout 29320b6fd828f8e0987a71426cf2d961b09dfed7
%pip install -r /kaggle/working/DeepLearning/traffic-mot/requirements.txt -r /kaggle/working/DeepLearning/traffic-mot/requirements-dev.txt
```

Nếu source đã clone thì dùng `git pull --ff-only` ở checkout DeepLearning thay
vì clone lại. Sau khi cài dependency, restart kernel trước khi chạy các cell
inference nếu torch/NumPy đã được import. Import notebook01 từ GitHub hoặc upload
file notebook, bật GPU và sửa BUNDLE_DIR/VIDEO sang đường dẫn Kaggle Input thật.
Checkpoint và video không nằm trong Git; gắn chúng riêng bằng Add Input.

1. Clone/mount repo có `traffic-mot/src/traffic_mot`; cài runtime và dev requirements
   với torch/torchvision CUDA tương thích; chạy `pip check`.
2. Bật GPU accelerator. Mount bundle chứa best.pth, demo_config.yml, label_map.json,
   classes.txt, bundle_manifest.json. Không thay best bằng resume giữ manifest cũ.
3. Nếu thiếu RT-DETR source, clone official repo vào `RT-DETR/`, checkout commit
   `29320b6fd828f8e0987a71426cf2d961b09dfed7` và giữ checkout sạch.
4. Sửa REPO_ROOT/BUNDLE_DIR/UPSTREAM_REPO/VIDEO ở cell setup; dùng video thật.
   Preflight xác minh GPU, source, checkpoint/config/mapping và video trước load.
5. Smoke48 frame; kiểm tra JSONL/MP4 đọc lại và manifest. Khi đạt, bỏ max_frames
   để xử lý đầy đủ. Lưu artifact Kaggle để nghiệm thu GPU. Không lấy smoke startup
   time làm steady-state FPS, không tính HOTA trên video không GT.

## MOT split manifest cho notebook02–04

Đây là ví dụ **định dạng**, chưa phải dữ liệu có sẵn:

```json
{
  "dataset": "VisDrone-MOT",
  "validation": [{"name": "seq_val_01", "video": "videos/seq_val_01.mp4", "gt": "annotations/seq_val_01.txt"}],
  "test": [{"name": "seq_test_01", "video": "videos/seq_test_01.mp4", "gt": "annotations/seq_test_01.txt"}]
}
```

Path tương đối tính từ manifest. GT10 cột phải có persistent IDs, frame1-based
và occlusion/ignore đúng release. Validation/test khác source/camera; không chỉ
đổi tên/cắt hai đoạn cùng clip. Checker chặn trùng name/path/hash, còn camera/
source provenance cần dataset audit. Preflight validation không parse test GT.
Final test chỉ chạy sau selected_config đã freeze.

Tạo lại notebook: `python traffic-mot/scripts/build_kaggle_notebooks.py`.
