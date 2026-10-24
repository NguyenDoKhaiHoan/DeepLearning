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

## Chuẩn bị VisDrone MOT thật và manifest cho notebook02–04

Hai ZIP đã được kiểm tra local ngày24/10/2026:

- `VisDrone2019-MOT-val.zip`: 7 sequence, 2846 ảnh, có GT10 cột với Track ID.
- `VisDrone2019-MOT-test-dev.zip`: 17 sequence, 6635 ảnh, có annotation.

Upload cả hai vào dataset rồi **Create → Add Input**. Dùng `val` để tune/
validation, giữ `test-dev` cho final evaluation sau khi cấu hình đã freeze.
Không cần tạo MP4 từ các JPG, không gán FPS giả. Raw GT target ID0 là hợp lệ;
public tracker output ID dương là contract riêng, không được làm mất GT0.

Trong Kaggle, lấy code mới rồi tạo manifest theo mount path thực tế:

```python
!git -C /kaggle/working/DeepLearning pull --ff-only
!python /kaggle/working/DeepLearning/traffic-mot/scripts/prepare_mot_dataset.py --root /kaggle/input/visdrone-mot --output /kaggle/working/mot_split_manifest.json
```

Nếu path mount khác, thay `--root` bằng đường dẫn Copy path trong Add Input.
Script tìm đúng hai release ZIP hoặc thư mục đã giải nén; không tự giải nén/
chuyển video, đọc index ảnh/annotation và hash nguồn. Nếu cùng lúc có cả ZIP
và bản giải nén, chọn tường minh `--validation-source <val_zip_or_directory>
--test-source <test_dev_zip_or_directory>` thay `--root`.

Output là `mot_split_manifest.json` và `mot_split_manifest.preflight.json` cạnh
nhau. Lệnh mặc định chỉ đọc GT/ảnh đầu mỗi sequence của **validation**, không
parse annotation test để tune. Không ghi đè manifest/report đã có; dùng path
mới nếu đổi dataset. Preflight test chỉ dùng ở final gate đã freeze.

Notebook02–04 mới có cell `prepare_mot_manifest` dùng chung source, tạo manifest
tại `OUTPUT_ROOT / "mot_split_manifest.json"`. Chỉnh `MOT_DATA_ROOT` đúng mount;
hoặc đặt `MOT_MANIFEST` trỏ vào file đã tạo bằng lệnh trên. File local trỏ `D:/`
không dùng nguyên trên Kaggle: cần tạo lại theo mount `/kaggle/input/`.

Manifest hỗ trợ `source_kind=images_zip` với archive/member paths hoặc
`images_dir` với frames_dir/gt. Giữ thứ tự ảnh1-based và raw IDs; checker không
chấp nhận trùng sequence/source UAV giữa validation và test. Fingerprint index
ZIP dựa filename/size/CRC; hash archive giữ integrity. Đối với images_dir, checker
hash ảnh thật; chưa làm perceptual duplicate detection hoặc kiểm tra tất cả ảnh
decode được. Chỉ khi có evaluator chuẩn mới báo HOTA/IDF1/MOTA/IDSW.

**Có manifest không đồng nghĩa notebook02–04 chạy tracking được:** các runner
ByteTrack/BoT-SORT/evaluation/OAR/final còn thiếu và notebook vẫn dừng ở guard.

Tạo lại notebook: `python traffic-mot/scripts/build_kaggle_notebooks.py`.
