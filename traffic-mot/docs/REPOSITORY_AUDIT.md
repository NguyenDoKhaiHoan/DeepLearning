# Kiểm kê repository — Phase 0, 23/10/2026

Root: `C:/Users/Dell/Desktop/DeepLearning`. Đã lập danh sách thư mục/file source,
config, notebook, artifact và checkpoint, đọc các entry point liên quan đến
detector/tracking/evaluation và manifest. Các cache, node_modules, output media
và binary không được xem như source; không train lại, chạy notebook, trích PDF
hay tái tính detection AP trong đợt kiểm kê này.

## 1. Working tree trước thay đổi

Root Git HEAD: `d60031c` (`Add Kaggle RT-DETRv2 P2 training and evaluation workflow`).
Working tree đã có sửa `.gitignore`, `README.md`, `rtdetrv2_next/README.md`; xóa
`rtdetrv2_next/{build_kaggle_notebook,validate_kaggle_notebook}.py` và notebook
`rtdetrv2_p2_visdrone_kaggle.ipynb`; nhiều file chưa track gồm artifacts, docs,
archive, notebooks, web_demo và công cụ tracking/demo mới. Đây là trạng thái có
sẵn, không phải thay đổi do Phase 0 tạo. Không restore/reset các file đó.

Chỉ bổ sung `traffic-mot/` và liên kết từ README gốc trong đợt này. Không chỉnh
checkpoint, checkout RT-DETR, mã train, tracker/evaluator cũ hoặc frontend.
Không có AGENTS.md trong repository hay các thư mục cha đã kiểm tra.

## 2. Thành phần hiện có và quyết định sử dụng

| Khu vực | Nội dung đã xác định | Quyết định |
|---|---|---|
| `RT-DETR/` | official repo, PyTorch v1/v2 và Paddle; nested Git sạch khi audit | chỉ dùng `rtdetrv2_pytorch`, khóa commit |
| `prepare_visdrone_kaggle.py` | VisDrone DET → COCO; category 1–10 | tham chiếu mapping; không tạo MOT GT |
| `rtdetrv2_next/prepare_p2.py` | cấu hình/patch P2 | tham chiếu cấu trúc, không chạy lại train |
| `rtdetrv2_next/demo.py` | strict EMA loader, official Compose/postprocessor, ảnh/video | tham chiếu adapter Phase 1, tests real bundle |
| `rtdetrv2_next/export_predictions.py` | export mapping tường minh và EMA | tham chiếu postprocessing/export provenance |
| `rtdetrv2_next/evaluate.py` | COCO và diagnostic size/occlusion DET | giữ tách khỏi evaluator MOT mới |
| `rtdetrv2_next/track.py` | ByteTrackLite: greedy IoU, smoothing velocity, high/low confidence | legacy smoke/demo, không baseline ByteTrack chuẩn |
| `rtdetrv2_next/evaluate_tracking.py` | majority-map IDF1, approximate frag, greedy matching | không dùng cho HOTA/IDF1 chuẩn |
| `rtdetrv2_next/track_video.py` | offline tracker-lite CLI | giữ nguyên, không quảng bá thành pipeline mới |
| `notebooks/` | E1 P2 training 28 cells, tracking-lite notebook 8 cells | legacy workflow; chưa chạy lại |
| `archive/legacy/` | notebook YOLOv8/Faster-RCNN/RT-DETR cũ | tham chiếu lịch sử, không sửa |
| `web_demo/` | FastAPI, media/job control, React UI, tests | ngoài scope phase hiện tại |
| `docs/` | đề cương tracking-lite, demo setup, dataset analysis | tài liệu lịch sử; thiết kế mới ở traffic-mot/docs |
| `artifacts/` | bundle final, checkpoint resume/baseline, metadata | read-only, không copy/overwrite weights |
| `output/`, `outputs/`, `tmp/`, `TrafficVision.pdf` | kết quả/demo/report/render cũ | không diễn giải thành kết quả MOT mới |

README gốc vẫn có link tới notebook/build scripts không tồn tại theo working
tree hiện tại; không khôi phục khi chưa rõ chủ đích xóa. Notebook tracking có BOM
và một số chuỗi tiếng Việt lỗi encoding; không sửa trong Phase 0.

MOT-lite hiện dùng `zip(gt_frames,tr_frames)` nên cũng có thể bỏ qua phần dư nếu
độ dài không khớp; majority-map IDF1 không tính global one-to-one identity đúng
chuẩn. Tests legacy đạt chỉ xác nhận hành vi cũ, không xác nhận metric chuẩn.

## 3. Detector source, config và checkpoint

Source commit: `29320b6fd828f8e0987a71426cf2d961b09dfed7`.
Inference config: `artifacts/E1_p2_final/demo_config.yml`; train config/manifest:
`run_config.yml`, `run_manifest.json`. Config inference giữ kiến trúc và transforms
nhưng tắt backbone download, không phụ thuộc dataset Kaggle.

Checkpoint chọn: `artifacts/E1_p2_final/best.pth`, 357736373 bytes.
SHA256 tính lại trong audit:

`bd8bbed95bee31fae95d8bdead39f8ac072ea4fb48998339d7aef7988a1eca88`.

Hash khớp bundle manifest. Đọc bằng `torch.load(weights_only=True,map_location='cpu')`:
last_epoch119, có `ema.module` và `model`; classifier score-head shape `[11,256]`;
encoder input_proj.3 `[256,512,1,1]`, decoder input_proj.3 `[256,256,1,1]`.
Config: PResNet18, return indices0–3, stride4/8/16/32, hidden256, decoder3 layer,
queries300, input640×640. Regression tests hiện có xác nhận strict load EMA,
preprocessing và output scaling trên ảnh không vuông.

Mapping được đối chiếu giữa `label_map.json`, `classes.txt`, run manifest,
model config và classifier shape: slot0 bỏ; ID1 pedestrian,2 people,3 bicycle,
4 car,5 van,6 truck,7 tricycle,8 awning-tricycle,9 bus,10 motor.
Weights không tự chứa ngữ nghĩa tên lớp; bằng chứng semantic dựa bundle đi cùng
checkpoint đã hash, không suy tên từ shape. Phương tiện giữ ID3–10.

Các checkpoint khác tìm thấy:

| File | Kích thước byte | Vai trò |
|---|---:|---|
| `artifacts/checkpoints/E1_p2_resume_epoch89.pth` | 357736373 | resume cũ, không chọn mặc định |
| `artifacts/checkpoints/rtdetrv2_baseline_best.pth` | 322607981 | detector baseline khác, không trộn P2 |

`verification.json` cũ ghi kiểm tra ngày07/10/2026; đó là provenance lịch sử.
Đợt này chạy lại ba tests detector nhưng không tái tính full validation AP/FPS.
Training patch trong bundle liên quan solver/resume, không tự áp vào checkout sạch.

## 4. Môi trường và dependency gap

Python3.11.9; torch2.6.0+cpu; torchvision0.21.0; OpenCV4.11.0.86;
NumPy1.26.4; SciPy1.16.3; PyYAML6.0.3; pytest9.0.3; pandas2.3.2;
matplotlib3.10.8; ultralytics8.4.171. CUDA không sẵn có.
TrackEval/motmetrics/lap chưa cài; ffmpeg và ffprobe không có trên PATH.
Không sửa môi trường toàn cục để kiểm kê. Source Ultralytics được đọc qua đường
dẫn distribution, không import tracker vì backend có thể tự cài lap khi thiếu.

Direct dependency profile ở requirements.txt dùng phiên bản đã kiểm kê và thêm
lap0.5.12; SciPy1.15.3 cho Python3.10. TrackEval commit khóa qua GitHub API;
upstream package metadata chỉ yêu cầu NumPy/SciPy, nhưng legacy requirements.txt
pin bản rất cũ. Clean install runtime và evaluator import chưa được kiểm thử.

## 5. Dữ liệu và quyết định kiến trúc

Trong các thư mục dự án đã kiểm kê chưa có dataset MOT với GT persistent IDs,
split manifest và occlusion episodes dùng được để đánh giá. Một số demo path
trỏ tới VisDrone DET ngoài repository; không suy ra rằng có GT tracking ở đó.
Cần cung cấp dataset MOT đúng release ở Phase 5 và xác nhận scope class.

Chọn skeleton tách `traffic-mot/`, package tên `traffic_mot` để tránh collision
với `src` của upstream. Backend ByteTrack/BoT-SORT dùng thư viện đã pin thay cho
tracker-lite. OAR là nghiên cứu riêng có baseline/off và component ablation.
Evaluation dùng TrackEval adapter multi-class, không dùng MOTChallenge pedestrian
reader nguyên trạng. Các quyết định và gate chi tiết nằm trong bốn tài liệu chính.
