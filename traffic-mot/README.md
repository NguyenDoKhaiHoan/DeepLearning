# Multi-Object Detection and Tracking under Occlusion in Traffic Videos

Ưu tiên hoàn thiện đề tài Computer Vision trước Đồ án3. **Detection và baseline
ByteTrack/BoT-SORT đã chạy smoke thật CPU; full Kaggle validation còn pending.**
Không triển khai web/frontend, Kafka, Flink hoặc database trong lộ trình CV.
Các module cũ của repository cha được giữ nguyên.

## Trạng thái và lộ trình

[STATUS](docs/STATUS.md) chỉ ghi kết quả có bằng chứng; [IMPLEMENTATION_PLAN](docs/IMPLEMENTATION_PLAN.md)
quy định thứ tự: detection + Kaggle → ByteTrack/BoT-SORT → MOT baseline evaluation
Kaggle → Occlusion-Aware Recovery → validation ablation → trajectory/events →
final heldout evaluation/benchmark/demo. Baseline dùng backend chuẩn đã pin và
official TrackEval metrics; ByteTrackLite/MOT-lite cũ được giữ nguyên riêng biệt.

## Cài đặt

Baseline local: Python3.11.9, torch2.6.0 CPU/torchvision0.21.0. Python tối thiểu3.10;
clean install trên tổ hợp khác cần chạy lại tests. Từ `DeepLearning`:

```powershell
python -m venv traffic-mot/.venv
.\traffic-mot\.venv\Scripts\Activate.ps1
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r traffic-mot/requirements.txt -r traffic-mot/requirements-dev.txt
python -m pip install -e ./traffic-mot --no-deps
```

CUDA cần torch/torchvision wheels phù hợp thiết bị; giữ cặp phiên bản trong
[hướng dẫn PyTorch2.6](https://pytorch.org/get-started/previous-versions/#v260).
Requirements khóa dependency trực tiếp, manifest lưu environment thực tế.
`lap`/TrackEval thuộc bước tracking/evaluation, không cần để chạy detection đã có.
FFmpeg CLI chưa có trên PATH local; OpenCV đã đọc/ghi MP4 bằng backend của nó.
Không cài đồng thời opencv-python và opencv-python-headless trong một environment.

## Kiểm tra và chạy detection thật

```powershell
python traffic-mot/scripts/validate_phase0.py --check-assets
python -m pytest -c traffic-mot/pyproject.toml traffic-mot/tests --run-model -q
python traffic-mot/scripts/run_video.py --source path/to/traffic.mp4 --detector rtdetrv2_p2 --config traffic-mot/configs/detector.yaml --device cpu --save-video --save-json --max-frames 48
python traffic-mot/scripts/verify_detection_run.py --run-dir <output_run_directory>
```

Thay source bằng MP4 thật, webcam index hoặc RTSP. Output mặc định tạo run UUID
mới trong `traffic-mot/outputs/`; có thể chọn `--output-dir <new_directory>`.
Mỗi run có manifest, tùy chọn detections.jsonl và annotated.mp4; không overwrite
source/run cũ. `--help` liệt kê các argument đã thực thi. Bỏ max-frames để chạy
đến EOF. Webcam/RTSP là live source, dùng Ctrl+C hoặc frame limit để kết thúc.

Config YAML trỏ repo/config/checkpoint/labels tương đối `traffic-mot/`, hỗ trợ
absolute path khi mount Kaggle. Preflight kiểm tra source commit+sạch, architecture,
transforms, mapping và SHA256 trước `torch.load(weights_only=True)` + strict load.
Không tải model mới hoặc fallback sang mock khi thiếu checkpoint.

Detector giữ ID3–10, original-pixel xyxy float32 và score≥0.10. Video mặc định chỉ
vẽ score≥0.30 cho dễ đọc; `--display-threshold` chỉ đổi overlay, không đổi JSONL
hoặc dữ liệu đầu vào tracker tương lai. Không thêm NMS vào output chuẩn hiện tại.
Ngưỡng thấp có thể tạo nhiều false positives; smoke không đo AP hay MOT quality.

Preprocessing đúng training: BGR→RGB/PIL, official Compose resize640×640 stretch,
float32/255, không ImageNet normalization; postprocessor nhận [W,H] để scale về
frame gốc. AMP chỉ cho CUDA, chưa được kiểm chứng số học trên máy local CPU.
Native batch inference có ở adapter, streaming CLI mặc định xử lý từng frame.

File clock dùng PTS, hoặc `--clock frame_fps` khi đã biết input CFR. MP4 writer là
CFR, reject PTS bất thường/VFR để tránh retime âm thầm; odd dimensions bị từ chối
thay vì crop âm thầm. Output không giữ audio. RTSP reconnect tạo session và MP4
riêng; live clock là monotonic session, không tự nhận UTC đồng bộ camera.

## Mock và tests

`--detector mock --mock-fixture <json>` là lựa chọn tường minh. Fixture là list
frame lists, mỗi detection `{bbox:[x1,y1,x2,y2], confidence:0.9, class_id:4}`.
Manifest/JSONL/video có mock flag/watermark. Không dùng fixture để báo cáo metrics.

`pytest` mặc định skip tests đánh dấu real_model; `--run-model` bắt buộc nạp bundle
thật và lỗi nếu thiếu. Unit tests cần ML/runtime dependencies; Phase0 checks tối
thiểu có thể chạy riêng `tests/test_phase0.py` với requirements-dev.txt.
Kết quả và lệnh smoke thực tế: [PHASE_1_REPORT](docs/PHASE_1_REPORT.md).

## Kaggle

Xem [notebooks/kaggle](../notebooks/kaggle/README.md). Notebook01 dùng cùng adapter
và pipeline; notebook02 đã có CLI E0/E1 và TrackEval, mặc định smoke trước khi
chạy full validation. Notebook03 đã có OAR/component ablation runner dùng cùng
source/cache; notebook04 vẫn thiếu final runner. Không có notebook nào được tuyên bố đã chạy GPU
Kaggle trong session hiện tại. Thiếu MOT dataset/GT thì HOTA/IDF1/MOTA/IDSW là N/A.
Chỉ tune validation; test giữ cho đánh giá cấu hình đã freeze.

Ngày24/10 đã kiểm tra MOT val/test-dev ZIP thật và bổ sung
`scripts/prepare_mot_dataset.py` tạo manifest trực tiếp từ native JPG sequences
(ZIP hoặc thư mục), không chuyển thành MP4 giả FPS. Xem
[MOT data preflight report](docs/MOT_DATA_PREPARATION_REPORT.md).

## MOT baselines đã triển khai

```powershell
python -m pip install -r traffic-mot/requirements-eval.txt
python traffic-mot/scripts/run_mot_baselines.py --detector-config traffic-mot/configs/detector.yaml --manifest <native_manifest.json> --split validation --sequence uav0000305_00000_v --max-frames 48 --output-root <new_run_directory>
```

Bỏ `--sequence`/`--max-frames` để chạy toàn bộ validation. Không nhận split=test
ở runner tuning này. Native JPG/ZIP giữ resolution/frame order; timestamp nguồn
chưa biết để null/frame_index. Optional `--save-video --visualization-fps 10`
chỉ chọn tốc độ playback; video pad tối đa1 pixel phải/dưới khi kích thước lẻ,
không thay ảnh inference/evaluation. Không coi encoded FPS là processing FPS.

Detector chạy một lần tạo cache được kiểm hash/config/data/pixel/frame range;
E0/E1 cùng replay cache đó. `--detection-cache <cache_directory>` cho run mới
có đúng signature để tránh inference lại. Không dùng cache thiếu/mock/thay đổi.
BoT-SORT sparseOptFlow GMC, ReID=false; OAR=false. Output có predictions JSONL,
MOT TXT, class sidecars, metrics JSON/raw, summary JSON/CSV và manifest.
HOTA/IDF1/MOTA là ratio0..1 (MOTA có thể âm); IDSW/MT/ML là count.

Protocol `visdrone_vehicle_research_v1` đánh giá class4/5/6/9 theo evaluation.yaml,
bảo vệ valid GT matches rồi suppress unmatched prediction trong explicit ignore.
Không nhận là VisDrone challenge-equivalent. TrackEval dtype bridge chỉ sửa
np.float/np.int trong module toolkit, không thay matching/formula hoặc NumPy global.
Known raw Frag limitation của bản pin được ghi trong metrics; summary không báo Frag.
Xem [MOT baseline report](docs/MOT_BASELINE_REPORT.md) để xem scope/test/metrics thật.

## Occlusion-aware recovery và ablation

```powershell
python traffic-mot/scripts/run_ablation.py --config traffic-mot/configs/experiments.yaml --detector-config <same_detector_profile.yaml> --manifest <same_native_manifest.json> --split validation --baseline-run <completed_baseline_run> --output-root <new_ablation_directory>
```

`--baseline-run` dùng lại exact cache/frame range từ notebook02, xác minh lại
E0/E1 trước so sánh. Nếu không có reference, runner tự tạo một cache và tính
lại E0/E1; dùng `--sequence <name> --max-frames 48` cho smoke, bỏ giới hạn để
chạy toàn bộ validation. `--detection-cache` cũng được hỗ trợ với signature khớp.
ReID off, không nhận test split. Không sao chép kết quả full thành công từ log
data preflight; toàn bộ metrics đều phải qua actual inference/cache và TrackEval.

E2/E3 thêm policy `motion_overlap_v1`: overlap lịch sử, exit hypothesis, retention
có giới hạn, class/Kalman/IoU gates và unmatched one-to-one reassociation có
ambiguity rejection. Không cướp stock matches, không predict hai lần/frame,
không dùng GT trong inference. Giới hạn ms chỉ áp dụng khi có timestamp nguồn;
native JPG dùng frame bounds và báo rõ chưa enforce ms. Không có appearance/ReID.

Ngoài E0–E3 còn bốn component runs cho hai backend. Output có diagnostics theo
occlusion/size, fixed GT episode denominator, decision traces/counters và paired
sequence bootstrap. HOTA-heavy không được suy ra từ recall bins. Recovery rate
là same-ID continuity sau episode GT có nhãn, không chứng minh causal OAR recovery;
không có eligible episode thì N/A. Labeled occlusion khác unannotated full gaps.

Smoke chỉ ghi `candidate_config.yaml` với frozen=false. Chỉ run đầy đủ validation
mới ghi `selected_config.yaml`, chọn HOTA→IDF1→ít IDSW, baseline thắng khi hòa;
không tự ưu tiên OAR nếu metrics xấu. [OAR report](docs/OAR_ABLATION_REPORT.md)
ghi kết quả thật48-frame hiện có: OAR không cải thiện HOTA/IDF1 trên đoạn này.
Full Kaggle validation, OAR tuning và final-test evidence còn pending.

## Tài liệu thiết kế

- [ARCHITECTURE](docs/ARCHITECTURE.md): detector/tracker bridge và OAR.
- [MOT_EVALUATION_PLAN](docs/MOT_EVALUATION_PLAN.md): TrackEval, ignores, occlusion/recovery và ablation.
- [INTEGRATION_SPEC](docs/INTEGRATION_SPEC.md): contract/schema cho giai đoạn sau; không triển khai Đồ án3.
- [REPOSITORY_AUDIT](docs/REPOSITORY_AUDIT.md), [PHASE_0_REPORT](docs/PHASE_0_REPORT.md): lịch sử Phase0.

Package `src/traffic_mot/` tránh collision với `src` của RT-DETR. Data/model/output
ngoài Git; giữ nguyên checksum nguồn. Không suy tốc độ km/h từ pixel khi chưa
calibration và không báo cải thiện che khuất từ một video demo detection.
