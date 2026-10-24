# Lộ trình ưu tiên hoàn thiện đề tài Computer Vision

Cập nhật 24/10/2026. Mục tiêu: **Phát hiện và theo dõi đa đối tượng dưới điều kiện
che khuất trong video giao thông**. Hoàn thiện và nghiệm thu đề tài này trước
Đồ án3. Không phát triển frontend, web backend, Kafka, Flink hoặc database ở đây.
Giữ nguyên web demo, train workflow, checkpoint và module cũ đang hoạt động.

## Trạng thái thực tế và bằng chứng

| Hạng mục | Bằng chứng | Trạng thái |
|---|---|---|
| Phase0: audit/design/config/schema | PHASE_0_REPORT, asset validator, 22 tests lịch sử | đã kiểm chứng local |
| RT-DETRv2-P2 | SHA256 bundle, EMA epoch119 strict load, source pin | có thật/load được |
| Detection/VideoReader/MP4/JSONL | 48 frame giao thông thật CPU, video đọc lại được | local có bằng chứng; xem PHASE_1_REPORT |
| Kaggle detection | notebook01 dùng source chung + preflight GPU/assets/video | chuẩn bị; chưa chạy Kaggle |
| ByteTrackLite/MOT-lite cũ | tests legacy, greedy association/approximate metrics | giữ nguyên; không phải baseline chuẩn |
| ByteTrack/BoT-SORT chuẩn | pinned stock adapters, real MOT48-frame CPU smoke + cache replay | local đạt; full Kaggle validation pending |
| MOT data preparation | native ZIP/image manifest + validation preflight trên dữ liệu thật | local đạt, chưa metric |
| MOT baseline evaluation | official TrackEval + multi-class ignore protocol | real CPU prefix verified, chưa toàn split |
| OAR/ablation | versioned policy + runner/common cache + real48-frame CPU study8 variants | local smoke đã chạy; improvement/full Kaggle pending |
| Trajectory/events | schema/thiết kế | chưa engine/run |
| Final test/benchmark | CLI+freeze verifier+runtime measurement | tests + real validation component smoke; test run pending |

Folder/notebook được tạo không chứng minh thuật toán đã chạy. Trạng thái hiện tại
ở [STATUS](STATUS.md); lịch sử Phase0 giữ nguyên. Các số Phase1–6 trong thiết kế
ban đầu là nhóm module cũ; **thứ tự có hiệu lực là bảy bước dưới đây**.

## 1. Detection integration và Kaggle inference — bước đang hoàn thiện

Dùng BaseDetector, MockDetector fixture rõ ràng, RTDETRv2P2Detector, VideoReader,
renderer và CLI mới; tái sử dụng model/config/transforms chính thức. EMA strict,
BGR→RGB, resize640/float32/255, postprocessor [W,H], original IDs3–10, cutoff0.10.
Không thêm NMS hoặc đổi preprocessing giữa run. Display cutoff độc lập JSONL.

**Lệnh local đã triển khai, từ repository root:**

```powershell
python traffic-mot/scripts/validate_phase0.py --check-assets
python -m pytest -c traffic-mot/pyproject.toml traffic-mot/tests --run-model -q
python traffic-mot/scripts/run_video.py --source outputs/web_demo/0315259f-8088-44c9-8c3c-129efe920d21/input.mp4 --device cpu --max-frames 48 --save-video --save-json --output-dir traffic-mot/outputs/your_new_run
```

Kaggle: `../../notebooks/kaggle/01_detection_inference.ipynb`, mount bundle/video
thật, checkout RT-DETR đúng commit và chọn GPU. Notebook gọi source repository.

**Đầu ra:** resolved detector profile/hash, manifest, detections.jsonl,
annotated.mp4, evidence đọc lại. Mock có flag/watermark, không dùng báo cáo.
**Gate:** negative/unit/real model tests đạt; class/confidence/box/time đúng;
thiếu asset phải lỗi, không fallback mock; có artifact notebook01 chạy thật GPU.
**Kết quả hiện có:** local smoke; Kaggle GPU pending, không coi CPU là gate GPU.

## 2. ByteTrack / BoT-SORT baseline

Dùng tracker implementation `ultralytics==8.4.171`, bridge DetectionBatch vào
backend; không thay detector bằng YOLO, không đổi tên ByteTrackLite thành chuẩn.
Kalman/high-low association/lifecycle dùng backend đã pin; ReID off, GMC tường
minh. Cùng detection cache có hash cho E0/E1; OAR off.

**Files hiện có:** tracking/{base,bytetrack,botsort,backend_bridge,baselines}.py,
video/mot_sequence.py và CLI run_mot_baselines.py cho notebook02. run_video.py
vẫn detection-only; không quảng bá --tracker khi CLI đó chưa nhận argument.
**CLI thực:** `python scripts/run_mot_baselines.py --detector-config <resolved>
--manifest <native_manifest.json> --split validation --output-root <new_dir>`.
**Đầu ra:** tracks.jsonl, MOT predictions+class sidecar, ID video, provenance.
**Gate:** empty/high/low/lost/remove tests, no birth từ low, đúng buffer; ID không
reset/lẫn session; off bằng backend gốc; collision/class-switch/GMC tests;
video thật đọc lại. **Hiện tại:** stock backend parity/lifecycle/ID tests đạt,
real48-frame CPU smoke + identical replay observations; full GPU validation pending.

## 3. Kaggle MOT baseline evaluation — bắt buộc trước cải tiến

Cần VisDrone MOT đúng release với persistent IDs, ignore/occlusion; split theo
sequence/camera. Không dùng VisDrone-DET làm tracking GT. Xác nhận scope class
(khởi đầu vehicle4/5/6/9) trước khi đóng băng. Tạo VisDroneMOTAdapter cho TrackEval
multi-class/ignore, không đưa car vào reader MOTChallenge pedestrian mặc định.

**CLI thực:** `python scripts/run_mot_baselines.py --detector-config <resolved>
--manifest <mot_split_manifest.json> --split validation --output-root <new_dir>`.
Notebook02 gọi CLI chung; CLI dùng evaluation/trackeval_runner.py. Standalone
evaluate_mot.py chưa có, không cần để chạy baseline/eval end-to-end hiện tại.
**Đầu ra:** E0/E1 predictions/cache, raw TrackEval JSON, HOTA/IDF1/MOTA/IDSW/MT/ML,
GT counts, ignore diagnostics, split/hash/environment và failure clips.
**Gate:** perfect/missing/FP/ID-switch/fragment/class/ignore/empty fixtures,
đối chiếu trực tiếp toolkit, timeline không bị zip cắt; validation Kaggle chạy
thật với E0/E1 cùng cache/precision/frame order. Thiếu GT → N/A, không số giả.
**Cập nhật24/10:** native images preflight và baseline/eval smoke48 frame thật
đạt; class4/5/6/9, protocol nghiên cứu versioned, official HOTA/Identity/CLEAR,
known raw Frag limitation được công bố. Notebook02 có smoke/full switch.
Không lấy prefix result để nghiệm thu full validation hoặc công bố GPU FPS.

## 4. Occlusion-Aware Tracking improvement

Sau baseline đo được, triển khai overlap/history evidence, phân biệt
suspected_exit/unknown/occluded, adaptive retention bounded và Kalman-gated
unmatched reassociation. GT chỉ dùng offline. ReID chỉ khi có extractor thật
được kiểm chứng domain vehicle; không nhận diện tính năng qua tên config.

**Files hiện có:** occlusion/evidence.py, recovery.py, ablation.py; backend extension
version motion_overlap_v1 dùng hooks sau stock association, trước removal/birth.
State/history/retention tập trung trong recovery.py, không nhân bản module trống.
**CLI thực:** run_ablation.py đọc policy từ configs/occlusion.yaml, cùng shared
tracking runner. run_video.py vẫn detection-only; không quảng bá CLI chưa hỗ trợ.
**Đầu ra:** E2/E3 predictions, recovery evidence/decision, public ID map/overhead.
**Gate:** một predict/update mỗi frame, không cướp active match, one-to-one,
ambiguity từ chối, exit/timeout/covariance/crowd/camera-motion/dropout tests;
mock chỉ chứng minh invariant. **Hiện tại:** tests kiểm invariant và hồi phục ID
trên controlled probes đạt; real48-frame smoke8 variants đã chạy. OAR làm metrics
giảm nhẹ trong đoạn này; không có claim improvement. Frame bounds native JPG,
ms cap chỉ khi source time đã biết; không ReID hoặc full-gap guarantees.

## 5. Kaggle ablation experiments

Notebook03 dùng run_ablation.py chạy E0–E3, retention-only, reassociation-only,
kết hợp. Tune **validation** với cùng budget/cache/GT episodes/hardware;
test giữ nguyên, chưa dùng để chọn threshold. E3 chỉ làm khi BoT-SORT hook đạt gate.

**CLI thực:** `python scripts/run_ablation.py --config configs/experiments.yaml
--detector-config <resolved> --manifest <manifest> --split validation --output-root <dir>`.
**Đầu ra:** per-sequence/class metrics, occlusion/size sau full-GT matching,
recovery success với mẫu số GT cố định, sequence-paired CI, overhead và
selected_config.yaml có hash/protocol selection được freeze **chỉ full validation**;
smoke chỉ ghi candidate_config frozen=false. --baseline-run dùng exact cache/range
và verify E0/E1, hoặc runner recompute tất cả từ một cache mới.
**Gate:** thực nghiệm thật tái lập được, không tune test, không rematch chỉ heavy
GT; giữ cả kết quả xấu. **Hiện tại:** runner implemented/tested local smoke,
full Kaggle/GT episode samples cần kiểm tra. Prefix smoke có0 eligible recovery
episodes → N/A, không biến thành success0 hay success100%. Bootstrap1 sequence
được đánh dấu degenerate, không dùng làm chứng cứ thống kê toàn dataset.

## 6. Trajectory và Traffic Event Detection

Sau khi identity được đánh giá, bổ sung bounded history, smoothing/displacement/
direction pixel; line crossing, region entry/exit, count, wrong-way, stationary.
Tái sử dụng track stream, YAML/schema. Không chuyển thành phát triển Đồ án3.

**CLI dự kiến:** run_video thêm event/trajectory configs; export_events.py hỗ trợ
JSONL→JSON/CSV, stable event_id và source/time range.
**Đầu ra:** trajectories.jsonl, events.jsonl/CSV, ROI/line/event video overlays.
**Gate:** hysteresis/jitter/finite line/ROI dwell, recovered ID không đếm trùng,
namespace camera/session, gap không tạo event giả, interval/confidence semantic
validation. Không pixel→km/h thiếu calibration. **Hiện tại:** chưa có engine/run.

## 7. Final evaluation, benchmarking và video demo

Notebook04 chạy final_evaluation.py trên heldout **test**, chỉ sau validation
freeze; không search tham số. HOTA/IDF1/MOTA/IDSW và occlusion/recovery/size cần GT
thật. FPS cần đo thật, tách detector/tracking/recovery/pipeline, warmup/CUDA sync,
batch/precision/encode/measured frames. Cache FPS không phải pipeline FPS.

**CLI thực:** `python scripts/final_evaluation.py --config <frozen_config>
--detector-config <resolved> --manifest <manifest> --split test --output-root <dir>`.
**Đầu ra:** final raw/results tables, timings, video demo có nguồn thật, failure
analysis và reproducibility bundle. **Gate:** regression+clean-install+Kaggle
rerun, config/data hash, disjoint splits và đủ artifact; lúc đó mới nghiệm thu CV.
**Hiện tại:** implemented/tested freeze gate (source validation artifact/hash,
full frame coverage, model/tracker/OAR/eval profiles, winner rule). Notebook04
tìm đúng ABLATION_RUN hoặc unique freeze; không đổi candidate smoke thành final.
Shared tracking engine tái sử dụng cho selected experiment/test, public baseline
và ablation vẫn validation-only. Core benchmark real decode/detector/tracking
được tích hợp, không gồm render/encode/export. Tests và real validation smoke
đạt; full validation/final heldout/GPU evidence vẫn pending, không nhận đã nghiệm thu.

## Cần bổ sung và giới hạn đợt triển khai này

Local đã có checkpoint/config/source. Kaggle cần mount đúng bundle và video,
checkout source pin, bật GPU rồi lưu artifact notebook01. Máy local hiện không
có CUDA. Đã có MOT val/test-dev ngoài Git, manifest local và preflight validation;
Kaggle cần Add Input rồi chạy prepare_mot_dataset.py để tạo path theo mount.
Source checker chặn trùng sequence/source UAV/fingerprint; nguồn camera và near
duplicates còn phải được dataset audit xác nhận. Eval/GT ignore policy chưa nghiệm thu.

Hai đợt sửa Stage not implemented đã bổ sung baseline/evaluation và OAR/ablation,
kiểm chứng local bằng model/cache/GT thật. Kaggle full-split evidence vẫn cần chạy.
Final CLI đã nối sau ablation theo yêu cầu, nhưng chưa có local full-validation
freeze nên không chạy thật/test-report heldout. Không triển khai events hoặc
tự công bố các gate Kaggle/final đã đạt từ unit/component smoke.
