# Bằng chứng detection integration local và chuẩn bị Kaggle

Ngày23/10/2026. Hoàn thiện phần local của bước1 trong lộ trình mới. **Chưa nghiệm
thu Kaggle GPU, standard MOT, OAR hoặc toàn bộ đề tài.** Bằng chứng dưới đây là
chạy thật trong workspace hiện tại, không phải dự kiến hoặc output mock.

## Repository và phần tái sử dụng

Giữ nguyên source/checkpoint/training/legacy tracking/web demo. Dùng official
RT-DETR model/postprocessor và validation Compose đã được kiểm chứng trước đó.
Tạo adapter trong package traffic_mot để tách nguồn video, model và rendering;
không viết lại network, ByteTrack hoặc evaluator trong bước này.
Source `RT-DETR` vẫn sạch tại commit29320b6fd828f8e0987a71426cf2d961b09dfed7.
Bundle best.pth EMA119 có SHA256:
`bd8bbed95bee31fae95d8bdead39f8ac072ea4fb48998339d7aef7988a1eca88`.

## Files bổ sung/cập nhật

- `src/traffic_mot/detection/{base,mock,bridge,rtdetrv2_p2}.py`: interface, fixture
  replay, preflight và model adapter CPU/CUDA/native batch.
- `video/{clock,reader,writer,pipeline}.py`: MP4/webcam/RTSP, explicit clocks,
  bounded reconnect/new session, JSONL/MP4 export và cleanup khi lỗi.
- `visualization/annotator.py`, `utils/{types,seed,kaggle,verify}.py`: original
  boxes, seed, notebook orchestration/preflight và đọc lại output để nghiệm thu.
- `scripts/{run_video,verify_detection_run,build_kaggle_notebooks}.py`, cập nhật
  validator Phase0 để chỉ rõ phạm vi kiểm tra, không phủ nhận runtime mới.
- Tests detector/video/real model/Kaggle, pytest real_model opt-in marker;
  version package0.2.0; README/scripts docs và IMPLEMENTATION_PLAN/STATUS.
- Bốn notebook ở `../../notebooks/kaggle/`, README đi kèm; không có output execution
  Kaggle. Notebook02–04 là templates guard vì runner/algorithms chưa có.

## Checkpoint/config/source trước load

Preflight xác minh file tồn tại, SHA256 với bundle manifest, mapping artifact với
profile (slot0 bỏ, phương tiện original IDs3–10), source commit và working tree
sạch, PResNet18/P2–P5/hidden256/decoder3/queries300/640×640, focal/top300, validation
transforms chính xác. Config self-contained, không tự resolve training includes
có thể đổi model; dùng config inference đã resolved.

Chỉ sau preflight mới `torch.load(weights_only=True,map_location='cpu')`, chọn
EMA hoặc model tường minh, strict state load. Missing EMA không fallback model;
missing checkpoint không sinh model mới hoặc chuyển mock. Foreign package src
đã import từ repo khác bị từ chối. GPU/CUDA/AMP không hợp lệ lỗi trước weight load.

## Kiểm tra preprocessing/class/coordinate/confidence

Real adapter nhận BGR uint8, chuyển RGB/PIL, dùng official Compose resize stretch
640×640 rồi float32/255. Probe màu BGR[10,80,240] cho tensor RGB[240,80,10]/255.
Không ImageNet normalization, letterbox hoặc NMS tự thêm.

Official postprocessor nhận [W,H]. Controlled-logit test sau strict load dùng
normalized box[0.5,0.5,0.2,0.4] và frame200×100, mong đợi xyxy[80,30,120,70],
sai lệch float32 cho phép1e-4 pixel. Test batch hai ảnh không vuông có kích thước
đảo nhau xác minh scale độc lập. Đây là probe unit có output biết trước, không
được tính thành kết quả model trên dataset thật. Initial test dùng equality tuyệt
đối đã được sửa sang numerical tolerance, không làm tròn output để vượt test.

Confidence[0,1] và finite được kiểm tra; original IDs được giữ; slot0/nonvehicle
bị lọc; label chưa map phải lỗi kể cả score thấp. Clip về original frame, loại
NaN/Inf/box đảo hoặc rỗng, không sửa box đảo thành box hợp lệ. Cutoff0.10 giữ
dải thấp cho tracker tương lai. Drawing cutoff0.30 chỉ ảnh hưởng MP4, JSONL giữ
toàn bộ detection qua cutoff0.10, không round tọa độ.

## Lệnh đã chạy và smoke thật

Từ repository root:

```powershell
python -m pytest -c traffic-mot/pyproject.toml traffic-mot/tests --run-model -q
python traffic-mot/scripts/validate_phase0.py --check-assets
python traffic-mot/scripts/run_video.py --source outputs/web_demo/0315259f-8088-44c9-8c3c-129efe920d21/input.mp4 --detector rtdetrv2_p2 --device cpu --max-frames 48 --save-video --save-json --output-dir traffic-mot/outputs/phase1_real_cpu_final
python traffic-mot/scripts/verify_detection_run.py --run-dir traffic-mot/outputs/phase1_real_cpu_final
```

CLI không overwrite run directory; để chạy lại, chọn output directory mới.

| Kiểm tra / measurement thực | Kết quả |
|---|---|
| Full tests với --run-model | 52 passed, không skip, 45.47s; gồm4 real_model tests |
| Targeted video/Phase0 regression sau cập nhật verifier | 22 passed, 13.26s |
| Config/asset validator | 8 configs passed; checkpoint hash/labels/commit khớp |
| Detector trong smoke | mock=false, EMA119, strict load, CPU FP32, threads4 |
| Source MP4 | clip giao thông có897 frame, 640×360, nominal24fps |
| Phạm vi inference | prefix48 frame, termination=frame_limit |
| Timestamp media | frame1=0ms, frame48=1958ms |
| JSONL | 48 records, tổng12000 detections qua cutoff0.10 |
| Confidence thực | min0.1065168455, max0.8924665451 |
| Classes thực xuất hiện | car/van/truck/bus/motor/tricycle/awning-tricycle, IDs gốc |
| MP4 đọc lại | đủ48 frame, 640×360, encoded24fps |
| Geometry/mapping/confidence/count checks | passed, verification.json |
| Smoke pipeline elapsed | 136.86668s, detector adapter total135567.521ms |

24fps trong bảng là tốc độ timeline/encode video, **không phải processing FPS**.
Elapsed smoke bao gồm decode/inference/render/IO/finalize, không bao gồm detector
khởi tạo trước pipeline; không warmup theo benchmark protocol. Không dùng run
này làm steady-state performance benchmark hoặc đối chiếu FPS với Tesla T4.

Source SHA256:
`8d6677b4c26270c1136ff3ac133d4350090c0e944e7bdbe23049bf07a54810c3`.
Output MP4 SHA256:
`93ff87d4032c2f8f8de9706ede3c52cd939ca00b6907e7a797ed592df53c5c3f`.

Artifact local trong `../outputs/phase1_real_cpu_final/`:
annotated.mp4, detections.jsonl, manifest.json, verification.json, preview.jpg.
Manifest có resolved detector profile, environment, seed, source/model/config
hash, source commit, precision/cutoff và frame limit. Outputs bị gitignore;
cần lưu/copy chúng khi bàn giao evidence, không chỉ commit report.

## Tests và giới hạn chưa kiểm chứng

Tests bao phủ abstract interface, explicit mock, empty/invalid frames, class/score/
geometry, wrong preprocessing/config/mapping, dirty/foreign source, missing
checkpoint, tampered checksum bị chặn trước weight load, CPU AMP/CUDA-unavailable,
strict real load, BGR→RGB/scale, native batch, real forward trên frame giao thông;
clock/EOF/truncated decode/release/error-finalization, MP4 roundtrip, mock CLI,
RTSP/webcam qua capture doubles, GPU guard/split leakage/DET-as-MOT rejection và
notebook syntax/no-output. Diagnostic `moov atom not found` đến từ test MP4 rỗng
cố ý; test xác nhận lỗi này, không phải lỗi smoke video thật.

Chưa thử thiết bị webcam hoặc RTSP server thật; tests live dùng capture doubles.
CUDA/AMP và GPU Kaggle chưa được chạy; local CUDA unavailable. Notebook01 kiểm
tra GPU/assets/video rồi gọi cùng source; templates02–04 dừng khi thiếu runner.
Giữ status pending thay vì tuyên bố Kaggle hoàn thành từ việc tạo notebook.

GT MOT với persistent IDs/ignore/occlusion và validation/test manifest chưa có.
Clip smoke chưa có tracking GT, nguồn gốc dataset/annotation không được giả định
từ nội dung video. Tổng12000 raw detections không phải12000 xe; cutoff thấp và
top-query output có nhiều candidates/false positives, chưa đánh giá precision/AP.
Visual QA preview cũng thấy nhiều box chồng nhau và candidates ở vùng cây/mái
nhà; giữ hiện tượng này trong failure analysis, không coi video có box là bằng
chứng detector đúng. Cần GT thật để định lượng lỗi và đánh giá tracking sau đó.
Không báo HOTA/IDF1/MOTA/ID switches/recovery hoặc hiệu quả che khuất từ smoke này.
MP4 writer hiện CFR/no audio, reject odd size và PTS/VFR bất thường; không crop/
retime âm thầm. Clean-install môi trường khác và final runtime benchmark pending.

## Bước tiếp theo phù hợp

Chạy notebook01 với bundle và video thật trên Kaggle GPU, lưu verification và
environment. Đây là phần chưa có bằng chứng đầu tiên của bước1. Sau gate đó mới
làm ByteTrack/BoT-SORT chuẩn, rồi **đo MOT baseline trên validation trước OAR**.
Không phát triển Đồ án3 trong đợt này.
