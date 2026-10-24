# Bổ sung baseline còn thiếu và kiểm chứng thật — 24/10/2026

Lỗi người dùng gặp là `Stage not implemented: run_mot_baselines.py`: notebook
đã gọi một entrypoint chưa tồn tại. Đã bổ sung CLI với đúng arguments đó, native
MOT image reader, stock ByteTrack/BoT-SORT, shared detection cache và official
TrackEval metrics. Notebook02 cập nhật smoke/full validation; notebook03/04 vẫn
chưa có OAR/ablation/final runner. Không đổi guard thành thông báo success giả.

## Implementation và phạm vi

`tracking/{base,backend_bridge,bytetrack,botsort,baselines}.py` dùng trực tiếp
`ultralytics==8.4.171`, `lap==0.5.12`, không YOLO detector. Association/Kalman/
lost/remove giữ backend gốc, không class gate/OAR. BoT-SORT sparseOptFlow GMC,
with_reid=false; không tự tải appearance model. Instance ID counter được swap
trong lock để hai tracker/camera xen kẽ không reset/collision ID của nhau.

`video/mot_sequence.py` đọc original JPG trong ZIP/thư mục, không đọc GT qua
inference interface. Timestamp nguồn chưa biết ghi null/time_basis=frame_index,
không giả lập FPS. `--save-video` cần `--visualization-fps` tường minh cho playback;
pad tối đa1 pixel phải/dưới khi encode odd size, không thay đổi ảnh/box evaluation.

Detector thật tạo cache một lần. Cache lưu full-float boxes/scores/original IDs,
frame/pixel checksum, model/config/adapter/manifest hash và exact frame range.
E0/E1 replay cùng cache, stock backend configs ghi vào từng experiment. Cache
incomplete/mock/tampered/sai nguồn bị lỗi; `--detection-cache` cho run mới chỉ
dùng nếu signature khớp. Output directory không ghi đè. Interrupted failure được
đánh dấu incomplete sau khi run khởi tạo, không được tái dùng như result hoàn chỉnh.

`evaluation/trackeval_runner.py` có VisDroneMOTAdapter xây dựng timeline per-class,
remap GT ID0 một lần, valid-match protection trước suppress explicit ignore,
và gọi official HOTA/Identity/CLEAR `eval_sequence` + combination methods.
Không tự viết lại HOTA/IDF1. Baseline đọc class scope từ evaluation.yaml, mặc định
4/5/6/9; class khác vẫn có trong raw predictions nhưng ngoài scope scoring.
Đây là protocol nghiên cứu `visdrone_vehicle_research_v1`, chưa xác nhận tương
đương toolkit challenge VisDrone. Cần đối chiếu trước khi gọi official challenge score.

TrackEval pin commit12c8791b303e0a0b50f753af204249e622d0281a, kiểm PEP610 install
provenance. Dtype np.float/np.int cũ được thay qua module-local NumPy view, không
đổi global NumPy/matching/metric formulas. Known CLEAR Frag undercount ở empty
prediction frames được ghi metadata, không đưa Frag vào summary. Raw metric vẫn
giữ đúng upstream để review; MT dùng >80%, ML <20% theo pin này.

## Lệnh thật đã chạy

Runtime kiểm thử nằm trong virtualenv riêng `traffic-mot/outputs/baseline_env`
có system-site-packages; chỉ cài bổ sung lap/TrackEval vào venv đó. Máy CPU,
torch2.6.0+cpu, threads4; không thực thi Kaggle GPU trong session này.

```powershell
traffic-mot/outputs/baseline_env/Scripts/python.exe traffic-mot/scripts/run_mot_baselines.py --detector-config traffic-mot/configs/detector.yaml --manifest traffic-mot/outputs/mot_preparation_20261024/mot_split_manifest.json --split validation --sequence uav0000305_00000_v --max-frames 48 --save-video --visualization-fps 10 --output-root traffic-mot/outputs/mot_baseline_smoke_20261024
```

Sau khi bổ sung evaluation config snapshot và disclosure Frag, chạy lại từ
**cùng cache detector thật**, không infer/sinh detection mới:

```powershell
traffic-mot/outputs/baseline_env/Scripts/python.exe traffic-mot/scripts/run_mot_baselines.py --detector-config traffic-mot/configs/detector.yaml --manifest traffic-mot/outputs/mot_preparation_20261024/mot_split_manifest.json --split validation --sequence uav0000305_00000_v --max-frames 48 --detection-cache traffic-mot/outputs/mot_baseline_smoke_20261024/detection_cache --save-video --visualization-fps 10 --output-root traffic-mot/outputs/mot_baseline_verified_20261024
```

Cache created48 real inference frames, EMA119 strict/checksum verified, same P2
preprocessing/source commit, cutoff0.10, no NMS added. Manifest/checkpoint hash:
`bd8bbed95bee31fae95d8bdead39f8ac072ea4fb48998339d7aef7988a1eca88`.
Selected GT annotation SHA256:
`6414b5f6ab2e35cb20d04a226438b58f69bb89ece8561fb471a8fa1b687a5280`.
Cache predictions file SHA256:
`edb79ab10dd8dc88decc415576e842a1635ccaf0936a9a8f1af66c10ef2bd4b3`.

## Kết quả thật — chỉ48-frame prefix, không đại diện toàn split

Sequence uav0000305_00000_v có184 frame; lần này chỉ đánh giá frame1–48.
GT scope4/5/6/9 trong đoạn có1380 observations. split=validation,
smoke_test=true ở manifest và metrics. Không xem test-dev hoặc tune threshold.

| Stock implementation | HOTA | IDF1 | MOTA | IDSW | MT | ML | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 ByteTrack | 0.678811 | 0.783375 | 0.531159 | 15 | 27 | 3 | 1272 | 524 | 108 |
| E1 BoT-SORT GMC/no ReID | 0.680680 | 0.774052 | 0.526087 | 23 | 29 | 2 | 1280 | 531 | 100 |

HOTA/IDF1/MOTA là ratios, không percentages trong file; IDSW/MT/ML/TP/FP/FN là
counts. Không suy BoT-SORT hay ByteTrack tốt hơn toàn dataset từ48 frames này.
Raw observed records cả8 vehicle classes trước ignore: E0=5264, E1=5394;
không nhầm số records với số xe độc lập hoặc GT-scored TP.
Visual QA cũng thấy box chồng và false candidates ở nền/cây. Giữ output stock
để baseline trung thực; chưa thêm NMS, sửa threshold hoặc gọi video này là đã
giải quyết che khuất. Cần full validation và failure analysis trước OAR.

Hai run cho **tracks/boxes/classes/scores giống hệt** khi so per-frame observations
(run/session UUID khác hợp lệ). MP4 E0/E1 đều đọc lại48 frame, encoded1904×1072
do input1904×1071 được pad1 row cho visualization; encoded10fps là playback,
không tốc độ camera hoặc performance. First run elapsed82.87346s bao gồm load/
cache creation/replay/eval/IO sau preflight; detector adapter total58.66916s.
Đây là cold smoke, không warmup/final benchmark, không báo pipeline FPS từ nó.

## Artifacts và kiểm thử

`traffic-mot/outputs/mot_baseline_verified_20261024/`:

- manifest.json, source_manifest.json, detector_profile.json, evaluation_profile.json.
- summary.json/summary.csv, verification.json (readback+same-observations check).
- E0/E1: tracker_config.json, <sequence>.jsonl (full-float class sidecar),
  mot/<sequence>.txt, metrics.json (per-sequence/class/raw/combined), preview JPG/MP4.
- Detection cache giữ trong first run `mot_baseline_smoke_20261024/detection_cache`;
  verified run ghi đường dẫn/hash reference; lưu cả cache khi export evidence.

Output bị gitignore; GitHub lưu source/tests/report, không chứa dataset/video/
checkpoint hoặc tự nhận đã bàn giao artifact Kaggle. Code version/source dirty
state lúc chạy, environment và hashes có trong manifest.

Full regression:

```powershell
traffic-mot/outputs/baseline_env/Scripts/python.exe -m pytest -c traffic-mot/pyproject.toml traffic-mot/tests --run-model -q
```

**91 passed, không skip,18.12s**.29 tests mới bao gồm stock-backend parity cho
cả hai tracker, high/low/no-birth/empty/lost/remove/tentative/class-switch,
interleaved instance ID state, ReID rejection, timestamp monotonicity;
perfect/missing/FP/ID-switch/global identity/ignore protection/class mismatch/
empty/duplicate/timeline/combination cases; direct toolkit comparison; cache
tamper/signature/mock rejection, native JPG/odd size, CLI và test-split guard.
Initial fragment test phát hiện upstream limitation, sau đó được disclosure
và regression riêng, không sửa metric để tạo số đẹp.

## Cách chạy Kaggle và bước tiếp theo

Git pull version mới; cài pinned runtime và requirements-eval.txt trong cùng
kernel. Notebook02 có `FULL_VALIDATION=False` để smoke một sequence48 frame;
khi inspect outputs đạt thì True cho đủ7 sequences. CLI cũ trong traceback đã
được triển khai nên không còn Stage not implemented cho baseline. Nếu dùng cell
cũ, nhớ output-root phải là directory chưa tồn tại; chọn tên/UUID mới khi rerun.

Full Kaggle validation/GPU evidence, occlusion/size diagnostics chuyên sâu,
recovery success, E2/E3/OAR, ablation, trajectory/events và final-test benchmark
còn pending. Notebook03/04 vẫn templates/guards. Không triển khai Đồ án3 trong
thay đổi này; không bypass test freeze hoặc báo metric từ mock.
