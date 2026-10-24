# OAR và ablation runner — 24/10/2026

Người dùng đã chạy data preflight validation trên Kaggle và gặp lỗi
`Stage not implemented: run_ablation.py`. Đã bổ sung entrypoint đó với đúng CLI
arguments, OAR versioned, component studies và diagnostics. Log của người dùng
chỉ xác nhận native images/GT, không chứng minh full baseline hay GPU metrics.
Notebook04/final test vẫn chưa được triển khai, không bỏ guard để báo success giả.

## Thuật toán đã triển khai

`occlusion/evidence.py` lưu bounded deque của overlap history. IoA có hướng
track-area, ít nhất2/5 frames overlap>=.50 là hypothesis occlusion; không suy
depth ordering từ overlap. Border3% + outward velocity là suspected exit;
không đủ evidence là unknown, không gọi detector miss là occlusion thật.
Chỉ track được quan sát gần nhất làm occluder; không dùng lost ghost tạo evidence.

`occlusion/recovery.py` tạo mixin subclass cho đúng stock ByteTrack/BOTSORT8.4.171.
Stock Kalman prediction, GMC, high/low/unconfirmed association chạy một lần/frame.
OAR geometry tính sau stock GMC trong hệ tọa độ hiện tại. Hook trước births chỉ
xét lost eligible + unmatched high detections, không cướp active match.
Class gate, min3 observed hits, gap/covariance cap, IoU>=.10 và squared4D
Mahalanobis<=9.4877; cost=.7(1-IoU)+.3*d²/gate, one-to-one Hungarian,
mutual-best row/column ambiguity margin.15. Reactivation giữ ID, không birth
rồi đổi ngược public identity đã xuất. No GT/ReID/appearance input.

Adaptive retention chỉ cho occlusion-evidenced tracks từ base30 tới max90 frame,
loại exit/uncertain candidates; expiration trước high association để không
revive extended track vượt cap. Bounded candidate64 và history theo live/lost
pool. Trace mỗi frame có row cap128 + truncation counter, không tăng bộ nhớ
theo tổng số frame. Rule max_gap_ms3000 chỉ enforce khi source timestamp biết;
native JPG giữ null/frame_index, không gán FPS giả. Frame/covariance bounds luôn
áp dụng. Lost state không tạo observed detections hoặc traffic events.

E0/E1 tạo backend stock, OAR không chạy. E2/E3 thêm full policy. Component runs:
BT_retention_only, BT_reassociation_only, BS_retention_only, BS_reassociation_only.
OAR policy/config được snapshot và version, implementation hashes trong run mới.
`tracking/baselines.py` được tái sử dụng, không có bản sao thuật toán notebook.

## Runner, cache và selection

`scripts/run_ablation.py` chỉ nhận validation. Có --baseline-run thì inherit exact
sequence/frame range/cache của E0/E1 completed real run, verify config/scoring
params, cache manifest/data/pixel hashes, tái tính stock metrics và kiểm numeric
equality trước attributing OAR effect. Descriptive adapter_status không được xem
là scoring parameter; mọi parameter scoring vẫn phải khớp. Không reference thì
recompute E0/E1 từ một cache mới; không yêu cầu người dùng bịa baseline file.

Không thay detector/checkpoint/class mapping/precision/cutoff giữa experiment.
Stock+OAR dùng chung immutable cache; không load GT vào detector/tracker. GT được
đọc riêng sau predictions cho đánh giá/diagnostics. Cache E0/E1 có thể chứa dữ liệu
ngoài class scoring, nhưng evaluation scope giữ4/5/6/9 và ignore protocolv1.
Interrupted/failed study không được freeze, output-root không overwrite.

`evaluation/occlusion.py` match đầy đủ valid GT theo class trong từng frame rồi
phân tầng none/partial/heavy/unknown, original-area small/medium/large. Không lọc
heavy GT trước rematching và không đặt tên recall bin thành HOTA-heavy.
Episode denominator tạo độc lập từ GT: visible3 frame trước/sau một đoạn labeled
partial/heavy, cùng ID/class, consecutive timeline, no truncation. Unannotated
gaps bị loại, không tự nhận full occlusion. Success yêu cầu same pre-ID quay lại
và giữ3 observations trong window10; pre-ID thiếu tính failure, không loại mẫu.
Diagnostic này đo same-ID continuity sau labeled occlusion, kể cả uninterrupted
tracking, không chứng minh OAR là nguyên nhân của recovery.

Bootstrap1000 samples theo sequence, seed42; báo mean paired per-sequence
difference (khác global toolkit aggregate). Một sequence đánh dấu degenerate CI.
Selection chỉ so E0–E3, ưu tiên validation global HOTA, rồi IDF1, rồi ít IDSW,
baseline thắng khi hòa. Component runs phục vụ attribution, không thêm search
budget âm thầm. Full validation mới ghi selected_config.yaml frozen=true;
smoke ghi candidate_config.yaml frozen=false. Không test-driven tuning.

## Chạy thật và dữ liệu

Checkpoint EMA119 SHA256
`bd8bbed95bee31fae95d8bdead39f8ac072ea4fb48998339d7aef7988a1eca88`;
native validation sequence uav0000305_00000_v, prefix frame1–48/184, original
resolution1904×1071. Cache được tạo bởi model thật ở MOT_BASELINE_REPORT;
không synthesize detection hoặc checkpoint. 1380 scored GT observations trong
scope4/5/6/9. CPU torch2.6.0+cpu, no CUDA, backend/evaluator deps như baseline.

```powershell
traffic-mot/outputs/baseline_env/Scripts/python.exe traffic-mot/scripts/run_ablation.py --config traffic-mot/configs/experiments.yaml --detector-config traffic-mot/configs/detector.yaml --manifest traffic-mot/outputs/mot_preparation_20261024/mot_split_manifest.json --split validation --baseline-run traffic-mot/outputs/mot_baseline_verified_20261024 --output-root traffic-mot/outputs/oar_ablation_final_20261024
```

Đã chạy cùng nội dung experiment nhiều lần với output directory mới, E0/E1
khớp baseline reference và các observation/metrics của tám variants khớp khi
replay. Reuse real cache nghĩa là không infer lại detector trong ablation run;
manifest có cache_reused=true, detector provenance thực, mock=false, smoke=true.

| Experiment | HOTA | IDF1 | MOTA | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| E0 ByteTrack | .678811 | .783375 | .531159 | 15 | 524 | 108 |
| E1 BoT-SORT GMC/no ReID | .680680 | .774052 | .526087 | 23 | 531 | 100 |
| E2 ByteTrack+OAR | .677998 | .782144 | .527536 | 15 | 529 | 108 |
| E3 BoT-SORT+OAR | .680506 | .773810 | .525362 | 23 | 532 | 100 |
| BT_retention_only | .678811 | .783375 | .531159 | 15 | 524 | 108 |
| BT_reassociation_only | .677998 | .782144 | .527536 | 15 | 529 | 108 |
| BS_retention_only | .680680 | .774052 | .526087 | 23 | 531 | 100 |
| BS_reassociation_only | .680506 | .773810 | .525362 | 23 | 532 | 100 |

Ratios0..1, counts raw. **OAR không cải thiện HOTA/IDF1 trong đoạn này**, FP tăng
nhẹ. Không tune policy sau khi xem bảng smoke để tuyên bố improvement; cần full
validation và failure analysis. Candidate winnerE1 nhưng frozen=false, không có
selected_config.yaml. Smoke không đại diện cả dataset, không có final-test result.

Occlusion bins GT:none1096, partial188, heavy96. Heavy TP48 (recall.5) ở cả tám
variants; không tăng heavy recall trên đoạn này. Eligible GT recovery episodes=0
ở tất cả runs, cùng fingerprint denominator; recovery_success_rate=null/N/A,
không báo0 hoặc100%. E2 có19 và E3 có25 thao tác reassociation, nhưng đây là
algorithm decisions trên cả8 vehicle classes, không phải confirmed GT recoveries.
Native JPG không có timestamp nên milliseconds_cap_enforced=false được công bố.
Instrumented replay timing gồm OAR evidence/matching nhưng không warmup; không
chuyển thành pipeline/GPU FPS. ReID không tích hợp và không tuyên bố đã dùng.

## Artifacts và tests

Local `traffic-mot/outputs/oar_ablation_final_20261024/` có manifest/source manifest,
resolved detector/evaluation/experiment policy, summary JSON/CSV,
ablation_summary.json, paired_differences.json, selection.json và candidate_config.
Mỗi experiment có predictions JSONL + MOT TXT, metrics.json/raw/perclass/sequence,
diagnostics.json, recovery_statistics.json, preview. OAR còn có recovery decision
JSONL. Không có video mới khi không bật --save-video, không nhận đã xuất MP4 từ
run này. Các video E0/E1 baseline thật đã có riêng ở lần chạy trước.

verification.json của final local run xác nhận8 experiments×48 frames,
observations giống lần replay trước, cùng episode denominator và smoke không
freeze. Local ablation final elapsed38.77376s, cache_reused=true; không phải
steady-state benchmark hoặc detector end-to-end throughput.

Output/cache/data/model nằm ngoài Git; giữ cả baseline cache và manifest nếu
download Kaggle artifact. Không copy manifest chứaD:/ sang Kaggle nguyên trạng;
tạo manifest đúng mounts như notebook02. Không tạo kết quả score từ fixture tests.

Unit/integration tests kiểm tra: stock parity khi modules off, recover gap hơn
base buffer với overlap, không evidence không extension, expiration trước
revival, covariance/class/exit/time gates, unmatched reassociation, one predict,
one-to-one/ambiguity, reset/bounded state; GT-first bins và fixed denominator,
wrong-ID/missing pre-track/truncation/gap episodes; config isolation, no test
tuning, no overwrite, smoke không freeze và direct toolkit bootstrap aggregation.
Controlled fixtures chỉ chứng minh invariants; real GT/cached model output ở
bảng trên là bằng chứng smoke riêng.

Lệnh full regression:

```powershell
traffic-mot/outputs/baseline_env/Scripts/python.exe -m pytest -c traffic-mot/pyproject.toml traffic-mot/tests --run-model -q
```

**118 passed, không skip,19.08s**, gồm27 tests OAR/diagnostics/ablation mới và
regression của detection/baseline/data. Source/notebook syntax, unexecuted output
metadata và internal document links cũng qua kiểm tra. RT-DETR checkout vẫn sạch.

**Full validation Kaggle, positive improvement, eligible recovery episode samples,
final-test benchmark và notebook04 vẫn pending.** Phần mềm đã có để chạy notebook03,
không tự coi data preflight hoặc việc tạo file là hoàn thành đề tài.
