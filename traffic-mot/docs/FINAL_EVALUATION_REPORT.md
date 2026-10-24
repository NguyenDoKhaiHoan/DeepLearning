# Nối ablation → frozen test evaluation — 24/10/2026

Yêu cầu: đoạn code gọi final_evaluation.py phải dùng được sau phần ablation.
Trước thay đổi này script chưa tồn tại và notebook04 hardcode thư mục ablation,
trong khi notebook03 thường tạo ablation_<uuid>. Đã sửa cả đường dẫn và runtime.
Không tạo selected_config.yaml giả từ smoke để vượt prerequisite.

## Các thành phần đã tạo/sửa

- `evaluation/final.py`: resolver, kiểm provenance/full-validation selection,
  selected-config test execution, benchmark model thật.
- `scripts/final_evaluation.py`: config/detector-config/manifest/split/output-root
  theo đúng cell người dùng, thêm check-only/benchmark/subset/video options.
- Shared engine chuyển thành `_run_tracking_experiments` với selected split;
  public `run_baselines` vẫn chặn test, ablation vẫn validation-only. Không copy
  detector/tracking/GT preprocessing/evaluation vào final script hoặc notebook.
- Notebook04 dùng ABLATION_RUN hoặc unique frozen artifact, output final UUID,
  gọi script thật; cùng source từ notebook02/03.
- Tests freeze/selection/negative cases/benchmark plumbing, README/status/plan.

## Gate trước test và behavior

Resolver ưu tiên ABLATION_RUN tường minh từ notebook03; kernel mới có thể đặt
đường dẫn full-validation run. Không có variable thì tìm đúng một file ở
ablation/selected_config.yaml hoặc ablation_*/selected_config.yaml. Nhiều file
yêu cầu chọn run cụ thể, không tự chọn winner dựa vào test. Candidate-only giải
thích cần FULL_VALIDATION=True và baseline reference cũng phải full.

Final CLI xác minh frozen=true/smoke=false/selection_split=validation; checksum
selected YAML và source run manifest; completed real occlusion_ablation record;
frame counts đủ **toàn bộ validation khai báo trong source manifest**; manifest
data hash; winner đúng validation HOTA→IDF1→ít IDSW→baseline tie preference.
Profiles detector/tracker/OAR/evaluator phải khớp snapshots. Checkpoint/config
hash, source/architecture preflight và decision-code hash phải khớp freeze.
JSON snapshot stringify class-map keys được canonicalize khi compare YAML,
không thay semantic model class IDs. Shared execution orchestration có thể thêm
test mode, nhưng detector adapter/backend bridge/OAR decision code không được
thay từ validation mà âm thầm dùng lại freeze.

Giữ cả ablation manifest/selection/config snapshots cạnh frozen YAML; không chỉ
copy một YAML sang kernel mới. Final đọc đúng1 selected experiment E0/E1/E2/E3
và không tối ưu tham số trên test. Frozen E0/E1 dùng stock backend, E2/E3 giữ
OAR policy đã chọn, ReID off như validation. Original frozen config không bị sửa.
GT test được đọc **sau** freeze gate; output-root không overwrite.

Default test toàn split17 sequences; --sequence/--max-frames tạo subset được
gắn smoke_test=true nhưng vẫn yêu cầu full-validation freeze. Diagnostics và
metrics giữ protocol nghiên cứu/TrackEval như baseline; chưa claim official
challenge equivalence. Optional MP4 sử dụng playback FPS tường minh như baseline.

## Benchmark và output

Sau evaluation, default benchmark100 measured frames với warmup từ frozen
evaluation profile (mặc định30). Tái chạy detector thật + selected tracker,
không dùng cached replay làm FPS. Tách decode, detector (preprocess/forward/
postprocess/device sync), tracking gồm OAR, core total; mean/p50/p95/p99 và
measured_core_fps từ tổng interval đo thật. Reset state khi đổi sequence;
warmup một lần, cold sequence state mới nếu đi qua boundary được tính/ghi rõ.

Scope không gồm rendering/encoding/export/event engine; không tự gọi là full
video-output throughput, camera FPS hoặc GPU benchmark nếu chạy CPU. Native
JPG FPS vẫn chưa biết. Nếu subset quá ngắn sau warmup, fail rõ hoặc người dùng
chọn --benchmark-frames0 để chỉ eval; không bịa sample. Benchmark failure đánh
dấu run incomplete, không ghi final_summary hoàn chỉnh.

Output: manifest/source/config snapshots, selected experiment JSONL+MOT TXT,
raw/perclass/sequence metrics, diagnostics, freeze_verification.json,
selected_config snapshot, final_summary.json; optional benchmark.json +
benchmark_timings.csv và MP4 khi bật. Trong manifest lưu hash nguồn, code status,
environment như shared runner. Cache creation/evaluation không được dùng thay
measurement benchmark riêng.

## Kiểm thử đã chạy và kết quả thật

```powershell
traffic-mot/outputs/baseline_env/Scripts/python.exe -m pytest -c traffic-mot/pyproject.toml traffic-mot/tests --run-model -q
```

**135 passed, không skip,19.19s.**17 tests mới kiểm resolution unique/explicit,
smoke reject, full validation positive gate trên fixture có annotation tách,
checksum/run/data/profile/code/winner tampering, JSON/YAML ID mapping, selected
E2/E3 policy preservation, exact CLI options, failure state và benchmark direct
execution counts/timing scopes. Các fixture freeze/test là **unit fixtures**,
không đại diện validation/test run thật hoặc checkpoint giả thay model thật.

Đã chạy shared selected E1 execution trên **validation** sequence
uav0000305_00000_v frame1–48 dùng cache model thật của lần baseline trước.
Output `traffic-mot/outputs/final_engine_validation_smoke_20261024/`.
HOTA0.6806802087, IDF10.7740520213, MOTA0.5260869565, IDSW23 khớp baseline E1;
đây vẫn là validation prefix smoke, **không phải held-out final score**.
GT/model/cache hashes giữ nguyên bằng chứng MOT_BASELINE_REPORT.

Benchmark component trên cùng validation sequence **chạy model thật không cache**,
8 frame warmup +8 measured, torch2.6 CPU FP32, threads4:

| Mean đo thật | ms |
|---|---:|
| Decode | 12.5660 |
| Detector gồm pre/post/forward | 1116.4796 |
| Tracking (selected stock BoT-SORT) | 58.6313 |
| Core total | 1187.6770 |

Measured core throughput0.84198fps trong sample8 frame này; không khái quát thành
GPU/whole-pipeline/dataset benchmark. Scope/timing rows có trong benchmark.json
và benchmark_timings.csv, environment trong manifest.json. Detector nạp EMA checkpoint thật, không
MockDetector. Benchmark command được gọi từ Python qua benchmark_selected với
split=validation; production final CLI luôn split=test.

Đã chạy CLI --check-only với **candidate_config.yaml thật của ablation smoke**:
trả error requires frozen full-validation selection, không tạo final output,
không parse test GT hay infer. Workspace chưa có local full-validation artifact;
vì vậy **chưa chạy production final_evaluation trên test thật**. Test-dev được
giữ nguyên, không dùng để chọn/tune policy. Kaggle full validation/final evidence
cần chính người dùng chạy notebook với artifact đủ điều kiện rồi giữ raw outputs.

## Cách dùng sau notebook03

```python
from pathlib import Path
from uuid import uuid4
from traffic_mot.evaluation.final import resolve_frozen_config

FROZEN_CONFIG = resolve_frozen_config(OUTPUT_ROOT, globals().get("ABLATION_RUN"))
FINAL_RUN = OUTPUT_ROOT / ("final_" + uuid4().hex)
run_stage(PROJECT_ROOT, "final_evaluation.py", [
    "--config", str(FROZEN_CONFIG), "--detector-config", str(CONFIG_PATH),
    "--manifest", str(MOT_MANIFEST), "--split", "test",
    "--output-root", str(FINAL_RUN)])
print("DONE. Outputs:", FINAL_RUN)
```

Nếu kernel mới, gán ABLATION_RUN trước cell cho đúng directory; CONFIG_PATH là
cùng resolved detector profile đã dùng validation. Không cần print GT test
preflight trước freeze gate vì engine đã làm sau gate. Cài cùng pinned runtime/
TrackEval dependencies của notebook02/03; không cần model ReID mới hoặc dịch vụ
Đồ án3. Trajectory/events, full official challenge verification và Kaggle GPU
final evidence chưa được tuyên bố hoàn tất trong thay đổi này.
