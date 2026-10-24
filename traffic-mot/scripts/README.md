# CLI thực tế và trạng thái

Chạy từ repository root (`DeepLearning`). Mọi command trả exit nonzero khi lỗi;
real detector không fallback mock.

| File hiện có | Thực thi |
|---|---|
| validate_phase0.py | YAML cross-contract checks, tùy chọn --check-assets đọc/hash bundle |
| run_video.py | detection MP4/webcam/RTSP → annotated MP4/JSONL/manifest; chưa tracking |
| verify_detection_run.py | đọc lại toàn bộ JSONL/video, kiểm tra count/class/confidence/box, lưu verification.json |
| build_kaggle_notebooks.py | tạo lại 4 notebook thin wrappers, không chạy hoặc tạo metric |
| prepare_mot_dataset.py | tạo manifest native JPG sequences từ val/test-dev ZIP hoặc directory; preflight selected GT |
| run_mot_baselines.py | validation E0/E1 từ cùng cache detector thật, native JPG/ZIP, official TrackEval metrics |
| run_ablation.py | validation E0–E3 + hai component ablations mỗi backend; OAR/inference tách GT diagnostics |

```powershell
python traffic-mot/scripts/run_video.py --help
python traffic-mot/scripts/run_video.py --source <traffic.mp4> --device cpu --save-video --save-json --max-frames 48
python traffic-mot/scripts/build_kaggle_notebooks.py
```

```powershell
python traffic-mot/scripts/run_mot_baselines.py --detector-config <resolved_detector.yaml> --manifest <native_manifest.json> --split validation --output-root <new_directory> --max-frames 48 --sequence <validation_sequence>
```

Omit sequence/max-frames cho full validation. Optional save-video cần
visualization-fps tường minh; cache reuse cần detection-cache trỏ complete cache.
GT đọc sau tracking, schema/config/data/frame hashes được lưu. Runner chỉ nhận
validation, test dành cho final frozen-config runner.

```powershell
python traffic-mot/scripts/run_ablation.py --config traffic-mot/configs/experiments.yaml --detector-config <same_detector_profile.yaml> --manifest <native_manifest.json> --split validation --baseline-run <completed_E0_E1_run> --output-root <new_directory>
```

Baseline reference dùng lại cache/range và verify stock metrics. Không reference
thì runner tự recompute E0/E1; smoke có thể chọn sequence/max-frames. Full-only
freeze, không test tuning. Source chung trong tracking.baselines, không copy
thuật toán trong script/notebook. Baseline E0/E1 recovery=false như trước.

CLI chưa có: standalone evaluate_mot.py (metric nằm trong shared runner),
export_events.py (bước6), final_evaluation.py/benchmark.py (bước7). Notebook04 vẫn
có guard chống chạy entrypoint chưa tồn tại.
