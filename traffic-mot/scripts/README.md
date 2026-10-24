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

```powershell
python traffic-mot/scripts/run_video.py --help
python traffic-mot/scripts/run_video.py --source <traffic.mp4> --device cpu --save-video --save-json --max-frames 48
python traffic-mot/scripts/build_kaggle_notebooks.py
```

CLI chưa có, được bổ sung theo gate mới: run_mot_baselines.py/evaluate_mot.py
(bước2–3), run_ablation.py (bước5), export_events.py (bước6), final_evaluation.py/
benchmark.py (bước7). Notebook02–04 dừng rõ ràng khi entrypoint chưa tồn tại;
không dùng ByteTrackLite/MOT-lite để thay metrics chuẩn.
