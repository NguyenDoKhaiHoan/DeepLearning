# Báo cáo Phase 0 — 23/10/2026

Đã hoàn tất phần kiểm kê, thiết kế và skeleton có thể kiểm thử. Dừng ở Phase 0
theo yêu cầu; chưa triển khai toàn bộ hệ thống hoặc bắt đầu Phase 1.

## File tạo/sửa

- Tạo `traffic-mot/docs/{ARCHITECTURE,IMPLEMENTATION_PLAN,MOT_EVALUATION_PLAN,
  INTEGRATION_SPEC,REPOSITORY_AUDIT,PHASE_0_REPORT}.md`.
- Tạo `configs/{detector,bytetrack,botsort,occlusion,events,pipeline,evaluation,
  experiments}.yaml`.
- Tạo skeleton 9 domain packages trong `src/traffic_mot/`, utility kiểm tra
  `utils/config.py`, CLI `scripts/validate_phase0.py` và tài liệu CLI.
- Tạo JSON schema/example cho track và event, tests Phase0, pyproject,
  requirements runtime/dev/eval, README, data/outputs placeholders và gitignore riêng.
- Thêm liên kết dự án từ README gốc; giữ nguyên nội dung và working tree có sẵn.

## Thiết kế thuật toán

Detector adapter dùng EMA checkpoint thực, strict load, đúng resize/RGB/scale và
postprocessor `[W,H]`. Mapping giữ model IDs3–10, ngưỡng0.10 để không làm mất
low-confidence band của ByteTrack. Backend pin cho baseline E0/E1, BoT-SORT ReID
off. Recovery dự kiến dùng evidence overlap+history, retention có giới hạn,
Kalman covariance gate và unmatched reassociation một-một có từ chối ambiguity.
Ground truth chỉ xuất hiện trong offline evaluation.

MOT protocol giữ timeline/ignore/class, dùng TrackEval metrics qua adapter riêng,
diagnostic occlusion phân tầng sau matching đầy đủ và GT episode denominator cố
định để đo recovery. E0–E3 cùng detector cache/split/precision; runtime tách từng
thành phần. Đây là thiết kế có ablation, chưa có kết luận tăng chất lượng.

## Kiểm tra đã chạy

Từ repository root:

| Lệnh | Kết quả |
|---|---|
| `python -m pytest -c traffic-mot/pyproject.toml traffic-mot/tests -q` | 11 passed, exit0 |
| `python traffic-mot/scripts/validate_phase0.py --check-assets` | 8 config passed, hash/labels/commit khớp, exit0 |
| `python -m unittest discover -s rtdetrv2_next -p test_demo.py -v` | 3 passed, strict EMA thật + transforms + mapping/scaling, exit0 |
| `python -m unittest discover -s rtdetrv2_next -p test_track.py -v` | 8 passed, hành vi tracker/evaluator legacy, exit0 |

22 tests đạt; không có test skip trong các lệnh trên. Fixture synthetic dùng kiểm
thử hợp đồng, không phải dữ liệu thí nghiệm thật. Không chạy training/notebook,
web suite, MOT evaluation, GPU benchmark hoặc toàn bộ test repository.

Phase0 validator không import torch/backend; tùy chọn asset chỉ hash và đọc
provenance. Tests detector cũ mới là lệnh load model thật, vẫn không đo AP/FPS mới.
Môi trường Python3.11.9/CPU, xem REPOSITORY_AUDIT.

## Giới hạn và việc cần làm tiếp

Chưa có GT MOT trong phần dữ liệu dự án; chưa cài lap/TrackEval, FFmpeg/ffprobe
không có trên PATH; chưa test clean-install đầy đủ hoặc CUDA/AMP. Schema hiện
kiểm tra cấu trúc, semantic validation sẽ làm ở Phase4. Các YAML runtime ngoài
validator là template cho phase tiếp theo; không mô tả chúng thành runtime đã có.
CLI inference/tracking/evaluation/export chưa được tạo, tránh nhận argument rồi
không thực thi. ReID chưa tích hợp; OAR chưa triển khai; không có số liệu HOTA,
IDF1 hay recovery mới. Cơ chế che khuất hoàn toàn không có bảo đảm giữ đúng ID.

Gate Phase0 đạt trong môi trường đã kiểm tra. Bước tiếp theo của kế hoạch là
Phase1: video reader, detector interface/mock và adapter RT-DETR thật với tests
CPU/ảnh không vuông/empty frames. Không tự chuyển phase trong đợt này.
