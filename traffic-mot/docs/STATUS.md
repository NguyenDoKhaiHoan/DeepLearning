# Trạng thái thực tế — 23/10/2026

Ưu tiên đề tài CV; chưa chuyển sang Đồ án3. Giữ module cũ, phát triển phần còn
thiếu theo IMPLEMENTATION_PLAN, không dùng tên phase/folder làm chứng cứ chạy.

| Bước | Trạng thái có thể chứng minh | Bằng chứng / còn thiếu |
|---|---|---|
| Phase0 | local verified | audit/docs/config/schema, PHASE_0_REPORT |
| 1a Detection local | model/video/export đã chạy thật CPU | PHASE_1_REPORT, run manifest/MP4/JSONL |
| 1b Detection Kaggle | code/notebook/preflight chuẩn bị | chưa có phiên Kaggle GPU thực tế |
| 2 Standard MOT baseline | chưa triển khai | chỉ configs; legacy lite không thay chuẩn |
| 3 MOT baseline evaluation | chưa triển khai/chưa chạy | thiếu dataset MOT/GT/split manifest và runner |
| 4 OAR | thiết kế | chưa implementation hoặc run |
| 5 Kaggle ablation | template | chưa baseline/OAR/data/runner |
| 6 Trajectory/events | schema/config thiết kế | chưa engine/run |
| 7 Final eval/benchmark/demo | template | chưa frozen config/test run/benchmark |

Checkpoint P2 best.pth thực có, SHA256 khớp manifest; EMA119 load strict được.
Source RT-DETR đúng commit và sạch. Máy local torch CPU, CUDA unavailable.
Video smoke thật là prefix48 frame của clip897 frame, không có GT tracking;
không dùng nó để báo cáo HOTA/IDF1/MOTA/IDSW, AP hoặc recovery success.

**Bước tiếp theo phù hợp:** chạy notebook01 với bundle/video thật trên Kaggle GPU,
lưu environment/checksums/outputs và đọc lại video. Chỉ khi gate bước1 đạt mới
triển khai ByteTrack/BoT-SORT chuẩn; tiếp theo là MOT baseline evaluation trước
khi nghiên cứu OAR. Các notebooks02–04 hiện có guard chống chạy CLI chưa tồn tại.
