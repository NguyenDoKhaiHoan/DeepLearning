# Trạng thái thực tế — 24/10/2026

Ưu tiên đề tài CV; chưa chuyển sang Đồ án3. Giữ module cũ, phát triển phần còn
thiếu theo IMPLEMENTATION_PLAN, không dùng tên phase/folder làm chứng cứ chạy.

| Bước | Trạng thái có thể chứng minh | Bằng chứng / còn thiếu |
|---|---|---|
| Phase0 | local verified | audit/docs/config/schema, PHASE_0_REPORT |
| 1a Detection local | model/video/export đã chạy thật CPU | PHASE_1_REPORT, run manifest/MP4/JSONL |
| 1b Detection Kaggle | code/notebook/preflight chuẩn bị | chưa có phiên Kaggle GPU thực tế |
| 2 Standard MOT baseline | chưa triển khai | chỉ configs; legacy lite không thay chuẩn |
| 3a MOT data preflight | đã kiểm tra local val/test-dev thật | 7/17 sequences; manifest và validation GT preflight đạt |
| 3b MOT baseline evaluation | chưa triển khai/chưa chạy | tracker chuẩn/evaluator/runner còn thiếu |
| 4 OAR | thiết kế | chưa implementation hoặc run |
| 5 Kaggle ablation | template | chưa baseline/OAR/data/runner |
| 6 Trajectory/events | schema/config thiết kế | chưa engine/run |
| 7 Final eval/benchmark/demo | template | chưa frozen config/test run/benchmark |

Checkpoint P2 best.pth thực có, SHA256 khớp manifest; EMA119 load strict được.
Source RT-DETR đúng commit và sạch. Máy local torch CPU, CUDA unavailable.
Video smoke thật là prefix48 frame của clip897 frame, không có GT tracking;
không dùng nó để báo cáo HOTA/IDF1/MOTA/IDSW, AP hoặc recovery success.

Ngày24/10 tìm thấy `D:/VisDrone2019-MOT-{val,test-dev}.zip` và kiểm tra index:
2846/6635 ảnh, 7/17 annotation files. Tạo native manifest, đọc toàn bộ val GT
và decode ảnh đầu mỗi val sequence. Raw target ID0 hợp lệ đã được checker sửa;
không ép dữ liệu thành video/FPS. Evidence ở MOT_DATA_PREPARATION_REPORT.
Dataset này mới đủ cho data preflight, chưa có kết quả inference/tracking/eval
trên MOT. Kaggle vẫn cần Add Input và tạo manifest với mount path thực tế.

**Bước tiếp theo phù hợp:** chạy notebook01 với bundle/video thật trên Kaggle GPU,
lưu environment/checksums/outputs và đọc lại video. Chỉ khi gate bước1 đạt mới
triển khai ByteTrack/BoT-SORT chuẩn; tiếp theo là MOT baseline evaluation trước
khi nghiên cứu OAR. Các notebooks02–04 hiện có guard chống chạy CLI chưa tồn tại.
