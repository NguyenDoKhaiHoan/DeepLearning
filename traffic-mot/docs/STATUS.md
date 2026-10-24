# Trạng thái thực tế — 24/10/2026

Ưu tiên đề tài CV; chưa chuyển sang Đồ án3. Giữ module cũ, phát triển phần còn
thiếu theo IMPLEMENTATION_PLAN, không dùng tên phase/folder làm chứng cứ chạy.

| Bước | Trạng thái có thể chứng minh | Bằng chứng / còn thiếu |
|---|---|---|
| Phase0 | local verified | audit/docs/config/schema, PHASE_0_REPORT |
| 1a Detection local | model/video/export đã chạy thật CPU | PHASE_1_REPORT, run manifest/MP4/JSONL |
| 1b Detection Kaggle | code/notebook/preflight chuẩn bị | chưa có phiên Kaggle GPU thực tế |
| 2 Standard MOT baseline | đã triển khai + real CPU smoke | backend stock8.4.171, E0/E1, same detection cache, ReID/OAR off |
| 3a MOT data preflight | đã kiểm tra local val/test-dev thật | 7/17 sequences; manifest và validation GT preflight đạt |
| 3b MOT baseline evaluation | implementation + 48-frame real CPU smoke đạt | TrackEval HOTA/Identity/CLEAR; full Kaggle validation pending |
| 4 OAR | implemented + controlled invariant tests + real48-frame smoke | bounded retention, gated reassociation; chưa có improvement evidence |
| 5 Kaggle ablation | runner E0–E3 + components, real CPU cache replay đạt | full Kaggle validation/frozen config còn pending |
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
Dataset đã có inference/tracking/eval smoke48 frame sequence uav0000305_00000_v;
chưa có toàn bộ validation hoặc GPU benchmark. Xem MOT_BASELINE_REPORT.
Kaggle vẫn cần Add Input và tạo manifest với mount path thực tế.

**Bước tiếp theo phù hợp:** cập nhật code/dependencies, chạy notebook02 smoke rồi
full validation trên Kaggle, lưu raw predictions/metrics/environment. Notebook01
GPU evidence chưa được cung cấp lại ở workspace này; không tự suy rằng đã đạt.
User đã chạy MOT preflight trên Kaggle (log24/10), nhưng chưa gửi summary/raw
baseline đầy đủ. Để sửa lỗi missing run_ablation.py, đã thêm OAR/runner và local
smoke eight variants trên cùng cache. E0/E1 tái tính khớp; E2/E3 HOTA/IDF1 giảm
nhẹ trong đoạn48 frame, không công bố improvement. Xem OAR_ABLATION_REPORT.
Notebook03 có smoke/reference/full modes; chỉ full validation mới freeze.
Notebook04 vẫn template có guard; không tự suy Kaggle ablation đã thành công.
