# Dữ liệu đầu vào

Đã kiểm tra ngày24/10/2026: hai ZIP `D:/VisDrone2019-MOT-val.zip` và
`D:/VisDrone2019-MOT-test-dev.zip` có chuỗi JPG và GT MOT10 cột. Chúng nằm ngoài
Git; trên Kaggle dùng Add Input. Không dùng VisDrone2019-DET làm GT tracking.

Tạo manifest: `python traffic-mot/scripts/prepare_mot_dataset.py --root <dataset_mount>
--output <new_manifest.json>`. Xem notebooks/kaggle/README.md và báo cáo
MOT_DATA_PREPARATION_REPORT trong docs để biết scope đã kiểm tra. Chưa có kết quả
MOT metrics; runner vẫn thiếu. Raw GT ID0 hợp lệ, FPS chưa xác định từ chuỗi JPG.
