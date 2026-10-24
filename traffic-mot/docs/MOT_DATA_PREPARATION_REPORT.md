# Kiểm tra dữ liệu MOT thật và tạo manifest — 24/10/2026

Sau khi người dùng cung cấp ảnh Kaggle Upload hai archive, đã tìm thấy và kiểm
tra chính hai tên archive đó trên máy. Không truy cập dataset Kaggle private,
không upload thêm dữ liệu, không giả định server đã giải nén xong từ screenshot.

| Archive local | Sequence/GT files | Image frames | SHA256 |
|---|---:|---:|---|
| D:/VisDrone2019-MOT-val.zip | 7 | 2846 | e53571990dfc79229e0a8ae10264bc4fa604a027c44b06e3a097417e4fa55705 |
| D:/VisDrone2019-MOT-test-dev.zip | 17 | 6635 | 758abe40bf20246e7e778ac61eaa557cf004b034b6e68e85006e5add68e17eb5 |

Archive gồm `VisDrone2019-MOT-<split>/sequences/<sequence>/<frame>.jpg` và
`annotations/<sequence>.txt`. Frame filenames bắt đầu0000001, tăng liên tục.
Annotation10 cột có persistent target IDs, bao gồm ID0 hợp lệ. Shape/cột annotation
được spot-check cả hai archive; chưa chạy model/evaluator hay chọn tham số bằng
nhãn test-dev. Full preflight chỉ parse GT và ảnh đầu mỗi **validation** sequence.

## Lỗi phát hiện và thay đổi

Checker cũ chỉ nhận `video` + `gt` file, và buộc target_id>=1. Điều này không
phù hợp native VisDrone MOT: chuỗi JPG và GT có ID0. Đã sửa để hỗ trợ
`source_kind=images_zip` hoặc `images_dir`; giữ raw ID0 (public tracker IDs dương
vẫn là contract riêng), không chuyển video hoặc suy FPS không có căn cứ.

`utils/mot_data.py` tạo và kiểm tra manifest. Chuẩn bị chỉ đọc index/hash archive,
không parse nhãn test. Kiểm tra disjoint sequence/source UAV, frame timeline,
archive hash/index fingerprint, decode ảnh đầu và GT selected-split. Legacy video
manifest vẫn được hỗ trợ. Notebook02–04 có cell gọi cùng utility này; notebook01
không cần MOT manifest. Không nhúng thuật toán vào notebook.

Script không giải nén, không chỉnh dataset hoặc ghi đè manifest/report. Hai loại
source trong cùng manifest không được trộn với legacy video entries. Khi ZIP
và bản giải nén đều tồn tại, dùng explicit sources để tránh chọn sai bản.

## Lệnh thực tế và đầu ra local

```powershell
python traffic-mot/scripts/prepare_mot_dataset.py --validation-source D:/VisDrone2019-MOT-val.zip --test-source D:/VisDrone2019-MOT-test-dev.zip --output traffic-mot/outputs/mot_preparation_20261024/mot_split_manifest.json
```

Kết quả exit0, validation preflight passed. File:

- `traffic-mot/outputs/mot_preparation_20261024/mot_split_manifest.json`.
- `traffic-mot/outputs/mot_preparation_20261024/mot_split_manifest.preflight.json`.

Manifest ánh xạ official val→validation và official test-dev→held-out test.
Validation GT: 114132 valid observations thuộc758 identities khi cộng từng
sequence, trong đó987 observations có target_id0. Đây là **thống kê annotation**,
không phải TP/recall/detection/tracking metrics. Native image FPS để null.
Ảnh đầu các validation sequence decode được; độ phân giải từ1344×756 tới3840×2160,
có sequence1904×1071, nên không được crop/rescale âm thầm để ép MP4 dimensions.

Trên Kaggle chạy lại script với `--root <actual_input_mount>` và
`--output /kaggle/working/mot_split_manifest.json`. Không upload nguyên manifest
local có đường dẫnD:/ rồi dùng trên Linux. Script nhận ZIP hoặc thư mục release
đã giải nén. Nó chưa nối native-image inference vào tracking runner vì runner
đó chưa được triển khai.

## Kiểm thử và giới hạn

`test_mot_data.py` kiểm tra ZIP/extracted formats, raw GT0, không đọc test GT khi
validation, missing/overwrite/checksum/frame gaps, cùng UAV qua hai split, invalid
GT/DET rows. Fixture dùng cho test checker, không báo thành dữ liệu thực nghiệm.
Lệnh focused data/Kaggle tests chạy16 tests đạt. Full regression với
`python -m pytest -c traffic-mot/pyproject.toml traffic-mot/tests --run-model -q`
đạt62 tests, không skip, 17.83s; bao gồm strict load và inference checkpoint thật.

Preflight chưa decode mọi ảnh hoặc đánh giá ignore/class/challenge metrics; index
fingerprint ZIP dùng filename/size/CRC, không phải perceptual pixel hash. Hash
archive bảo vệ integrity theo manifest, không chứng minh dataset do ai cung cấp.
GT scope/class filtering/ignore remapping cần evaluator chuẩn và test riêng.
Không có HOTA/IDF1/MOTA/IDSW/FPS mới, không có recovery/ablation results.
Notebook02–04 tiếp tục dừng ở guard nếu thiếu runner, không fake success.

Nguồn mô tả split: [VisDrone official MOT task](https://github.com/VisDrone/VisDrone-Dataset#task-4-multi-object-tracking)
công bố val và test-dev, test-dev có GT. Nội dung thực tế của hai archive local
được kiểm tra trực tiếp, số lượng trong bảng lấy từ archive, không suy từ website.
