# Các bước tiếp theo: RT-DETRv2 trên VisDrone

## 1. Kết quả đã có và phạm vi kết luận

`baseline_results.csv` ghi lại số liệu từ bảng bạn cung cấp. Ảnh log bổ sung trong
`baseline_log.json` xác nhận AP và các chỉ số độ chính xác sau khi làm tròn;
chưa có đủ thông tin để xác nhận toàn bộ protocol và benchmark tốc độ.
RT-DETRv2 có AP = 0,2839, APsmall = 0,1942 và FPS = 46,0944.
So với YOLO11s, APsmall tăng 0,0712 (7,12 điểm phần trăm, khoảng 57,9% tương đối).
RT-DETRv2 có độ chính xác cao nhất trong ba mô hình, còn YOLO11s có FPS cao nhất.
Chưa có số liệu để kết luận mô hình nào tốt nhất trên phương tiện **bị che khuất**.

Ảnh log bổ sung có best AP = 0,28387425072039507 tại `epoch=119`, ARsmall = 0,348,
ARmedium = 0,566 và ARlarge = 0,667. Tổng 10 class có 38.759 GT, khớp thống kê
validation đã lưu; riêng 8 class phương tiện có 24.790 GT.
Lấy trung bình các AP đã làm tròn trong bảng, AP-vehicle xấp xỉ 0,2988 và
APsmall-vehicle xấp xỉ 0,1923. Đây là số suy ra từ bảng, cần evaluator/predictions để lấy
độ chính xác đầy đủ. Không suy ra Recall-heavy từ ARsmall.

| Nhóm cần ưu tiên phân tích | APsmall trong log |
|---|---:|
| bicycle | 0,0844 |
| awning-tricycle | 0,1003 |
| truck | 0,1219 |
| tricycle | 0,1615 |

Car có APsmall = 0,4253, cao hơn đáng kể các lớp trên. Cần xem ca bỏ sót/localization và
phân bố số GT-small/occlusion theo class trước khi gán nguyên nhân cho kiến trúc.
Thời gian `0,0848 s/it` trong log là thời gian vòng validation với batch size chưa rõ,
không dùng nó thay cho latency batch 1 hay FPS benchmark.

Kế hoạch Excel ban đầu chọn RT-DETRv3. Nhánh này lấy RT-DETRv2 làm baseline theo kết quả hiện có;
đây là điều chỉnh hướng thực nghiệm, không phải bằng chứng rằng v2 tốt hơn v3.
Nếu đề tài đã được duyệt bắt buộc dùng v3, cần giữ baseline v3 theo kế hoạch đó.

## 2. Chạy trên máy đang train

### Notebook Kaggle cho bước B–C

Mở [`rtdetrv2_p2_visdrone_kaggle.ipynb`](../rtdetrv2_p2_visdrone_kaggle.ipynb),
import lên Kaggle, bật Internet/GPU và gắn dataset VisDrone train/val gốc.
Notebook đã nhúng converter và các script P2/export/evaluate; không cần dataset chứa script.
Cell cấu hình mặc định train `E1_p2`, 120 epoch, 640×640, batch 2 và AMP.
Đặt `RUN_MODE="proxy"` để thử 2 epoch trước; chạy E0 với cùng mode/batch/seed để ablation.
Giữ category 1–10 và classifier 11 đầu ra theo notebook cũ, bỏ slot 0 tường minh khi export.
Chỉ resume `last.pth` cùng experiment/mode với config, manifest, log và best checkpoint đi kèm.
Kết quả gồm AP/APsmall toàn bộ và riêng phương tiện, Recall-heavy kèm GT, ảnh lỗi,
benchmark batch 1 và ZIP chứa artifact. Đánh giá B2 cũ bật riêng bằng `RUN_B2_EVALUATION`.

`build_kaggle_notebook.py` tái tạo notebook từ các script trong workspace.
`validate_kaggle_notebook.py --repo <rtdetrv2_pytorch>` kiểm tra các cell với ảnh giả trên CPU;
đây không phải bằng chứng train thành công trên VisDrone thật hay GPU Kaggle.

Dùng môi trường Python của repo RT-DETR đã train thành công. `--repo` trỏ tới thư mục
`rtdetrv2_pytorch` chứa `src/core` và `tools/train.py`. Các script không dùng Ultralytics.
Giữ phiên bản và các thay đổi của repo đang train; không cần thay repo bằng bản mới.

Ảnh terminal cho thấy máy train dùng Linux và môi trường `RTDETR_Env`. Trong môi trường đó,
có thể dùng các lệnh Bash dưới đây. Thay đường dẫn mẫu bằng đường dẫn thật; `repo` phải
trỏ tới thư mục chứa `src/core`, không chỉ thư mục clone ngoài cùng:

```bash
repo='/home/share/RT-DETR/RT-DETR/rtdetrv2_pytorch'
config='/duong/dan/config_da_train.yml'
checkpoint='/duong/dan/best.pth'
gt='/duong/dan/instances_val.json'
scripts='/duong/dan/rtdetrv2_next'
out='/duong/dan/experiments/rtdetrv2'
python -m pip install -r "$scripts/requirements.txt"
# Chon contiguous0/category hoac --label-map theo mapping thuc te da train.
python "$scripts/export_predictions.py" --repo "$repo" --config "$config" --checkpoint "$checkpoint" --annotations "$gt" --label-mode contiguous0 --device cuda:0 --output "$out/B2"
python "$scripts/evaluate.py" --annotations "$gt" --predictions "$out/B2/predictions.json" --coverage "$out/B2/coverage.json" --output "$out/B2/evaluation"
```

Ví dụ PowerShell, thay các đường dẫn bằng đường dẫn thật:

```powershell
$repo = 'D:/RT-DETR/rtdetrv2_pytorch'
$config = 'D:/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/visdrone.yml'
$checkpoint = 'D:/RT-DETR/rtdetrv2_pytorch/output/visdrone/best.pth'
$gt = 'D:/data/visdrone_coco/instances_val.json'
$images = 'D:/data/VisDrone2019-DET-val/images'
$scripts = 'D:/DeepLearning/rtdetrv2_next'
$out = 'D:/experiments/rtdetrv2'
python -m pip install -r "$scripts/requirements.txt"
```

### Bước A: xuất predictions đầy đủ từ baseline

**Kiểm tra ánh xạ lớp đã dùng lúc train trước khi chạy.**
Nếu output label 0–9 ứng với 10 category trong GT theo thứ tự `categories`, dùng `contiguous0`.
Nếu output label đã chính là `category_id` của GT, dùng `category`.
Nếu thứ tự khác, tạo JSON ánh xạ rõ ràng, ví dụ `{"0": 1, "1": 2, ..., "9": 10}`,
rồi dùng `--label-map path/to/map.json` thay cho `--label-mode`.
Không cộng 1 tự động khi chưa xác nhận cách train.

Checkpoint `best.pth` trong workspace hiện tại có classifier 11 đầu ra và `last_epoch=119`.
Chưa xác nhận đây có phải checkpoint của bảng kết quả hay cách dùng đầu ra thứ 11.
Giữ đúng config đã train. Nếu đã xác nhận label 0 là đầu ra dự phòng và label 1–10 là class
VisDrone, có thể dùng JSON `{"0": null, "1": 1, ..., "10": 10}`. `null` bỏ đầu ra đó một cách
tường minh. Các ví dụ có `...` chỉ mô tả cấu trúc; JSON thực tế phải liệt kê đủ từng nhãn.
Nếu model có class `others`, chỉ bỏ nó khi protocol đánh giá thực sự là 10 class chuẩn.

Script `prepare_visdrone_kaggle.py` hiện tại tạo category 1–10. Repo chính thức có phiên bản
giữ nguyên category ID khi `remap_mscoco_category=False`; cần kiểm tra bản repo/cách chuyển label
của bạn để bảo đảm nhãn train nằm trong khoảng hợp lệ của classifier `num_classes=10`.
Không đổi mapping của mô hình đã train chỉ bằng cách sửa `num_classes`.

```powershell
python "$scripts/export_predictions.py" --repo $repo --config $config --checkpoint $checkpoint --annotations $gt --label-mode contiguous0 --device cuda:0 --output "$out/B2"
```

Exporter dùng chính validation dataloader/transforms của YAML, nạp EMA nếu checkpoint có EMA,
load weights với `strict=True`, không lọc confidence trước khi lưu predictions.
`coverage.json` gồm cả ảnh không có detection, hash GT, mapping và thông tin repo.
`resolved_config.yml` lưu cấu hình thực tế dùng để xuất.
Các đường dẫn tương đối trong YAML được hiểu từ thư mục repo.

### Bước B: đo small/occlusion và phân tích lỗi

```powershell
python "$scripts/evaluate.py" --annotations $gt --predictions "$out/B2/predictions.json" --coverage "$out/B2/coverage.json" --score 0.3 --iou 0.5 --images $images --output "$out/B2/evaluation"
```

Các đầu ra:

- `metrics.json`: 12 chỉ số COCO cho toàn bộ class và riêng phương tiện, kèm protocol.
- `per_class.csv`: AP/APsmall từng class.
- `recall_groups.csv`: TP/GT/FN/Recall theo small, none, partial, heavy và small+occlusion.
- `failure_images.csv`: xếp ảnh theo số phương tiện nhỏ/che khuất nặng bị bỏ sót.
- `errors.json`: FN và FP, gợi ý miss/localization/class confusion/duplicate/background.
- `failures/*.jpg`: minh họa các ca lỗi; đỏ là GT bị bỏ sót, vàng là FP.

AP và AR dùng toàn bộ predictions, IoU 0,50:0,95 và COCO maxDets 1/10/100.
Recall nhóm là **micro TP/GT tại confidence 0,3 và IoU 0,5**, không phải COCO ARsmall.
Matching theo class trên toàn bộ GT trước khi chia nhóm che khuất; một detection không được
tính đúng cho hai đối tượng. Giới hạn chẩn đoán là 100 detection/ảnh/class theo COCO.
Small trong chẩn đoán dùng diện tích GT <= 1024 pixel trên ảnh gốc. COCO có ranh giới diện tích
bao gồm cả điểm biên; bảng thống kê dataset cũ dùng < 1024 nên số đếm có thể lệch ở điểm biên.
Nhóm không có GT được ghi null/ô trống. Annotation thiếu occlusion vào nhóm unknown.
Nhóm phương tiện dùng tên class, gồm bicycle/car/van/truck/tricycle/awning-tricycle/bus/motor;
pedestrian và people vẫn giữ trong chỉ số toàn bộ 10 class để so sánh bảng cũ.

Phân loại nguyên nhân lỗi dựa trên overlap là chẩn đoán heuristic, không thay thế TIDE hay
đánh giá chính thức. Converter hiện tại bỏ ignored regions; các số liệu là COCO trên VisDrone
đã chuyển đổi, không được gọi là kết quả challenge VisDrone chính thức.

Để đối chiếu công bằng, chạy evaluator này cho predictions của cả ba model với cùng GT,
đúng mapping, cùng confidence diagnostic và đầy đủ validation. Không dùng APsmall toàn bộ
10 class để thay cho APsmall riêng phương tiện.

### Bước C: ablation P2 đầu tiên

```powershell
python "$scripts/prepare_p2.py" --repo $repo --base-config $config --output "$out/configs" --smoke --device cuda:0
```

Script tạo `E0_control.yml` và `E1_p2.yml` từ cấu hình baseline đã resolve.
E1 thêm feature stride 4 (P2), đồng bộ số kênh, stride, số level và sampling points ở decoder.
Attention của encoder giữ trên P5 bằng cách chuyển index 2 thành 3.
Hỗ trợ PResNet R18/R34/R50/R101 và decoder RTDETRTransformerv2 chính thức.
Không sửa số epoch, image size, augmentation, learning rate hay số query của baseline.
`--smoke` chạy inference và train forward/backward trên dữ liệu giả, kiểm tra loss/gradient hữu hạn.
Đây chỉ là kiểm tra kỹ thuật; còn phải thử 1–2 epoch trên dữ liệu thật.

```powershell
Set-Location $repo
python tools/train.py -c "$out/configs/E0_control.yml" --seed 42 --output-dir "$out/E0_control"
python tools/train.py -c "$out/configs/E1_p2.yml" --seed 42 --output-dir "$out/E1_p2"
```

Hai lệnh trên train theo ngân sách trong YAML gốc. Để chạy proxy ít epoch/subset, tạo cùng một
base-config proxy cho E0 và E1 trước khi sinh config; cập nhật cả LR schedule và mốc dừng
augmentation phù hợp với ngân sách proxy. Không so checkpoint full training với P2 train ngắn
rồi kết luận đóng góp của P2.

**Không resume trực tiếp checkpoint ba level vào P2 bốn level.** Khi thêm level, index của
projection thay đổi và shape của deformable attention đổi. Tuning theo tên/shape đơn thuần có
thể nạp sai layer. Dùng cùng backbone pretrained và cùng cách khởi tạo cho E0/E1; nếu cần
warm-start từ B2, phải viết remapping và cho control cùng ngân sách fine-tuning.

Export/evaluate checkpoint của E0 và E1 bằng Bước A/B với config tương ứng. Đo speed trên cùng
GPU, batch 1, input size/precision/backend; có warm-up và CUDA synchronize. Tách latency model
khỏi tốc độ end-to-end đọc/resize/vẽ/ghi video. Không so hai định nghĩa FPS khác nhau.

## 3. Thứ tự hoàn thành nhánh thực nghiệm

| Giai đoạn | Công việc | Điều kiện chuyển bước |
|---|---|---|
| S1 | B2 small/occlusion + ca lỗi, kiểm tra mapping/protocol | Có APsmall-vehicle và Recall-heavy có số lượng GT |
| S2 | E0/E1 P2: smoke, proxy rồi full nếu có tín hiệu | APsmall hoặc Recall-heavy tăng, AP/FPS vẫn hợp lý |
| S3 | E2 Fusion riêng; E3 P2+Fusion nếu từng nhánh có lợi | Ablation tách được đóng góp từng module |
| S4 | E4 occlusion-aware training trên kiến trúc tốt nhất, thêm control training | Recall-heavy tăng và không gây hại nhóm none/small |
| S5 | Benchmark, video chưa dùng train, demo Detection/Compare/Metrics | Có checkpoint/config/bảng kết quả và minh chứng video |

P2 là giả thuyết cần kiểm chứng. Quy tắc từ kế hoạch: ưu tiên FPS còn >=85% baseline khi có thể,
tương đương khoảng 39,18 FPS nếu B2 được đo lại vẫn là 46,0944 FPS.
Ngưỡng tăng AP/Recall và mức giảm AP tổng chấp nhận cần xác định trước thí nghiệm; chạy nhiều
seed cho cấu hình cuối nếu tài nguyên cho phép. Chọn model trên validation, đánh giá cuối trên
test/clip giữ riêng, không dùng clip đã tune để chứng minh khả năng tổng quát hóa.

Fusion, occlusion-aware augmentation, inference video và demo là các giai đoạn sau;
bộ script này triển khai S1 và chuẩn bị S2. Chưa có kết quả thực nghiệm để điền E1–E4.
VisDrone-DET là bài toán ảnh; kết quả DET không đủ chứng minh chất lượng trên video giao thông.
Với video, phân chia theo clip/camera và giữ tập đánh giá riêng; chạy detector từng frame trước,
tracking nằm ngoài phạm vi kế hoạch hiện tại.

## 4. Kiểm tra bộ công cụ

```powershell
python -m unittest discover -s "$scripts" -p 'test_*.py' -v
```

Đã kiểm tra ngày 2026-10-02: 11 unit test và một integration test với repo chính thức,
nạp EMA từ `best.pth` hiện có, xuất predictions trên hai ảnh giả, chạy evaluator và tạo ảnh
ca lỗi. E0 và E1 P2 đều qua inference + train forward/backward tại 640×640 trên CPU.
Đây không phải đánh giá checkpoint trên VisDrone thật và không xác nhận AP/FPS của bảng gốc.
Integration test được bật bằng biến `RTDETR_TEST_REPO`; có thể thêm
`RTDETR_TEST_CHECKPOINT` trỏ tới checkpoint classifier 11 đầu ra tương thích R18, 3 decoder layer.
Khi không đặt biến này, lệnh unittest trên chạy unit test và bỏ qua integration test.

Nguồn API/kiến trúc: [RT-DETR chính thức](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch),
[COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py),
[VisDrone](https://github.com/VisDrone/VisDrone-Dataset).
