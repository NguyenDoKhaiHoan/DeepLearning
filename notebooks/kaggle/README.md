# Kaggle — hoàn thiện đề tài Computer Vision

Các notebook dùng chung package `traffic_mot`, không chứa bản sao thuật toán.
Tất cả hiện chưa có execution/output Kaggle trong session phát triển này.

| Notebook | Trạng thái |
|---|---|
| 01_detection_inference | detection đã chạy CPU local, chờ GPU Kaggle |
| 02_mot_baselines | E0/E1 + TrackEval đã chạy smoke CPU; full Kaggle validation pending |
| 03_occlusion_ablation | OAR E2/E3 + component runs đã có CPU smoke thật; full Kaggle pending |
| 04_final_benchmark | preflight/template; cần validation freeze/final CLI |

Notebook02/03 đã có entrypoints baseline/OAR, notebook04 còn chờ final runner.
Không dùng ByteTrackLite/MOT-lite để thay baseline chuẩn.

## Chạy notebook01

Trong một cell Kaggle, bật Internet rồi lấy source từ GitHub:

```python
!git clone https://github.com/NguyenDoKhaiHoan/DeepLearning.git /kaggle/working/DeepLearning
!git clone https://github.com/lyuwenyu/RT-DETR.git /kaggle/working/DeepLearning/RT-DETR
!git -C /kaggle/working/DeepLearning/RT-DETR checkout 29320b6fd828f8e0987a71426cf2d961b09dfed7
%pip install -r /kaggle/working/DeepLearning/traffic-mot/requirements.txt -r /kaggle/working/DeepLearning/traffic-mot/requirements-dev.txt
```

Nếu source đã clone thì dùng `git pull --ff-only` ở checkout DeepLearning thay
vì clone lại. Sau khi cài dependency, restart kernel trước khi chạy các cell
inference nếu torch/NumPy đã được import. Import notebook01 từ GitHub hoặc upload
file notebook, bật GPU và sửa BUNDLE_DIR/VIDEO sang đường dẫn Kaggle Input thật.
Checkpoint và video không nằm trong Git; gắn chúng riêng bằng Add Input.

1. Clone/mount repo có `traffic-mot/src/traffic_mot`; cài runtime và dev requirements
   với torch/torchvision CUDA tương thích; chạy `pip check`.
2. Bật GPU accelerator. Mount bundle chứa best.pth, demo_config.yml, label_map.json,
   classes.txt, bundle_manifest.json. Không thay best bằng resume giữ manifest cũ.
3. Nếu thiếu RT-DETR source, clone official repo vào `RT-DETR/`, checkout commit
   `29320b6fd828f8e0987a71426cf2d961b09dfed7` và giữ checkout sạch.
4. Sửa REPO_ROOT/BUNDLE_DIR/UPSTREAM_REPO/VIDEO ở cell setup; dùng video thật.
   Preflight xác minh GPU, source, checkpoint/config/mapping và video trước load.
5. Smoke48 frame; kiểm tra JSONL/MP4 đọc lại và manifest. Khi đạt, bỏ max_frames
   để xử lý đầy đủ. Lưu artifact Kaggle để nghiệm thu GPU. Không lấy smoke startup
   time làm steady-state FPS, không tính HOTA trên video không GT.

## Chuẩn bị VisDrone MOT thật và manifest cho notebook02–04

Hai ZIP đã được kiểm tra local ngày24/10/2026:

- `VisDrone2019-MOT-val.zip`: 7 sequence, 2846 ảnh, có GT10 cột với Track ID.
- `VisDrone2019-MOT-test-dev.zip`: 17 sequence, 6635 ảnh, có annotation.

Upload cả hai vào dataset rồi **Create → Add Input**. Dùng `val` để tune/
validation, giữ `test-dev` cho final evaluation sau khi cấu hình đã freeze.
Không cần tạo MP4 từ các JPG, không gán FPS giả. Raw GT target ID0 là hợp lệ;
public tracker output ID dương là contract riêng, không được làm mất GT0.

Trong Kaggle, lấy code mới rồi tạo manifest theo mount path thực tế:

```python
!git -C /kaggle/working/DeepLearning pull --ff-only
!python /kaggle/working/DeepLearning/traffic-mot/scripts/prepare_mot_dataset.py --root /kaggle/input/visdrone-mot --output /kaggle/working/mot_split_manifest.json
```

Nếu path mount khác, thay `--root` bằng đường dẫn Copy path trong Add Input.
Script tìm đúng hai release ZIP hoặc thư mục đã giải nén; không tự giải nén/
chuyển video, đọc index ảnh/annotation và hash nguồn. Nếu cùng lúc có cả ZIP
và bản giải nén, chọn tường minh `--validation-source <val_zip_or_directory>
--test-source <test_dev_zip_or_directory>` thay `--root`.

Output là `mot_split_manifest.json` và `mot_split_manifest.preflight.json` cạnh
nhau. Lệnh mặc định chỉ đọc GT/ảnh đầu mỗi sequence của **validation**, không
parse annotation test để tune. Không ghi đè manifest/report đã có; dùng path
mới nếu đổi dataset. Preflight test chỉ dùng ở final gate đã freeze.

Notebook02–04 mới có cell `prepare_mot_manifest` dùng chung source, tạo manifest
tại `OUTPUT_ROOT / "mot_split_manifest.json"`. Chỉnh `MOT_DATA_ROOT` đúng mount;
hoặc đặt `MOT_MANIFEST` trỏ vào file đã tạo bằng lệnh trên. File local trỏ `D:/`
không dùng nguyên trên Kaggle: cần tạo lại theo mount `/kaggle/input/`.

Manifest hỗ trợ `source_kind=images_zip` với archive/member paths hoặc
`images_dir` với frames_dir/gt. Giữ thứ tự ảnh1-based và raw IDs; checker không
chấp nhận trùng sequence/source UAV giữa validation và test. Fingerprint index
ZIP dựa filename/size/CRC; hash archive giữ integrity. Đối với images_dir, checker
hash ảnh thật; chưa làm perceptual duplicate detection hoặc kiểm tra tất cả ảnh
decode được. Chỉ khi có evaluator chuẩn mới báo HOTA/IDF1/MOTA/IDSW.

## Chạy notebook02 và sửa lỗi Stage not implemented

Code đã bổ sung `traffic-mot/scripts/run_mot_baselines.py`. Lấy version mới và
cài cả dependency evaluator trong **cùng kernel Python** trước khi chạy:

```python
!git -C /kaggle/working/DeepLearning pull --ff-only
%pip install ultralytics==8.4.171 lap==0.5.12 -r /kaggle/working/DeepLearning/traffic-mot/requirements-eval.txt
```

Runtime detector yêu cầu requirements.txt như notebook01. Nếu pip thay các module
đã import, restart kernel rồi chạy lại setup. Import notebook02 bản mới từ GitHub;
chỉnh BUNDLE_DIR/MOT_DATA_ROOT hoặc MOT_MANIFEST đúng Input/config đã có.
Mặc định `FULL_VALIDATION=False`: một sequence48 frame, output ghi smoke_test=true.
Kiểm tra predictions/metrics/preview trước, sau đó đặt True để chạy đủ7 sequence.
Output run có UUID mới mỗi lần, không ghi đè kết quả hoặc cache cũ.

CLI nhận `--detector-config`, `--manifest`, `--split validation`, `--output-root`
đúng các arguments trong cell báo lỗi. Chỉ dùng native manifest từ script chuẩn bị,
E0/E1 cùng cache detection. Optional `--save-video --visualization-fps 10` tạo
MP4 playback, không dùng10fps làm tốc độ camera hoặc benchmark pipeline.
Không cần mở notebook04 để chạy baseline; final benchmark/test cần bước OAR/
ablation và cấu hình đã freeze, chưa triển khai trong lần sửa lỗi này.

## Chạy notebook03 và sửa lỗi run_ablation.py

Lấy code mới bằng `git pull --ff-only`; cùng requirements runtime/eval với
notebook02, không thêm ReID model. Trong notebook03 bản mới, chỉnh đường dẫn
config/Input như notebook02. Nếu notebook02 đã hoàn thành, đặt:

```python
BASELINE_REFERENCE = Path("/kaggle/working/traffic_mot_results/baselines_<run_uuid>")
```

Runner kế thừa exact cache, sequence và frame range từ baseline, không infer lại;
kiểm sha/config/pixel hashes rồi tính E0/E1 mới và so metrics cũ trước so OAR.
Baseline smoke chỉ dẫn tới ablation smoke, không thể freeze để đánh giá test.
Nếu baseline dùng descriptor `adapter_status` cũ, khác biệt status được bỏ qua
nhưng mọi scoring parameter vẫn phải khớp. Không thay class scope để tạo improvement.

Không có reference: giữ `BASELINE_REFERENCE=None` và `FULL_VALIDATION=False` cho
smoke48 frame; khi đạt, đặt True cho đủ validation. Script sẽ infer/cache một lần
cho tất cả8 experiments. Các cell cũ gọi trực tiếp `run_ablation.py` với các
arguments config/detector-config/manifest/split/output-root hiện cũng đã hoạt động.
Dùng thư mục output mới cho mỗi lần chạy, không overwrite run cũ.

Output: E0–E3, BT/BS retention-only/reassociation-only metrics+diagnostics+
prediction files; ablation_summary.json, paired_differences.json, selection.json.
Đường dẫn cấu hình được chọn nằm trong selection.json: smoke ghi candidate_config,
full validation ghi selected_config. Nếu full validation chọn baseline, đó là
kết quả hợp lệ; không đổi threshold trên test để ép OAR thắng.
Chưa chạy final_test/benchmark notebook04 trong thay đổi này.

Tạo lại notebook: `python traffic-mot/scripts/build_kaggle_notebooks.py`.
