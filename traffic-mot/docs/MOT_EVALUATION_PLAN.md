# Kế hoạch đánh giá MOT và cải tiến dưới che khuất

Protocol ban đầu Phase0; cập nhật24/10/2026: baseline/evaluation runner đã triển
khai, E0/E1 đã có real48-frame CPU smoke (MOT_BASELINE_REPORT). Full validation
Kaggle và final test còn pending. OAR/E2/E3/components và occlusion/size/episode
diagnostics đã được triển khai24/10, kiểm chứng48-frame local cache/GT thật ở
OAR_ABLATION_REPORT. Full recovery episodes/benchmark vẫn cần dữ liệu/gate riêng.

## 1. Dữ liệu hợp lệ và split

Dùng VisDrone **MOT** có persistent target IDs hoặc dataset traffic MOT tương đương.
Không chuyển mỗi annotation VisDrone2019-DET thành một ID rồi đánh giá MOT.
Kiểm tra chính xác release: ID/category/ignore/truncation/occlusion, số frame,
resolution/FPS, license và GT coverage. VisDrone MOT format 10 cột:

`frame, target_id, left, top, width, height, score, category, truncation, occlusion`.

Giữ bản raw chỉ đọc và sidecar JSONL có đầy đủ thuộc tính. Ở GT, score 0 nghĩa
ignore. Class 0/11 và vùng ignore không phải background vehicle. Tracking scope
không mặc định bằng 8 class detector: toolkit VisDrone2018 MOT nêu 5 class car,
bus, truck, pedestrian, van; profile phương tiện khởi đầu chỉ đánh giá 4/5/6/9.
Xác minh release thực tế trước khi khóa scope; các class không có GT không được
báo recall bằng 0 hoặc suy ra thiếu sót của tracker.

Split train/tuning/heldout theo sequence/camera, không random frame. Giữ nguyên
official split khi dùng; các đoạn cùng video nguồn không xuất hiện ở hai split.
Tuning recovery trên train/validation có IDs, sau đó đóng băng toàn bộ threshold;
heldout dùng một lần cho báo cáo. Manifest gồm sequence IDs, SHA256 annotation,
nguồn, camera, fps/timebase, range, class scope, split và ignore policy.
Nếu chưa có GT hoặc occlusion không có nhãn, metric tương ứng là N/A kèm lý do.

## 2. Kết nối TrackEval và chính sách ignore

Khóa [TrackEval](https://github.com/JonathonLuiten/TrackEval/tree/12c8791b303e0a0b50f753af204249e622d0281a)
tại commit `12c8791b303e0a0b50f753af204249e622d0281a`. Dùng HOTA, Identity và CLEAR
implementation của toolkit; không tự viết công thức HOTA/IDF1. Bộ đọc
`MotChallenge2DBox` mặc định chỉ chấp nhận pedestrian; không truyền VisDrone
category 4 vào reader đó và tin rằng đó là car evaluation.

Implementation hiện có `VisDroneMOTAdapter` tạo metric data dùng IoU helper của
`BaseDataset`, rồi gọi trực tiếp official metric eval_sequence/combine_sequences
và combine_classes_det_averaged (không dùng pedestrian dataset reader).
Adapter cung cấp valid class,
timeline đầy đủ, GT/tracker IDs remapped nhất quán theo sequence, similarity IoU,
num_gt_ids/dets và num_tracker_ids/dets. Chỉ remap ID phục vụ evaluator;
không chỉnh prediction gốc hoặc split identity theo occlusion. Export MOTChallenge
10 cột `frame,id,x,y,w,h,score,-1,-1,-1` là artifact phụ, không thay sidecar class/ignore.

Baseline nghiên cứu dùng ignore policy có version:

1. Xác thực box/ID/frame/class, unique `(sequence,frame,target_id)`; không tự sửa
   duplicate GT, không bỏ frame rỗng. Nạp đúng width/height và coordinate origin
   của release, chuyển một lần sang original-pixel xyxy.
2. Giữ valid evaluation GT, non-evaluation class và ignored regions riêng biệt.
   Per class, tính assignment valid GT–prediction IoU ≥ 0.5 để bảo vệ prediction
   khớp valid object. Không xóa dự đoán trước khi xét valid GT.
3. Chỉ prediction chưa được bảo vệ mà overlap vùng explicit ignore có IoA ≥ 0.5
   mới bị loại khỏi scoring. Không mặc định xóa mọi prediction gần class không
   được đánh giá. GT marked ignored không tính FN/identity denominator.
4. Chạy metric trên timeline đầy đủ sau preprocessing; lưu số bị ignore và reason.

Đây là protocol nghiên cứu cần được kiểm thử và khóa, chưa khẳng định tương
đương challenge VisDrone chính thức. Đối chiếu với toolkit VisDrone khi cần
báo cáo challenge; ghi rõ khác biệt ignore và phạm vi class. Test chồng valid
GT/ignore region để tránh che mất FP/FN hoặc xóa đúng detection.

## 3. Metrics chuẩn và test evaluator

| Nhóm | Chỉ số / cách báo cáo |
|---|---|
| Association + detection | HOTA, DetA, AssA do TrackEval tính qua dải alpha chuẩn |
| Identity | IDF1, IDP, IDR bằng global identity assignment của toolkit |
| CLEAR | MOTA, IDSW, FP, FN, MT, ML và fragmentation nếu toolkit có |
| Runtime | detector/tracking/recovery/pipeline latency và FPS, protocol mục 7 |

MT/ML dùng đúng ngưỡng bản toolkit đã pin (MT >80%, ML <20% đời sống GT được theo dõi).
MOTA có thể âm. Ghi số GT tracks/observations/sequences và scale metric 0–1 hay
0–100. Multi-class báo từng class, macro nếu có định nghĩa và combined counts
do toolkit tổng hợp; không trung bình MOTA/HOTA các sequence tùy ý.

Acceptance fixtures: perfect sequence, missing observation, FP, đổi ID giữa đoạn,
fragment + unmatched gap, class sai, all-empty, GT-empty nhưng có prediction,
duplicate ID phải lỗi, ignore overlap, native resolution khác và frame origin.
So trực tiếp output toolkit trên fixture, không dựa vào `MOT-lite` hiện có.
Known upstream limitation: CLEAR bản pin có thể undercount Frag khi frame không
có prediction nào. Summary hiện không báo Frag, raw vẫn giữ toolkit output với
disclosure. np.float/np.int cũ được bridge local trong module toolkit, không sửa
metric formula hoặc global NumPy. Protocol hiện không tuyên bố challenge-equivalent.

## 4. Đánh giá theo occlusion và size

GT occlusion enum: none=0, partial=1, heavy=2 khi release có thuộc tính này.
Unknown loại khỏi phân tầng, vẫn giữ trong metric tổng. Truncation được báo
riêng, không gộp vào heavy occlusion. Model overlap evidence không thay GT.

Chạy association **với đầy đủ valid GT và predictions trong frame trước**, rồi
gán kết quả matching cho nhóm GT. Báo GT count, TP/FN, recall và identity continuity
theo none/partial/heavy; lưu prediction unmatched chung. Không lọc chỉ heavy GT
rồi rematch, vì detection của xe rõ có thể bị ghép sai vào xe bị che.
Không gọi diagnostic recall là HOTA-heavy. Muốn metric HOTA cho subset phải có
protocol xử lý ngoài-subset/identity riêng được kiểm chứng; chưa hỗ trợ ở Phase 0.

Size dựa diện tích GT pixel ở **frame gốc**: small < 32², medium [32²,96²), large
≥ 96². Đây là ngưỡng nghiên cứu theo COCO, không tự nhận là standard MOT size.
Báo resolution distribution và count từng nhóm; thêm bảng small×heavy nếu đủ mẫu.
Toàn sequence giữ nguyên để tránh mất continuity khi object đổi nhóm size.

## 5. Recovery success rate có mẫu số cố định

Tạo tập episode từ GT/annotation bổ sung **trước và độc lập mọi tracker**. Episode
gồm cùng GT ID có ít nhất 3 observation liên tiếp trước, một đoạn partial/heavy
được gắn nhãn, rồi ít nhất 3 observation visible liên tiếp sau. Exit/re-entry,
truncation biên và detector misses không có evidence GT được phân tích riêng.
Nếu fully occluded frames không có box/identity nối qua gap, cần annotation
episode thủ công độc lập, không đoán từ prediction; thiếu thì không báo fully
occluded recovery rate. Lưu episode ID, GT ID, pre/start/end/post frames và label source.

Với mỗi tracker, dùng diagnostic Hungarian per-frame class-aware IoU≥0.5 trên GT
đầy đủ. Pre-ID phải khớp cùng public prediction ID ở 3 pre-frames. Thành công
khi trong tối đa W=10 frame sau kết thúc episode, cùng ID đó được khớp lại và
duy trì K=3 frame observed liên tiếp, không đồng thời nhận một GT khác. Không có
pre-ID hoặc không khớp lại được tính failure trong mẫu số cố định, không loại đi.

`RecoverySuccess = successful episodes / all eligible GT episodes`.
Implementation đặt tên protocol gt_labeled_occlusion_continuity_v1: includes
uninterrupted same-ID tracking, không chứng minh causal OAR reassociation. Gap
không annotation không được tự gọi fully-occluded recovery. Không eligible GT
episode thì ratio null/N/A, số thao tác reassociation không thay mẫu số này.
Báo thêm pre-episode tracked coverage; nếu báo recovery có điều kiện trên số có
pre-ID, phải ghi mẫu số riêng và không dùng thay số chính. Phân tích wrong-ID
recovery, số IDSW quanh episode, recovery delay frame/ms, gap length bins và
false recovery trong vùng không occlusion. Detector miss episodes là bảng riêng.

## 6. Ablation và reproducibility

| Run | Tracker | OAR | ReID | Vai trò |
|---|---|---|---|---|
| E0 | ByteTrack backend gốc | off | off | baseline |
| E1 | BoT-SORT backend gốc + GMC | off | off | so sánh tracker |
| E2 | ByteTrack + extension | on | off | hiệu quả OAR so E0 |
| E3 | BoT-SORT + extension + GMC | on | off | hiệu quả OAR so E1 nếu gate đạt |

Tên E0/E1 ở đây là **tracking experiment**, khác E0/E1 detector training cũ.
Manifest dùng prefix `mot_E0`… để không trộn báo cáo. Detector weights/config,
low cutoff 0.10, image size, preprocessing, split, frame order, FPS/timebase và
cache detection giống nhau. Cache có schema/hash, model hash và precision; cache
khác không được so như cùng detector. GMC dùng frame gốc cả khi dùng cache.

Component ablation: retention-only, reassociation-only, kết hợp. Cấu hình adaptive
cost/appearance chỉ thêm thành run riêng, không đổi nhiều yếu tố mà gán toàn bộ
lợi ích cho một module. Nếu tune baseline, cùng budget search và validation;
giữ thêm stock-default run để rõ ảnh hưởng tuning. Seed42, deterministic khi khả
thi; ghi CUDA nondeterministic ops. Bootstrap 1000 lần **theo sequence**, paired
difference E2−E0/E3−E1 và CI95%; sequence quá ít thì công bố độ bất định.

Artifact mỗi run: raw/resolved YAML, command, source commit + dirty diff, detector
hash, backend version/extension diff, cache/split/GT hashes, environment, predictions
JSONL+MOT, raw TrackEval JSON, summary CSV, episode/error tables và failure clips.
Không điền bảng trước khi run; đánh giá kết quả xấu cũng cần giữ lại.

## 7. Runtime benchmark

Hai chế độ tách biệt: cache replay đo tracker/recovery; chạy detector thật đo
pipeline end-to-end. Không lấy cached FPS rồi gọi là whole pipeline throughput.
Batch1/stride1, warmup30 frame (không tính), sau đó đo toàn bộ clip hoặc số frame
công bố trước. Khởi tạo model và encode finalization ghi riêng. GPU synchronize
ở ranh giới đo hoặc CUDA events với sync; CPU dùng perf_counter.

Tách decode/preprocess/model-forward/postprocess/tracking/recovery/events/render/
encode và total. Báo mean/p50/p95/p99 ms, tổng wall elapsed, measured frame count,
processed FPS=N/time và dropped frame count. Model-forward FPS, full-detector FPS
và pipeline FPS là ba số khác nhau. Instrumentation có thể giảm concurrency;
ghi overhead và dùng cùng cách đo cho các run. Video writing on/off là hai profile.
Lưu GPU/CPU, torch/CUDA/driver, OS, threads, resolution, precision, backend, warmup.

## 8. Nguồn và giới hạn

- [VisDrone dataset và MOT task](https://github.com/VisDrone/VisDrone-Dataset).
- [Toolkit VisDrone MOT: format, class scope và ignore](https://github.com/VisDrone/VisDrone2018-MOT-toolkit).
- [TrackEval reader MOTChallenge](https://github.com/JonathonLuiten/TrackEval/blob/12c8791b303e0a0b50f753af204249e622d0281a/trackeval/datasets/mot_challenge_2d_box.py).

Nội dung nguồn được kiểm tra qua GitHub raw/API ngày 23/10/2026. Chưa xác thực
protocol trên dataset thật, chưa có số liệu improvement hay tốc độ mới.
