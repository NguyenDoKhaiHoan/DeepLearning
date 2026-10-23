# Đặc tả tích hợp Đồ án 3 — phiên bản 1.0.0

Đây là contract thiết kế Phase 0. JSON Schema và example có thể validation ngay;
các interface runtime sẽ triển khai trong phase tương ứng. Không có Kafka/Flink,
PostgreSQL writer hoặc frontend mới trong giai đoạn này.

## 1. Interface và data ownership

| Interface dự kiến | Input → output | Quyền quản lý |
|---|---|---|
| `Detector.detect(frame)` | BGR uint8 H×W×3 → DetectionBatch | chỉ model/pre/postprocessing |
| `Tracker.update(detections, frame, timestamp)` | original xyxy batch, BGR, ms → list TrackObservation | lifecycle + public identity |
| `EventEngine.process(tracks, timestamp)` | confirmed observed tracks → list TrafficEvent | rule/episode/dedup state |
| `VideoPipeline.process(source)` | file path / webcam int / RTSP → RunSummary và streaming sinks | session, frame clock, source, output |
| `EventExporter.export(events)` | iterable TrafficEvent → số record ghi | serialize/flush/errors, không sửa event IDs |

Pipeline cấp `FrameContext(camera_id,session_id,frame_id,timestamp_ms,time_basis,
video_source,width,height)` cho tracker/event engine khi bắt đầu từng frame.
`timestamp` là integer ms trong session; frame context là argument keyword tùy
chọn hoặc context bind có type trong implementation, không global mutable state.
`DetectionBatch` dùng float32 boxes/scores, int class_ids và stable detection index;
empty batch hợp lệ. EventEngine không cần detector/tracker class cụ thể.
Các API đều có reset/close/context-manager thích hợp khi implementation bắt đầu.

## 2. Identity, box và timestamp

- Khóa object: `(camera_id,session_id,track_id)`; track_id dương, không tái dùng
  trong session. Session UUID mới khi restart hoặc reconnect làm gián đoạn clock.
  Camera ID được cấu hình, không dùng mỗi backend global counter làm ID toàn hệ thống.
- Frame ID bắt đầu 1 theo frame nguồn decode, không theo số frame thành công có
  detection. File timestamp theo media PTS tính từ đầu clip; CFR fallback khi có
  FPS được xác nhận: `round((frame_id-1)*1000/fps)`. MP4 VFR không dùng công thức CFR.
- Webcam/RTSP dùng monotonic elapsed từ session start nếu không có clock nguồn
  tin cậy. Clock này không phải UTC và không đủ để đồng bộ camera. Lưu time_basis;
  future source UTC/clock calibration sẽ là field bổ sung có version.
- Timestamp ms có thể trùng do rounding, frame_id luôn tăng; clock đi lùi tạo
  lỗi hoặc session mới có reason, không tự ép thời gian để che discontinuity.
- Box là `[x1,y1,x2,y2]` float pixel frame gốc, x2>x1, y2>y1, clipped W/H. Không
  làm tròn tới lúc visualization/CSV. Model IDs 3–10 được giữ nguyên.
- Track state `lost` có thể kèm predicted box và confidence observation cuối,
  nhưng `observed=false`. Không dùng lost record để count hoặc làm MOT detection.

Frame skip/drop là metadata bắt buộc của run nếu xảy ra. Recovered track giữ ID
đã công bố; không phát event đổi ID sau khi consumer đã dùng khóa cũ.

## 3. Track và trajectory records

Schema: [`track.schema.json`](../schemas/track.schema.json), ví dụ minh họa
[`track.example.json`](../schemas/track.example.json). Trường bắt buộc bao gồm
camera/session/frame/track, class_id/class_name, bbox/confidence, track_state,
observed, timestamp_ms/time_basis và video_source. Class_name phải khớp class_id
theo mapping; source là asset key hoặc URL đã bỏ credentials, không chứa password/token.

Trajectory sidecar tương lai: camera/session/track ID, list điểm
`{frame_id,timestamp_ms,center:[x,y],smoothed_center:[x,y],observed:true}`, tổng
displacement pixel, hướng vector/angle image-coordinate (x phải, y xuống) và
window/gap config. Không phát tốc độ km/h khi chưa có homography/calibration.
History giới hạn, exporter stream points nếu cần lịch sử dài hơn deque.

## 4. Event semantics và chống đếm trùng

Schema: [`event.schema.json`](../schemas/event.schema.json), ví dụ minh họa
[`event.example.json`](../schemas/event.example.json). Required fields có version,
event_id UUID, camera/session/track, rule_id, event_type, vehicle_type, frame,
timestamp, source, confidence, interval và attributes. Ví dụ không phải measurement.

| Event | Điều kiện và episode |
|---|---|
| line_crossing | observed bottom-center chuyển hai phía line hữu hạn, qua hysteresis; lưu hướng |
| vehicle_count | dẫn xuất từ crossing đã dedup; aggregate theo camera/session/rule/direction/class |
| region_entry / region_exit | chuyển state ROI sau dwell; không phát entry chỉ vì track mới sinh trong ROI |
| wrong_way | displacement tối thiểu và góc sai hướng giữ đủ duration; một event khi episode kết thúc |
| stationary_vehicle | điểm observed nằm trong bán kính pixel đủ duration; reset khi gap dài, một event khi kết thúc |

Line counting profile mặc định tối đa một lần/track/rule/direction. ROI có thể
có nhiều episode ra/vào hợp lệ, cần episode_start_frame riêng. Các event duration
phát khi episode kết thúc hoặc flush cuối session; có reason `ended/session_end`
trong attributes. Nếu muốn phát provisional updates sau này phải có revision
semantics/schema riêng; phiên bản này chỉ event hoàn chỉnh.

`timestamp_ms = end_timestamp_ms`, `duration_ms = end-start`, `frame_id` là frame
cuối episode, `start_frame_id` là frame bắt đầu. Point event start=end/duration=0.
Confidence là min detection confidence trên các observation dùng xác nhận rule,
không tuyên bố là xác suất đúng đã calibrated. Lost frames không được thêm vào
để chứng minh stationary. Thời gian đoạn gap dài làm ngắt candidate episode.

Dedup key logical: camera/session/track/rule/type/direction/episode_start_frame.
Phase 4 dùng UUIDv5 với namespace cố định được công bố trong code từ key chuẩn
JSON, thay vì tạo UUID ngẫu nhiên khi retry ghi cùng event. Dedup in-memory
trong session + unique event_id ở consumer. Không đếm cả crossing và vehicle_count
như hai xe. Counting output dẫn xuất có source_event_id liên kết crossing.

JSON Schema kiểm tra shape/type/range; semantic validator Phase 4 phải kiểm tra
class-name agreement, box extent, timestamp/interval equality và transition
legality. Schema Phase 0 chưa kiểm tra được các bất đẳng thức giữa nhiều field.

## 5. Serialization, storage và future transport

JSONL UTF-8 một object/dòng là canonical stream; JSON array xuất bằng bước finalize,
CSV flattened một record/event, bbox/attributes encode JSON trong cell. Exporter
tạo run directory riêng, không overwrite source, flush định kỳ và đóng file
trong finally; báo lỗi ghi đĩa không lặng lẽ mất event. Retry giữ nguyên event_id.

Run manifest tối thiểu: run_id, started_at UTC, camera/session/source đã sanitize,
clock/fps/frame range, seed, detector/checkpoint/hash/weights, raw+resolved config,
backend/GMC/ReID/OAR flags, schema version, Git commit+dirty diff và environment.
Output đề xuất: `runs/<run_id>/{tracks.jsonl,events.jsonl,trajectories.jsonl,
mot/<sequence>.txt,annotated.mp4,manifest.json,timings.csv}`.

Kafka tương lai: topic riêng track/event, partition key `camera_id:session_id`,
event time lấy source timestamp + declared time basis; consumer dedup event_id.
Ordering theo camera/session/frame; không đảm bảo ordering giữa camera. Producer
retry có thể at-least-once; không tuyên bố exactly-once trước khi có transaction
protocol. Thay file sink bằng transport sink qua EventExporter, không sửa rule engine.

PostgreSQL tương lai: events.event_id primary key; index
`(camera_id,session_id,timestamp_ms)` và `(event_type,vehicle_type)`; object FK
camera/session/track; JSONB attributes; source asset table liên kết clip.
Tìm kiếm và replay dùng source asset key + timestamp range. Với live nguồn không
được lưu, đánh dấu replay unavailable; timestamp không tự tạo file video lưu trữ.

## 6. Versioning và failure contract

Schema 1.0.0 từ chối field top-level chưa biết, attributes dành cho metadata của
rule. Breaking change đổi major; consumer phải chọn schema đúng version, không
đoán tên field. Mọi semantic extension cần schema/test/example/changelog đồng bộ.

Input invalid/checkpoint mismatch → fail-fast có thông báo; capture EOF → finalize
thành công; stream mất kết nối → giới hạn retry và ghi reason/session; writer
không mở được → lỗi rõ, không trả thông báo video đã lưu. Interrupted run đánh dấu
incomplete và file JSONL vẫn đọc được phần đã flush; MP4 có thể chưa finalized.
Không nối append vào session cũ khi không có checkpoint state đầy đủ tracker/event.
