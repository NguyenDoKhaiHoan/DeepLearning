"""Build the standalone Kaggle notebook from the checked-in experiment tools."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip() + '\n'))


md('''
# RT-DETRv2-R18 + P2 trên VisDrone — Kaggle (bước B–C)

Notebook chuyển từ bản chỉ `--test-only` sang train **E1 P2**. E0 là control 3 level;
E1 thêm P2 stride 4, encoder attention vẫn trên P5. Các script chuyển đổi dữ liệu,
tạo P2, export và đánh giá được nhúng sẵn; không cần gắn dataset chứa script `data111`.

**Trước khi Run All:** bật **Internet**, chọn **GPU** trong Settings và Add Input dataset
VisDrone gốc có cả train/val (`images/` và `annotations/*.txt`). Đường dẫn dataset cũ đã
được điền bên dưới; nếu Kaggle gắn đường dẫn khác, sửa `DATA_ROOT` hoặc để `None` để dò.
Notebook dùng một GPU, kể cả khi phiên có hai T4.

**Luồng chạy:** cấu hình → COCO/mapping → E0/E1 → smoke + batch thật → train → export
toàn bộ validation → APsmall/Recall-heavy + ảnh lỗi → benchmark → đóng gói output.
Mặc định `RUN_MODE="full"`, 120 epoch, batch 2, AMP, train E1. Để thử **2 epoch trước**,
đặt `RUN_MODE="proxy"`; khi train chính thức đổi về `full` và chạy lại từ đầu.
Để ablation đúng kế hoạch, chạy lại với `EXPERIMENTS=["E0_control"]` và cùng các thông số.
Không dùng P2 2 epoch để kết luận tốt hơn baseline 120 epoch.
''')
code('''
from pathlib import Path
import os, sys, json, subprocess, shutil, hashlib, math, copy

DATA_ROOT = Path("/kaggle/input/datasets/nguyendokhaihoan/aaaaaaa")  # hoặc None
RUN_MODE = "full"                         # "proxy" (2 epoch) / "full" (120 epoch)
EXPERIMENTS = ["E1_p2"]                   # hoặc ["E0_control", "E1_p2"]
BATCH_SIZE = 2                            # cùng batch cho E0/E1; giảm về 1 nếu OOM
VAL_BATCH_SIZE = 2
NUM_WORKERS = 2
SEED = 42
USE_AMP = True
RUN_SMOKE = True
RUN_REAL_PREFLIGHT = True
RUN_BENCHMARK = True

# Chỉ resume last.pth của ĐÚNG experiment và mode đã chạy bằng notebook này.
# Gắn output phiên trước vào Input; thư mục checkpoint phải có run_config.yml,
# run_manifest.json, log.txt và best.pth. Để None khi train mới.
RESUME = {"E0_control": None, "E1_p2": None}
# Ví dụ: RESUME["E1_p2"] = "/kaggle/input/my-p2-output/full/E1_p2/last.pth"

# Bước B cho baseline cũ là tùy chọn. Không nạp B2 vào E1 để train.
RUN_B2_EVALUATION = False
B2_CHECKPOINT = Path("/kaggle/input/datasets/nguyendokhaihoan/133333333/best.pth")
# Nếu có YAML thực tế lúc train baseline, gắn vào Input và điền tại đây.
# Nếu None: dùng kiến trúc R18 / 3 decoder layer / 11 đầu ra của notebook cũ.
B2_CONFIG = None

REPO_COMMIT = "29320b6fd828f8e0987a71426cf2d961b09dfed7"
WORK = Path("/kaggle/working")
REPO = WORK / "RT-DETR" / "rtdetrv2_pytorch"
SCRIPTS = WORK / "rtdetrv2_next"
COCO_ROOT = WORK / "visdrone_coco"
OUT = WORK / "rtdetrv2_experiments" / RUN_MODE
CONFIGS = OUT / "configs"
DEVICE = "cuda:0"
assert RUN_MODE in {"proxy", "full"}
assert EXPERIMENTS and len(EXPERIMENTS) == len(set(EXPERIMENTS))
assert set(EXPERIMENTS) <= {"E0_control", "E1_p2"}
assert BATCH_SIZE >= 1 and VAL_BATCH_SIZE >= 1 and NUM_WORKERS >= 0
for path in (SCRIPTS, OUT, CONFIGS):
    path.mkdir(parents=True, exist_ok=True)

def run(args, cwd=None, log=None):
    args = [str(x) for x in args]
    print(" ".join(args), flush=True)
    stream = Path(log).open("a", encoding="utf-8") if log else None
    try:
        with subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                if stream:
                    stream.write(line)
            status = process.wait()
        if status:
            raise subprocess.CalledProcessError(status, args)
    finally:
        if stream:
            stream.close()
''')
md('''
## 1. Repo và môi trường

Giữ PyTorch/CUDA sẵn có của Kaggle; chỉ cài thư viện cần cho train/evaluate.
Repo được khóa commit để E0/E1 dùng cùng mã nguồn. Sửa nhỏ AMP để `cuda:0` dùng đúng
`device.type`, và giữ best AP của các epoch trước khi resume. Không thay evaluator trong vòng train.
''')
code('''
run([sys.executable, "-m", "pip", "install", "-q", "faster-coco-eval>=1.6.6,<2",
     "pycocotools>=2.0.7", "PyYAML>=6", "scipy", "tensorboard", "packaging", "Pillow"])
import torch, torchvision, yaml
if not torch.cuda.is_available():
    raise RuntimeError("Chọn GPU trong Kaggle Settings rồi khởi động lại phiên.")
print("torch:", torch.__version__, "torchvision:", torchvision.__version__)
print("GPU:", torch.cuda.get_device_name(0), "VRAM GB:",
      round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
if not REPO.exists():
    run(["git", "clone", "https://github.com/lyuwenyu/RT-DETR.git", REPO.parent])
actual_commit = subprocess.check_output(
    ["git", "-C", str(REPO.parent), "rev-parse", "HEAD"], text=True).strip()
if actual_commit != REPO_COMMIT:
    run(["git", "-C", REPO.parent, "checkout", REPO_COMMIT])

engine = REPO / "src/solver/det_engine.py"
text = engine.read_text(encoding="utf-8")
text = text.replace("device_type=str(device)", "device_type=device.type")
engine.write_text(text, encoding="utf-8")

# Official solver initializes best_stat from scratch, even on resume.
solver = REPO / "src/solver/det_solver.py"
text = solver.read_text(encoding="utf-8")
needle = "        best_stat = {'epoch': -1, }"
replacement = """        best_stat = {'epoch': -1, }
        # Kaggle resume: retain the best AP from previous completed validations.
        if self.cfg.resume and self.output_dir:
            history = self.output_dir / 'log.txt'
            if history.exists():
                for line in history.read_text().splitlines():
                    row = json.loads(line)
                    values = row.get('test_coco_eval_bbox')
                    if (values and row['epoch'] <= self.last_epoch
                            and values[0] > best_stat.get('coco_eval_bbox', -1)):
                        best_stat = {'epoch': row['epoch'], 'coco_eval_bbox': values[0]}
"""
if "# Kaggle resume:" not in text:
    if text.count(needle) != 1:
        raise RuntimeError("Solver khác phiên bản đã kiểm tra; kiểm tra REPO_COMMIT.")
    solver.write_text(text.replace(needle, replacement), encoding="utf-8")
''')
md('''
## 2. Script nhúng sẵn

Các cell `%%writefile` chỉ tạo công cụ trong `/kaggle/working`; chạy nguyên cell.
''')
for name in ['prepare_visdrone_kaggle.py', 'export_predictions.py', 'prepare_p2.py', 'evaluate.py']:
    source = ROOT / name if name.startswith('prepare_visdrone') else ROOT / 'rtdetrv2_next' / name
    code(f'%%writefile /kaggle/working/rtdetrv2_next/{name}\n' + source.read_text(encoding='utf-8'))

md('''
## 3. VisDrone → COCO và kiểm tra mapping

Giữ category **1–10**, `remap_mscoco_category=False`, `num_classes=11` như notebook cũ.
Classifier có slot 0 dự phòng; exporter bỏ slot này bằng JSON mapping tường minh.
Đây là classifier 11 đầu ra cho **10 class VisDrone**, không thêm class `others`.
GT train/val giữ nguyên occlusion/truncation để đo bước B. Converter bỏ ignored regions;
đánh giá là COCO trên VisDrone đã chuyển đổi, không phải challenge VisDrone chính thức.
''')
code('''
if DATA_ROOT is None or not DATA_ROOT.exists():
    DATA_ROOT = Path("/kaggle/input")
run([sys.executable, SCRIPTS / "prepare_visdrone_kaggle.py",
     "--input", DATA_ROOT, "--output", COCO_ROOT])
dataset = json.loads((COCO_ROOT / "rtdetr_dataset.json").read_text())
stats = json.loads((COCO_ROOT / "dataset_stats.json").read_text())
TRAIN_IMAGES = Path(dataset["train_images"])
VAL_IMAGES = Path(dataset["val_images"])
TRAIN_JSON = COCO_ROOT / "instances_train.json"
VAL_JSON = COCO_ROOT / "instances_val.json"
NUM_CLASSES = 11
LABEL_MAP = COCO_ROOT / "label_map.json"
LABEL_MAP.write_text(json.dumps({i: i if i else None for i in range(11)}, indent=2))
for split, ann_path, image_root in [("train", TRAIN_JSON, TRAIN_IMAGES),
                                     ("val", VAL_JSON, VAL_IMAGES)]:
    ann = json.loads(ann_path.read_text())
    ids = {c["id"] for c in ann["categories"]}
    assert ids == set(range(1, 11))
    assert ann["images"] and ann["annotations"]
    assert all(1 <= a["category_id"] < NUM_CLASSES for a in ann["annotations"])
    assert all("occlusion" in a for a in ann["annotations"])
    missing = [i["file_name"] for i in ann["images"]
               if not (image_root / i["file_name"]).is_file()]
    if missing or stats["per_split"][split]["missing_annotations"]:
        raise RuntimeError(f"{split}: thiếu ảnh/annotation; kiểm tra dataset. {missing[:5]}")
    print(split, "images:", len(ann["images"]), "annotations:", len(ann["annotations"]),
          "bad images:", stats["per_split"][split]["bad_images"])
del ann
GT_HASH = hashlib.sha256(VAL_JSON.read_bytes()).hexdigest()
TRAIN_HASH = hashlib.sha256(TRAIN_JSON.read_bytes()).hexdigest()
''')
md('''
## 4. Cấu hình E0/E1 (bước C)

Resolve recipe R18 chính thức rồi thay đường dẫn dữ liệu, batch và mapping cho Kaggle.
Giữ input **640×640**, 300 queries, 3 decoder layers, AdamW LR 1e-4 và augmentation của recipe.
Full: 120 epoch, augmentation dừng epoch 117, schedule gốc. Proxy: 2 epoch, LR giảm ở
epoch 1, augmentation dừng epoch 1, warmup rút ngắn. E0/E1 có cùng ngân sách trong mỗi mode.
Checkpoint phụ mỗi 20 epoch; luôn có `last.pth` và `best.pth` để hạn chế dung lượng output.
Batch Kaggle có thể khác lần train B2 cũ: dùng E0 mới để kết luận đóng góp của P2.
''')
code('''
sys.path.insert(0, str(REPO))
from src.core import YAMLConfig
base = copy.deepcopy(YAMLConfig(str(REPO / "configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml")).yaml_cfg)
base.pop("__include__", None)
base.update(num_classes=NUM_CLASSES, remap_mscoco_category=False,
            use_amp=USE_AMP, sync_bn=False, print_freq=100, checkpoint_freq=20)
for split, ann_file, images, batch in [
    ("train", TRAIN_JSON, TRAIN_IMAGES, BATCH_SIZE),
    ("val", VAL_JSON, VAL_IMAGES, VAL_BATCH_SIZE),
]:
    loader = base[f"{split}_dataloader"]
    loader.pop("total_batch_size", None)
    loader.update(batch_size=batch, num_workers=NUM_WORKERS,
                  shuffle=(split == "train"), drop_last=(split == "train"))
    loader["dataset"].update(img_folder=str(images), ann_file=str(ann_file))
if RUN_MODE == "proxy":
    base["epoches"] = 2
    base["lr_scheduler"].update(milestones=[1], gamma=0.1)
    base["lr_warmup_scheduler"]["warmup_duration"] = 100
    base["ema"]["warmups"] = 100
    base["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"] = 1
    base["train_dataloader"]["collate_fn"]["stop_epoch"] = 1
else:
    base["train_dataloader"]["collate_fn"]["stop_epoch"] = 117
BASE_CONFIG = CONFIGS / "visdrone_base.yml"
BASE_CONFIG.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
run([sys.executable, SCRIPTS / "prepare_p2.py", "--repo", REPO,
     "--base-config", BASE_CONFIG, "--output", CONFIGS], cwd=REPO)
CONFIG_PATHS = {name: CONFIGS / f"{name}.yml" for name in ["E0_control", "E1_p2"]}
for name, path in CONFIG_PATHS.items():
    cfg = yaml.safe_load(path.read_text())
    cfg["output_dir"] = str(OUT / name)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(name, "strides:", cfg["RTDETRTransformerv2"]["feat_strides"],
          "encoder attention index:", cfg["HybridEncoder"]["use_encoder_idx"],
          "epochs:", cfg["epoches"], "batch:", BATCH_SIZE)
''')

preflight = '''
import argparse, sys, torch
from pathlib import Path
from export_predictions import import_config
from prepare_p2 import smoke

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True)
p.add_argument('--config', required=True)
p.add_argument('--device', default='cuda:0')
p.add_argument('--synthetic', action='store_true')
a = p.parse_args()
torch.set_num_threads(4)
torch.manual_seed(42)
C = import_config(a.repo)
device = torch.device(a.device)
if a.synthetic:
    smoke(C, Path(a.config), device)
else:
    cfg = C(a.config)
    cfg.yaml_cfg['PResNet']['pretrained'] = False
    for split in ['train', 'val']:
        cfg.yaml_cfg[f'{split}_dataloader']['num_workers'] = 0
    model = cfg.model.to(device).train()
    criterion = cfg.criterion.to(device)
    optimizer = cfg.optimizer
    loader = cfg.train_dataloader
    loader.set_epoch(0)
    samples, targets = next(iter(loader))
    assert samples.shape[1:] == (3, 640, 640)
    for t in targets:
        assert t['boxes'].shape[-1] == 4 and torch.isfinite(t['boxes']).all()
        assert not t['labels'].numel() or (t['labels'].min() >= 1 and t['labels'].max() <= 10)
    samples = samples.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    amp = cfg.yaml_cfg.get('use_amp', False) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    with torch.autocast(device_type=device.type, enabled=amp):
        outputs = model(samples, targets=targets)
    with torch.autocast(device_type=device.type, enabled=False):
        loss = sum(criterion(outputs, targets, epoch=0, step=0, global_step=0).values())
    assert torch.isfinite(loss), loss
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    images, targets = next(iter(cfg.val_dataloader))
    with torch.inference_mode():
        sizes = torch.stack([t['orig_size'] for t in targets]).to(device)
        result = cfg.postprocessor.to(device)(model(images.to(device)), sizes)
    assert len(result) == len(targets)
    assert all(torch.isfinite(r['boxes']).all() for r in result)
    print(f'Real-data preflight passed: {Path(a.config).name}; loss={loss.item():.4f}')
'''
code('%%writefile /kaggle/working/rtdetrv2_next/preflight.py\n' + preflight)
md('''
## 5. Kiểm tra trước train

Smoke kiểm tra inference + train forward/backward của cả E0 và E1. Batch thật kiểm tra
augmentation, nhãn, loss/gradient, một bước optimizer với AMP và validation postprocessor
cho experiment được chọn. Chạy trong subprocess để trả lại VRAM trước khi train.
Preflight không tải pretrained; vòng train sẽ tải cùng backbone pretrained cho E0/E1.
''')
code('''
if RUN_SMOKE:
    for name, path in CONFIG_PATHS.items():
        run([sys.executable, SCRIPTS / "preflight.py", "--repo", REPO,
             "--config", path, "--device", DEVICE, "--synthetic"], cwd=REPO)
if RUN_REAL_PREFLIGHT:
    for name in EXPERIMENTS:
        run([sys.executable, SCRIPTS / "preflight.py", "--repo", REPO,
             "--config", CONFIG_PATHS[name], "--device", DEVICE], cwd=REPO)
''')

md('''
## 6. Train E1 P2 (hoặc E0 control)

Train mới dùng backbone pretrained, không dùng `best.pth` 3 level cũ để resume/tuning P2.
Để tiếp tục phiên Kaggle sau: lưu output hiện tại, gắn output đó vào Input và đặt `RESUME`
trỏ tới `last.pth`, giữ nguyên mode/batch/seed/config. Notebook kiểm tra manifest, ngân sách,
mapping và checkpoint rồi chép log/best vào output mới. Checkpoint lưu ở cuối epoch;
phần epoch đang chạy dở sẽ chạy lại khi resume.
Không đổi batch âm thầm nếu OOM: giảm `BATCH_SIZE`, chạy lại cấu hình/preflight và train mới;
E0 dùng cùng batch mới. Chạy từng experiment ở từng phiên nếu không đủ thời gian cho cả hai.
''')
code('''
def resume_signature(cfg):
    cfg = copy.deepcopy(cfg)
    cfg.pop("output_dir", None)
    cfg.pop("__include__", None)
    for split in ("train", "val"):
        ds = cfg[f"{split}_dataloader"]["dataset"]
        ds.pop("img_folder", None)
        ds.pop("ann_file", None)
    return cfg

def manifest_for(name, cfg):
    return {"experiment": name, "mode": RUN_MODE, "seed": SEED,
            "repo_commit": REPO_COMMIT, "train_sha256": TRAIN_HASH,
            "val_sha256": GT_HASH, "label_map": json.loads(LABEL_MAP.read_text()),
            "signature": resume_signature(cfg)}

for name in EXPERIMENTS:
    path = CONFIG_PATHS[name]
    cfg = yaml.safe_load(path.read_text())
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    current_manifest = manifest_for(name, cfg)
    resume_path = Path(RESUME[name]) if RESUME[name] else None
    if resume_path:
        previous = resume_path.parent
        for required in (resume_path.name, "run_config.yml", "run_manifest.json", "log.txt", "best.pth"):
            if not (previous / required).is_file():
                raise FileNotFoundError(f"Resume cần {previous / required}")
        old_manifest = json.loads((previous / "run_manifest.json").read_text())
        if old_manifest != current_manifest:
            raise ValueError("Resume khác architecture/budget/data/seed. Chỉ resume đúng experiment và mode.")
        old_cfg = yaml.safe_load((previous / "run_config.yml").read_text())
        assert resume_signature(old_cfg) == resume_signature(cfg)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        state = checkpoint["model"]
        levels = len({k.split(".")[2] for k in state if k.startswith("decoder.input_proj.")})
        if levels != cfg["RTDETRTransformerv2"]["num_levels"]:
            raise ValueError(f"Checkpoint có {levels} level; không khớp experiment {name}.")
        assert state["decoder.enc_score_head.weight"].shape[0] == NUM_CLASSES
        assert all(k in checkpoint for k in ["optimizer", "lr_scheduler", "last_epoch"])
        epoch = checkpoint["last_epoch"]
        history = [json.loads(line) for line in (previous / "log.txt").read_text().splitlines()]
        if not history or max(row["epoch"] for row in history) > epoch:
            raise ValueError("Dùng last.pth tương ứng log.txt; checkpoint cũ hơn log sẽ mất nhất quán best.")
        if epoch >= cfg["epoches"] - 1:
            raise ValueError("Checkpoint đã hoàn tất ngân sách này; không cần resume.")
        del state, checkpoint
        if previous.resolve() != folder.resolve():
            if (folder / "log.txt").exists() or (folder / "last.pth").exists():
                raise FileExistsError(f"Output {folder} đã có lượt train; dùng thư mục/phiên mới.")
            for filename in ("log.txt", "best.pth"):
                shutil.copy2(previous / filename, folder / filename)
    elif (folder / "last.pth").exists() or (folder / "log.txt").exists():
        raise FileExistsError(f"{folder} đã có train. Đặt RESUME trỏ tới last.pth để tiếp tục.")
    shutil.copy2(path, folder / "run_config.yml")
    (folder / "run_manifest.json").write_text(json.dumps(current_manifest, indent=2), encoding="utf-8")
    command = [sys.executable, "-u", REPO / "tools/train.py", "-c", path,
               "--seed", SEED, "--device", DEVICE, "--output-dir", folder]
    if USE_AMP:
        command.append("--use-amp")
    if resume_path:
        command.extend(["--resume", resume_path])
    run(command, cwd=REPO, log=folder / "console.log")
    assert (folder / "best.pth").exists()
''')
md('''
## 7. Export + small/occlusion (bước B)

Nạp **EMA** từ best checkpoint bằng `strict=True`, đúng validation transforms và cùng GT.
Không lọc confidence trước AP. Recall nhóm dùng confidence 0.3, IoU 0.5, matching trên toàn
bộ GT trước khi chia nhóm; đây không phải COCO ARsmall. Xuất `metrics.json`, `per_class.csv`,
`recall_groups.csv`, `failure_images.csv`, `errors.json` và 20 ảnh lỗi.

`RUN_B2_EVALUATION=True` thêm kết quả B2 từ checkpoint cũ. Nếu không có config thực tế,
config R18/11 đầu ra là giả định lấy từ notebook cũ: phải xác nhận protocol trước khi so AP.
''')
code('''
def export_evaluate(name, config, checkpoint):
    dest = OUT / name / "export"
    run([sys.executable, SCRIPTS / "export_predictions.py", "--repo", REPO,
         "--config", config, "--checkpoint", checkpoint, "--annotations", VAL_JSON,
         "--label-map", LABEL_MAP, "--device", DEVICE, "--workers", NUM_WORKERS,
         "--batch-size", VAL_BATCH_SIZE, "--output", dest], cwd=REPO)
    run([sys.executable, SCRIPTS / "evaluate.py", "--annotations", VAL_JSON,
         "--predictions", dest / "predictions.json", "--coverage", dest / "coverage.json",
         "--score", "0.3", "--iou", "0.5", "--images", VAL_IMAGES,
         "--output", dest / "evaluation"])

for name in EXPERIMENTS:
    export_evaluate(name, CONFIG_PATHS[name], OUT / name / "best.pth")
if RUN_B2_EVALUATION:
    if not B2_CHECKPOINT.is_file():
        raise FileNotFoundError(B2_CHECKPOINT)
    b2 = copy.deepcopy(YAMLConfig(str(B2_CONFIG or BASE_CONFIG)).yaml_cfg)
    b2.pop("__include__", None)
    b2["val_dataloader"]["dataset"].update(img_folder=str(VAL_IMAGES), ann_file=str(VAL_JSON))
    b2_path = CONFIGS / "B2_eval.yml"
    b2_path.write_text(yaml.safe_dump(b2, sort_keys=False), encoding="utf-8")
    export_evaluate("B2", b2_path, B2_CHECKPOINT)
''')

benchmark = '''
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from export_predictions import import_config, checkpoint_weights

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True)
p.add_argument('--config', required=True)
p.add_argument('--checkpoint', required=True)
p.add_argument('--output', required=True)
p.add_argument('--device', default='cuda:0')
a = p.parse_args()
cfg = import_config(a.repo)(a.config)
cfg.yaml_cfg['PResNet']['pretrained'] = False
model = cfg.model
checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
state, source = checkpoint_weights(checkpoint)
model.load_state_dict(state, strict=True)
del state, checkpoint
device = torch.device(a.device)
model = model.to(device).eval()
x = torch.rand(1, 3, *cfg.yaml_cfg['eval_spatial_size'], device=device)
def sync():
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
with torch.inference_mode():
    for _ in range(50):
        model(x)
    sync()
    times = []
    for _ in range(200):
        sync()
        start = time.perf_counter()
        model(x)
        sync()
        times.append(1000 * (time.perf_counter() - start))
mean = float(np.mean(times))
result = {'Latency_ms': mean, 'Latency_median_ms': float(np.median(times)),
          'Latency_std_ms': float(np.std(times)), 'FPS': 1000 / mean,
          'Parameters_M': sum(p.numel() for p in model.parameters()) / 1e6,
          'protocol': {'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                       'device': str(device), 'torch': str(torch.__version__),
                       'input': list(x.shape), 'precision': 'float32', 'backend': 'PyTorch eager',
                       'warmup': 50, 'runs': 200, 'weights': source,
                       'scope': 'model forward only, excludes preprocessing/postprocessing/I/O'}}
Path(a.output).write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
'''
code('%%writefile /kaggle/working/rtdetrv2_next/benchmark.py\n' + benchmark)
md('''
## 8. Benchmark riêng sau train

Batch 1, 640×640, FP32, PyTorch eager, warmup 50 lần, đo 200 lần và CUDA synchronize.
Đây là latency model, không gồm đọc ảnh/resize/postprocessor/vẽ/ghi video.
Train dùng AMP nhưng benchmark E0/E1 cùng FP32. Chỉ đối chiếu FPS trên cùng GPU/protocol;
mốc 85% phải dùng baseline đo lại, không lấy trực tiếp FPS cũ khác GPU.
''')
code('''
if RUN_BENCHMARK:
    jobs = [(name, CONFIG_PATHS[name], OUT / name / "best.pth") for name in EXPERIMENTS]
    if RUN_B2_EVALUATION:
        jobs.append(("B2", b2_path, B2_CHECKPOINT))
    for name, config, checkpoint in jobs:
        run([sys.executable, SCRIPTS / "benchmark.py", "--repo", REPO, "--config", config,
             "--checkpoint", checkpoint, "--device", DEVICE,
             "--output", OUT / name / "benchmark.json"], cwd=REPO)
''')
md('''
## 9. Bảng kết quả và output cần giữ

APsmall toàn bộ 10 class và APsmall riêng phương tiện được ghi riêng. Recall-heavy kèm
số GT; ô trống nghĩa là nhóm không có GT. Chỉ chọn model trên validation; test/clip riêng
là giai đoạn sau. Notebook không tự suy ra P2 có lợi trước khi có E0/E1 cùng ngân sách.
''')
code('''
import pandas as pd
from IPython.display import display, FileLink
rows = []
for folder in sorted(OUT.iterdir()):
    metrics_path = folder / "export/evaluation/metrics.json"
    if not metrics_path.exists():
        continue
    m = json.loads(metrics_path.read_text())
    heavy = next(g for g in m["groups"] if g["scope"] == "vehicles" and g["group"] == "heavy")
    small_heavy = next(g for g in m["groups"] if g["scope"] == "vehicles" and g["group"] == "small_heavy")
    b = json.loads((folder / "benchmark.json").read_text()) if (folder / "benchmark.json").exists() else {}
    rows.append({"Model": folder.name, "Mode": RUN_MODE, "AP": m["all_classes"]["AP"],
                 "APsmall_all": m["all_classes"]["APsmall"], "AP_vehicle": m["vehicles"]["AP"],
                 "APsmall_vehicle": m["vehicles"]["APsmall"], "Recall_heavy_vehicle": heavy["recall"],
                 "GT_heavy_vehicle": heavy["gt"], "Recall_small_heavy_vehicle": small_heavy["recall"],
                 "GT_small_heavy_vehicle": small_heavy["gt"],
                 "FPS": b.get("FPS"), "Latency_ms": b.get("Latency_ms"), "Parameters_M": b.get("Parameters_M")})
table = pd.DataFrame(rows)
table.to_csv(OUT / "results.csv", index=False)
display(table.round(4))

# Artifact tự chứa config, GT, mapping, scripts, log và checkpoint cho phiên kế tiếp.
for filename in ("instances_train.json", "instances_val.json", "dataset_stats.json",
                 "rtdetr_dataset.json", "label_map.json", "classes.txt"):
    (OUT / "data").mkdir(exist_ok=True)
    shutil.copy2(COCO_ROOT / filename, OUT / "data" / filename)
shutil.copytree(SCRIPTS, OUT / "scripts", dirs_exist_ok=True)
(OUT / "repo_patch.diff").write_text(subprocess.check_output(
    ["git", "-C", str(REPO.parent), "diff"], text=True), encoding="utf-8")
(OUT / "environment.json").write_text(json.dumps({
    "repo_commit": REPO_COMMIT, "torch": str(torch.__version__),
    "torchvision": str(torchvision.__version__), "gpu": torch.cuda.get_device_name(0),
    "mode": RUN_MODE, "seed": SEED, "batch_size": BATCH_SIZE,
    "epochs": base["epoches"], "train_amp": USE_AMP,
    "train_annotations_sha256": TRAIN_HASH, "val_annotations_sha256": GT_HASH,
}, indent=2), encoding="utf-8")
archive = shutil.make_archive(str(WORK / f"rtdetrv2_p2_{RUN_MODE}"), "zip", OUT)
print("Output folder:", OUT)
print("Giữ best.pth để đánh giá; last.pth + config/manifest/log/best để resume.")
display(FileLink(Path(archive).name))
''')
md('''
**Chạy tiếp theo kế hoạch:** thử `proxy` 2 epoch cho E0/E1 để xác nhận vòng train thật,
sau đó train `full` với cùng batch/seed cho hai model. Dùng `results.csv`,
`per_class.csv`, `recall_groups.csv` và ảnh lỗi để kiểm tra tín hiệu APsmall/Recall-heavy.
Output Kaggle phải được lưu trước khi kết thúc phiên; ZIP chỉ được tạo khi chạy tới cell cuối,
còn `last.pth`/`best.pth` đã nằm trong output từng experiment sau mỗi epoch.

Nguồn kiến trúc: [RT-DETR chính thức](https://github.com/lyuwenyu/RT-DETR/tree/29320b6fd828f8e0987a71426cf2d961b09dfed7/rtdetrv2_pytorch).
''')

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
    'kaggle': {'isInternetEnabled': True, 'isGpuEnabled': True,
               'accelerator': 'gpu', 'language': 'python', 'sourceType': 'notebook', 'dataSources': []},
})
nbf.validate(notebook)
output = ROOT / 'rtdetrv2_p2_visdrone_kaggle.ipynb'
nbf.write(notebook, output)
print(output)
