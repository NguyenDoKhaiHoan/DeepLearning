"""Export validation predictions using the official RT-DETRv2 loader."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


def import_config(repo):
    repo = Path(repo).resolve()
    if not (repo / 'src' / 'core').is_dir():
        raise ValueError('--repo must point to rtdetrv2_pytorch (contains src/core).')
    sys.path.insert(0, str(repo))
    from src.core import YAMLConfig
    return YAMLConfig


def checkpoint_weights(checkpoint, use_model=False):
    if not use_model and checkpoint.get('ema') is not None:
        return checkpoint['ema']['module'], 'ema'
    return checkpoint['model'], 'model'


def label_mapping(categories, mode, map_file=None):
    if map_file:
        data = json.loads(Path(map_file).read_text(encoding='utf-8'))
        mapping = {int(key): int(value) if value is not None else None
                   for key, value in data.items()}
    elif mode == 'category':
        mapping = {int(c['id']): int(c['id']) for c in categories}
    else:
        mapping = {index: int(c['id']) for index, c in enumerate(categories)}
    ids = {c['id'] for c in categories}
    retained = [value for value in mapping.values() if value is not None]
    if set(retained) != ids or len(retained) != len(ids):
        raise ValueError('Label mapping must map bijectively onto GT category IDs.')
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    labels = parser.add_mutually_exclusive_group(required=True)
    labels.add_argument('--label-mode', choices=['category', 'contiguous0'])
    labels.add_argument('--label-map', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--model-weights', action='store_true', help='Use model instead of EMA.')
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error('batch-size must be positive and workers nonnegative.')
    args.repo = args.repo.resolve()
    args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.annotations = args.annotations.resolve()
    args.output = args.output.resolve()
    YAMLConfig = import_config(args.repo)
    # Match repo-relative dataset paths and any local modifications in the config.
    os.chdir(args.repo)
    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg.setdefault('PResNet', {})['pretrained'] = False
    cfg.yaml_cfg['val_dataloader'].pop('total_batch_size', None)
    cfg.yaml_cfg['val_dataloader'].update(batch_size=args.batch_size,
                                         num_workers=args.workers,
                                         shuffle=False, drop_last=False)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use the training environment or --device cpu.')
    annotations = json.loads(args.annotations.read_text(encoding='utf-8'))
    mapping = label_mapping(annotations['categories'], args.label_mode, args.label_map)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state, source = checkpoint_weights(checkpoint, args.model_weights)
    model = cfg.model
    model.load_state_dict(state, strict=True)
    del state, checkpoint
    device = torch.device(args.device)
    model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval()
    loader = cfg.val_dataloader
    expected = {row['id']: row['file_name'] for row in annotations['images']}
    actual = {row['id']: row['file_name'] for row in loader.dataset.coco.dataset['images']}
    if actual != expected:
        raise ValueError('Config validation image IDs/names differ from --annotations.')
    predictions, seen = [], []
    with torch.inference_mode():
        for step, (images, targets) in enumerate(loader, 1):
            sizes = torch.stack([target['orig_size'] for target in targets]).to(device)
            outputs = postprocessor(model(images.to(device)), sizes)
            for target, output in zip(targets, outputs):
                image_id = int(target['image_id'].item())
                seen.append(image_id)
                for box, score, label in zip(output['boxes'].cpu().tolist(),
                                             output['scores'].cpu().tolist(),
                                             output['labels'].cpu().tolist()):
                    if label not in mapping:
                        raise ValueError(f'Unmapped output label {label}; check training class mapping.')
                    if mapping[label] is None:
                        continue
                    x1, y1, x2, y2 = box
                    if x2 <= x1 or y2 <= y1:
                        continue
                    predictions.append({'image_id': image_id, 'category_id': mapping[label],
                                        'bbox': [x1, y1, x2-x1, y2-y1], 'score': score})
            if step % 50 == 0:
                print(f'Validation batches: {step}/{len(loader)}')
    if len(seen) != len(set(seen)) or set(seen) != set(expected):
        raise ValueError('Loader did not cover validation exactly once.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'predictions.json').write_text(json.dumps(predictions, allow_nan=False),
                                                encoding='utf-8')
    try:
        commit = subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'],
                                          text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    metadata = {'image_ids': seen,
                'annotations_sha256': hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
                'export': {'config': str(args.config), 'checkpoint': str(args.checkpoint),
                           'weights': source, 'label_map': mapping, 'repo_commit': commit,
                           'torch': torch.__version__, 'device': str(device),
                           'precision': 'float32', 'score_filter': None}}
    (args.output / 'coverage.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    import yaml
    (args.output / 'resolved_config.yml').write_text(
        yaml.safe_dump(cfg.yaml_cfg, sort_keys=False), encoding='utf-8')
    print(f'Exported {len(predictions)} detections from {len(seen)} images to {args.output}')


if __name__ == '__main__':
    main()
