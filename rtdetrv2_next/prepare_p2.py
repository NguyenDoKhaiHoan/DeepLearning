"""Generate paired E0/E1 configurations from an existing RT-DETRv2 baseline."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import torch
import yaml

from export_predictions import import_config


def add_p2(base):
    result = copy.deepcopy(base)
    architecture = result['RTDETR']
    if (architecture['backbone'] != 'PResNet' or architecture['encoder'] != 'HybridEncoder'
            or architecture['decoder'] != 'RTDETRTransformerv2'):
        raise ValueError('P2 preparation supports official PResNet/HybridEncoder/RTDETRTransformerv2 only.')
    backbone, encoder, decoder = (result[key] for key in
                                  ['PResNet', 'HybridEncoder', 'RTDETRTransformerv2'])
    depth = backbone['depth']
    if depth not in [18, 34, 50, 101]:
        raise ValueError(f'Unsupported PResNet depth: {depth}')
    if (backbone.get('return_idx', [1, 2, 3]) != [1, 2, 3]
            or encoder.get('feat_strides', [8, 16, 32]) != [8, 16, 32]
            or decoder.get('num_levels', 3) != 3):
        raise ValueError('Expected an unmodified three-level P3/P4/P5 baseline.')
    channels = [64, 128, 256, 512] if depth < 50 else [256, 512, 1024, 2048]
    hidden = encoder.get('hidden_dim', 256)
    if encoder.get('in_channels') != channels[1:]:
        raise ValueError('Baseline encoder channels do not match PResNet depth.')
    if decoder.get('hidden_dim', 256) != hidden:
        raise ValueError('Encoder and decoder hidden_dim must agree.')
    points = decoder.get('num_points', 4)
    if isinstance(points, list):
        if len(points) != 3:
            raise ValueError('Expected three decoder sampling-point counts.')
        points = [points[0], *points]
    backbone['return_idx'] = [0, 1, 2, 3]
    encoder.update(in_channels=channels, feat_strides=[4, 8, 16, 32],
                   use_encoder_idx=[index+1 for index in encoder.get('use_encoder_idx', [2])])
    decoder.update(feat_channels=[hidden]*4, feat_strides=[4, 8, 16, 32],
                   num_levels=4, num_points=points)
    return result


def smoke(YAMLConfig, path, device):
    cfg = YAMLConfig(str(path))
    cfg.yaml_cfg['PResNet']['pretrained'] = False
    model = cfg.model.to(device)
    size = cfg.yaml_cfg.get('eval_spatial_size', [640, 640])
    if not size or any(s % 32 for s in size):
        raise ValueError('Smoke test needs eval_spatial_size divisible by 32.')
    images = torch.rand(2, 3, *size, device=device)
    model.eval()
    with torch.no_grad():
        output = model(images)
    assert output['pred_boxes'].shape[-1] == 4
    assert torch.isfinite(output['pred_logits']).all()
    model.train()
    targets = [{'labels': torch.tensor([0], device=device),
                'boxes': torch.tensor([[0.5, 0.5, 0.1, 0.1]], device=device)} for _ in range(2)]
    loss = sum(cfg.criterion.to(device)(model(images, targets), targets).values())
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite smoke-test loss.')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError('Non-finite smoke-test gradient.')
    print(f'Smoke passed: {path.name}; loss={loss.item():.4f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--base-config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    args.repo, args.base_config, args.output = (p.resolve() for p in
                                               [args.repo, args.base_config, args.output])
    YAMLConfig = import_config(args.repo)
    os.chdir(args.repo)
    base = YAMLConfig(str(args.base_config)).yaml_cfg
    base.pop('__include__', None)
    p2 = add_p2(base)
    args.output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, config in [('E0_control', base), ('E1_p2', p2)]:
        config['output_dir'] = str(args.output / name)
        path = args.output / f'{name}.yml'
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        paths.append(path)
        print(f'Config: {path}')
    if args.smoke:
        for path in paths:
            torch.manual_seed(42)
            smoke(YAMLConfig, path, torch.device(args.device))


if __name__ == '__main__':
    main()
