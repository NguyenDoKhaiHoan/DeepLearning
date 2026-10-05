"""Exercise generated notebook cells on tiny fixtures with the official CPU repo.

Usage: python validate_kaggle_notebook.py --repo <rtdetrv2_pytorch>
Writes all validation artifacts to a new temporary directory, never user data.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import nbformat
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    notebook = nbformat.read(root / 'rtdetrv2_p2_visdrone_kaggle.ipynb', as_version=4)
    nbformat.validate(notebook)
    sources = [c.source for c in notebook.cells if c.cell_type == 'code']
    for source in sources:
        python = source.partition('\n')[2] if source.startswith('%%writefile') else source
        compile(python, '<notebook cell>', 'exec')
    temp = Path(tempfile.mkdtemp(prefix='rtdetrv2-kaggle-verify-'))
    repo = temp / 'RT-DETR' / 'rtdetrv2_pytorch'
    for name in ('src', 'configs', 'tools'):
        shutil.copytree(args.repo / name, repo / name)
    scripts = temp / 'scripts'
    scripts.mkdir()
    for source in sources:
        if source.startswith('%%writefile'):
            header, _, body = source.partition('\n')
            (scripts / Path(header.split()[-1]).name).write_text(body, encoding='utf-8')
    data = temp / 'input'
    for split in ('train', 'val'):
        split_root = data / f'VisDrone2019-DET-{split}'
        (split_root / 'images').mkdir(parents=True)
        (split_root / 'annotations').mkdir()
        for index in range(2):
            Image.new('RGB', (128, 96), (55 + index * 40, 80, 90)).save(
                split_root / 'images' / f'{index}.jpg')
            (split_root / 'annotations' / f'{index}.txt').write_text(
                f'20,20,15,18,1,4,0,{index * 2}\n60,40,20,20,1,10,0,1\n0,0,5,5,0,0,0,0\n')
    # Execute actual notebook functions and cells with paths/settings overridden.
    import copy, hashlib, math, subprocess, yaml, torch, torchvision
    torch.set_num_threads(4)
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    env = dict(globals(), copy=copy, hashlib=hashlib, math=math, subprocess=subprocess,
               yaml=yaml, torch=torch, torchvision=torchvision)
    env.update(WORK=temp, REPO=repo, SCRIPTS=scripts, DATA_ROOT=data,
               RUN_MODE='proxy', EXPERIMENTS=['E0_control', 'E1_p2'],
               BATCH_SIZE=2, VAL_BATCH_SIZE=2, NUM_WORKERS=0, SEED=42,
               USE_AMP=False, RUN_SMOKE=True, RUN_REAL_PREFLIGHT=True,
               RUN_B2_EVALUATION=False, RUN_BENCHMARK=False,
               RESUME={'E0_control': None, 'E1_p2': None}, DEVICE='cpu',
               REPO_COMMIT='29320b6fd828f8e0987a71426cf2d961b09dfed7',
               COCO_ROOT=temp / 'coco', OUT=temp / 'experiments' / 'proxy')
    env['CONFIGS'] = env['OUT'] / 'configs'
    env['CONFIGS'].mkdir(parents=True)
    function = next(n for n in ast.parse(sources[0]).body if isinstance(n, ast.FunctionDef))
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<run function>', 'exec'), env)
    setup = next(s for s in sources if 'engine = REPO' in s)
    exec(setup[setup.index('engine = REPO'):], env)
    def execute(marker):
        source = next(s for s in sources if s.startswith(marker))
        exec(compile(source, f'<cell: {marker}>', 'exec'), env)
    execute('if DATA_ROOT is None')
    execute('sys.path.insert(0, str(REPO))')
    execute('if RUN_SMOKE:')
    for path in env['CONFIG_PATHS'].values():
        cfg = yaml.safe_load(path.read_text())
        cfg['PResNet']['pretrained'] = False  # Fixture training needs no network download.
        cfg['epoches'] = 1
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    execute('def resume_signature')
    execute('def export_evaluate')
    for name in env['EXPERIMENTS']:
        folder = env['OUT'] / name
        metrics = json.loads((folder / 'export/evaluation/metrics.json').read_text())
        assert metrics['protocol']['coverage_images'] == 2
        heavy = next(g for g in metrics['groups'] if g['scope'] == 'vehicles' and g['group'] == 'heavy')
        assert heavy['gt'] == 1
        assert (folder / 'best.pth').exists() and (folder / 'last.pth').exists()
        assert (folder / 'export/evaluation/failures/000001.jpg').exists()
    # Run the exact helper benchmark on CPU, with reduced repetitions for the fixture.
    benchmark = scripts / 'benchmark.py'
    benchmark.write_text(benchmark.read_text().replace('range(50)', 'range(2)').replace(
        'range(200)', 'range(3)'), encoding='utf-8')
    env['run']([sys.executable, benchmark, '--repo', repo,
                '--config', env['CONFIG_PATHS']['E1_p2'], '--device', 'cpu',
                '--checkpoint', env['OUT'] / 'E1_p2/best.pth',
                '--output', env['OUT'] / 'E1_p2/benchmark.json'], cwd=repo)
    print('PASSED: schema + syntax, converter/mapping, E0/E1 synthetic and real preflight,')
    print('one epoch E0/E1 train+validation, EMA export, diagnostics/rendering, CPU benchmark.')
    print('Validation artifacts:', temp)


if __name__ == '__main__':
    main()
