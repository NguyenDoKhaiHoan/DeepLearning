"""COCO bbox metrics and matched-GT small/occlusion diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

VEHICLES = {'bicycle', 'car', 'van', 'truck', 'tricycle',
            'awning-tricycle', 'bus', 'motor'}
METRICS = ['AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge',
           'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge']


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                   allow_nan=False), encoding='utf-8')


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_results(gt, predictions):
    if predictions:
        return gt.loadRes(copy.deepcopy(predictions))
    result = COCO()
    result.dataset = {'images': copy.deepcopy(gt.dataset['images']),
                      'categories': copy.deepcopy(gt.dataset['categories']),
                      'annotations': []}
    result.createIndex()
    return result


def validate_predictions(gt, predictions, coverage):
    expected = set(gt.getImgIds())
    seen = coverage['image_ids']
    if len(seen) != len(set(seen)) or set(seen) != expected:
        raise ValueError('Coverage must include every validation image exactly once.')
    categories = set(gt.getCatIds())
    for row in predictions:
        if row['image_id'] not in expected or row['category_id'] not in categories:
            raise ValueError(f'Unknown image/category ID: {row}')
        box = row['bbox']
        if (len(box) != 4 or not all(math.isfinite(x) for x in box)
                or box[2] <= 0 or box[3] <= 0
                or not math.isfinite(row['score']) or not 0 <= row['score'] <= 1):
            raise ValueError(f'Invalid xywh box/score: {row}')


def coco_metrics(gt, predictions, cat_ids=None):
    evaluator = COCOeval(gt, load_results(gt, predictions), 'bbox')
    evaluator.params.imgIds = sorted(gt.getImgIds())
    if cat_ids is not None:
        evaluator.params.catIds = sorted(cat_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    metrics = {key: float(value) if value >= 0 else None
               for key, value in zip(METRICS, evaluator.stats)}
    per_class = []
    for index, category_id in enumerate(evaluator.params.catIds):
        row = {'category_id': category_id, 'class': gt.cats[category_id]['name']}
        for key, area in [('AP', 0), ('APsmall', 1)]:
            values = evaluator.eval['precision'][:, :, index, area, -1]
            valid = values[values >= 0]
            row[key] = float(valid.mean()) if valid.size else None
        per_class.append(row)
    return metrics, per_class


def box_iou(box, other):
    x, y, w, h = box
    xx, yy, ww, hh = other
    intersection = max(0., min(x+w, xx+ww)-max(x, xx)) * max(
        0., min(y+h, yy+hh)-max(y, yy))
    union = w*h + ww*hh - intersection
    return intersection / union if union > 0 else 0.


def group_names(annotation):
    area = annotation['area']
    size = 'small' if area <= 1024 else 'medium' if area <= 9216 else 'large'
    occlusion = {0: 'none', 1: 'partial', 2: 'heavy'}.get(
        annotation.get('occlusion'), 'unknown')
    return ['all', size, occlusion, f'{size}_{occlusion}']


def diagnostics(gt, predictions, score=0.3, iou=0.5, vehicle_ids=None):
    selected = [row for row in predictions if row['score'] >= score]
    dt = load_results(gt, selected)
    ev = COCOeval(gt, dt, 'bbox')
    ev.params.iouThrs = np.array([iou])
    ev.params.areaRng = [[0, 1e10]]
    ev.params.areaRngLbl = ['all']
    ev.params.maxDets = [100]
    ev.evaluate()
    matched, evaluated_dt, ignored_dt = set(), set(), set()
    for result in ev.evalImgs:
        if result is None:
            continue
        matched.update(gid for gid, did, ignore in zip(
            result['gtIds'], result['gtMatches'][0], result['gtIgnore'])
                       if did > 0 and not ignore)
        for did, ignore in zip(result['dtIds'], result['dtIgnore'][0]):
            evaluated_dt.add(did)
            if ignore:
                ignored_dt.add(did)
    matched_dt = {int(did) for result in ev.evalImgs if result is not None
                  for did in result['gtMatches'][0] if did > 0}
    by_image = defaultdict(list)
    gt_by_image = defaultdict(list)
    for did in evaluated_dt - ignored_dt:
        by_image[dt.anns[did]['image_id']].append(dt.anns[did])
    for annotation in gt.anns.values():
        if not annotation.get('iscrowd', 0):
            gt_by_image[annotation['image_id']].append(annotation)

    counts = defaultdict(Counter)
    errors, image_counts = [], defaultdict(Counter)
    for annotations in gt_by_image.values():
        for annotation in annotations:
            detected = annotation['id'] in matched
            scopes = ['all_classes']
            if annotation['category_id'] in vehicle_ids:
                scopes.append('vehicles')
            for scope in scopes:
                for group in group_names(annotation):
                    counts[(scope, group)]['gt'] += 1
                    counts[(scope, group)]['tp'] += int(detected)
            if annotation['category_id'] not in vehicle_ids:
                continue
            image_id = annotation['image_id']
            image_counts[image_id]['gt_vehicles'] += 1
            if detected:
                image_counts[image_id]['tp_vehicles'] += 1
                continue
            candidates = by_image[image_id]
            same = [box_iou(annotation['bbox'], p['bbox']) for p in candidates
                    if p['category_id'] == annotation['category_id']]
            wrong = [box_iou(annotation['bbox'], p['bbox']) for p in candidates
                     if p['category_id'] != annotation['category_id']]
            reason = ('class_confusion' if max(wrong, default=0) >= iou else
                      'localization' if max(same, default=0) >= 0.1 else
                      'miss')
            image_counts[image_id]['fn_vehicles'] += 1
            groups = group_names(annotation)
            if 'small' in groups:
                image_counts[image_id]['fn_small'] += 1
            if 'heavy' in groups:
                image_counts[image_id]['fn_heavy'] += 1
            errors.append({'image_id': image_id, 'kind': 'FN', 'reason': reason,
                           'category_id': annotation['category_id'],
                           'gt_id': annotation['id'], 'bbox': annotation['bbox'],
                           'groups': groups})
    for did in sorted(evaluated_dt - matched_dt - ignored_dt):
        prediction = dt.anns[did]
        if prediction['category_id'] not in vehicle_ids:
            continue
        annotations = gt_by_image[prediction['image_id']]
        same = [box_iou(prediction['bbox'], a['bbox']) for a in annotations
                if a['category_id'] == prediction['category_id']]
        wrong = [box_iou(prediction['bbox'], a['bbox']) for a in annotations
                 if a['category_id'] != prediction['category_id']]
        reason = ('duplicate' if max(same, default=0) >= iou else
                  'class_confusion' if max(wrong, default=0) >= iou else
                  'localization' if max(same, default=0) >= 0.1 else 'background')
        image_counts[prediction['image_id']]['fp_vehicles'] += 1
        errors.append({'image_id': prediction['image_id'], 'kind': 'FP',
                       'reason': reason, 'category_id': prediction['category_id'],
                       'prediction_id': did, 'bbox': prediction['bbox'],
                       'score': prediction['score']})
    groups = []
    for scope in ['all_classes', 'vehicles']:
        for group in ['all', 'small', 'medium', 'large', 'none', 'partial',
                      'heavy', 'unknown', 'small_none', 'small_partial',
                      'small_heavy', 'small_unknown']:
            count = counts[(scope, group)]
            n, tp = count['gt'], count['tp']
            groups.append({'scope': scope, 'group': group, 'gt': n, 'tp': tp,
                           'fn': n-tp, 'recall': tp/n if n else None})
    rows = []
    for image_id in sorted(gt.getImgIds()):
        count = image_counts[image_id]
        rows.append({'image_id': image_id,
                     'file_name': gt.imgs[image_id]['file_name'],
                     **{key: count[key] for key in ['gt_vehicles', 'tp_vehicles',
                        'fn_vehicles', 'fn_small', 'fn_heavy', 'fp_vehicles']}})
    rows.sort(key=lambda r: (-r['fn_small']-r['fn_heavy'], -r['fn_vehicles'],
                             r['image_id']))
    return groups, errors, rows


def render_failures(gt, errors, rows, image_root, output, limit):
    from PIL import Image, ImageDraw
    grouped = defaultdict(list)
    for error in errors:
        grouped[error['image_id']].append(error)
    output.mkdir(exist_ok=True)
    for row in [r for r in rows if grouped[r['image_id']]][:limit]:
        with Image.open(image_root / row['file_name']) as source:
            image = source.convert('RGB')
        draw = ImageDraw.Draw(image)
        for error in grouped[row['image_id']]:
            x, y, w, h = error['bbox']
            color = 'red' if error['kind'] == 'FN' else 'yellow'
            draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
            label = f"{error['kind']} {gt.cats[error['category_id']]['name']}"
            draw.text((max(0, x), max(0, y-12)), label, fill=color,
                      stroke_width=1, stroke_fill='black')
        image.save(output / f"{row['image_id']:06d}.jpg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--coverage', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--score', type=float, default=0.3)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--images', type=Path)
    parser.add_argument('--visualize', type=int, default=20)
    args = parser.parse_args()
    if not 0 <= args.score <= 1 or not 0.1 < args.iou <= 1:
        parser.error('score must be in [0,1], IoU in (0.1,1].')
    gt = COCO(str(args.annotations))
    predictions = json.loads(args.predictions.read_text(encoding='utf-8'))
    coverage = json.loads(args.coverage.read_text(encoding='utf-8'))
    digest = hashlib.sha256(args.annotations.read_bytes()).hexdigest()
    if coverage.get('annotations_sha256') != digest:
        raise ValueError('Coverage manifest does not correspond to this GT file.')
    validate_predictions(gt, predictions, coverage)
    vehicle_ids = {c['id'] for c in gt.dataset['categories'] if c['name'] in VEHICLES}
    if not vehicle_ids:
        raise ValueError('No VisDrone vehicle category names found in GT.')
    overall, per_class = coco_metrics(gt, predictions)
    vehicles, _ = coco_metrics(gt, predictions, vehicle_ids)
    groups, errors, rows = diagnostics(gt, predictions, args.score, args.iou, vehicle_ids)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        'protocol': {'evaluation': 'COCO bbox; converted VisDrone, not official challenge',
                     'maxDets': [1, 10, 100], 'diagnostic_score': args.score,
                     'diagnostic_iou': args.iou,
                     'diagnostic_maxDets': '100 per image per category (COCO)',
                     'diagnostic_small': 'GT area <= 1024 original-image pixels',
                     'recall_definition': 'micro TP/GT after matching all GT; not COCO AR',
                     'annotations_sha256': digest,
                     'coverage_images': len(coverage['image_ids']),
                     'vehicle_categories': sorted(vehicle_ids),
                     'export': coverage.get('export', {})},
        'all_classes': overall, 'vehicles': vehicles, 'groups': groups,
        'error_counts': dict(Counter(f"{e['kind']}_{e['reason']}" for e in errors)),
        'warnings': ['Ignored regions were excluded by the original converter; '
                     'these results are not official VisDrone challenge metrics.']}
    write_json(args.output / 'metrics.json', summary)
    write_json(args.output / 'errors.json', errors)
    write_csv(args.output / 'recall_groups.csv', groups,
              ['scope', 'group', 'gt', 'tp', 'fn', 'recall'])
    write_csv(args.output / 'per_class.csv', per_class,
              ['category_id', 'class', 'AP', 'APsmall'])
    write_csv(args.output / 'failure_images.csv', rows,
              ['image_id', 'file_name', 'gt_vehicles', 'tp_vehicles', 'fn_vehicles',
               'fn_small', 'fn_heavy', 'fp_vehicles'])
    if args.images and args.visualize > 0:
        render_failures(gt, errors, rows, args.images, args.output / 'failures',
                        args.visualize)
    print(f'Results: {args.output.resolve()}')


if __name__ == '__main__':
    main()
