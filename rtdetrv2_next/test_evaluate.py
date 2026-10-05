import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pycocotools.coco import COCO

from evaluate import coco_metrics, diagnostics, group_names, validate_predictions
from export_predictions import import_config, label_mapping
from prepare_p2 import add_p2


def fixture(annotations):
    gt = COCO()
    gt.dataset = {'info': {}, 'images': [dict(id=1, file_name='a.jpg'),
                                        dict(id=2, file_name='empty.jpg')],
                  'categories': [dict(id=3, name='car'), dict(id=9, name='bus')],
                  'annotations': annotations}
    gt.createIndex()
    return gt


def ann(aid, x=0, category=3, occ=0):
    return dict(id=aid, image_id=1, category_id=category,
                bbox=[x, 0, 20, 20], area=400, iscrowd=0, occlusion=occ)


def pred(x=0, category=3, score=.9):
    return dict(image_id=1, category_id=category, bbox=[x, 0, 20, 20], score=score)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)

    def run_diagnostics(self, annotations, predictions):
        return diagnostics(fixture(annotations), predictions, vehicle_ids={3, 9})

    def test_perfect_detection_and_absent_group(self):
        groups, errors, images = self.run_diagnostics([ann(1, occ=2)], [pred()])
        lookup = {r['group']: r for r in groups if r['scope'] == 'vehicles'}
        self.assertEqual(lookup['small_heavy']['recall'], 1)
        self.assertIsNone(lookup['partial']['recall'])
        self.assertEqual(errors, [])
        self.assertEqual(len(images), 2)

    def test_matching_all_gt_before_occlusion_grouping(self):
        groups, _, _ = self.run_diagnostics([ann(1, x=0, occ=0), ann(2, x=3, occ=2)], [pred()])
        lookup = {r['group']: r for r in groups if r['scope'] == 'vehicles'}
        self.assertEqual(lookup['none']['tp'], 1)
        self.assertEqual(lookup['heavy']['tp'], 0)
        self.assertEqual(lookup['all']['recall'], .5)

    def test_duplicate_does_not_raise_recall(self):
        groups, errors, _ = self.run_diagnostics([ann(1)], [pred(), pred(score=.8)])
        self.assertEqual(groups[0]['tp'], 1)
        self.assertEqual([(e['kind'], e['reason']) for e in errors], [('FP', 'duplicate')])

    def test_wrong_class_is_fp_and_fn(self):
        _, errors, _ = self.run_diagnostics([ann(1)], [pred(category=9)])
        self.assertEqual([(e['kind'], e['reason']) for e in errors],
                         [('FN', 'class_confusion'), ('FP', 'class_confusion')])

    def test_low_confidence_excluded_only_from_diagnostics(self):
        gt = fixture([ann(1)])
        metrics, _ = coco_metrics(gt, [pred(score=.1)])
        self.assertAlmostEqual(metrics['AP'], 1)
        groups, _, _ = self.run_diagnostics([ann(1)], [pred(score=.1)])
        self.assertEqual(groups[0]['tp'], 0)

    def test_empty_predictions_are_zero_not_error(self):
        metrics, classes = coco_metrics(fixture([ann(1)]), [])
        self.assertEqual(metrics['AP'], 0)
        self.assertIsNone(metrics['APlarge'])
        self.assertIsNone(classes[1]['AP'])
        groups, errors, _ = self.run_diagnostics([ann(1)], [])
        self.assertEqual(groups[0]['recall'], 0)
        self.assertEqual(errors[0]['reason'], 'miss')

    def test_missing_occlusion_is_unknown(self):
        annotation = ann(1)
        del annotation['occlusion']
        groups, _, _ = self.run_diagnostics([annotation], [pred()])
        self.assertEqual(next(r['gt'] for r in groups if r['group'] == 'unknown'), 1)

    def test_coco_small_boundary_uses_original_gt_area(self):
        annotation = ann(1)
        annotation['area'] = 1024
        self.assertIn('small', group_names(annotation))

    def test_incomplete_coverage_and_bad_category_rejected(self):
        gt = fixture([ann(1)])
        with self.assertRaises(ValueError):
            validate_predictions(gt, [pred()], {'image_ids': [1]})
        with self.assertRaises(ValueError):
            validate_predictions(gt, [pred(category=0)], {'image_ids': [1, 2]})
        validate_predictions(gt, [], {'image_ids': [1, 2]})

    def test_explicit_label_mapping(self):
        categories = [dict(id=3, name='car'), dict(id=9, name='bus')]
        self.assertEqual(label_mapping(categories, 'contiguous0'), {0: 3, 1: 9})
        self.assertEqual(label_mapping(categories, 'category'), {3: 3, 9: 9})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'map.json'
            path.write_text(json.dumps({'0': None, '1': 3, '2': 9}))
            self.assertEqual(label_mapping(categories, None, path), {0: None, 1: 3, 2: 9})

    def test_p2_updates_all_levels_preserving_budget(self):
        base = {'RTDETR': dict(backbone='PResNet', encoder='HybridEncoder',
                              decoder='RTDETRTransformerv2'),
                'PResNet': dict(depth=18, return_idx=[1, 2, 3]),
                'HybridEncoder': dict(in_channels=[128, 256, 512],
                                      use_encoder_idx=[2], hidden_dim=256),
                'RTDETRTransformerv2': dict(num_layers=3, num_points=[2, 3, 4]),
                'epoches': 50}
        original = copy.deepcopy(base)
        p2 = add_p2(base)
        self.assertEqual(base, original)
        self.assertEqual(p2['PResNet']['return_idx'], [0, 1, 2, 3])
        self.assertEqual(p2['HybridEncoder']['use_encoder_idx'], [3])
        self.assertEqual(p2['RTDETRTransformerv2']['num_levels'], 4)
        self.assertEqual(p2['RTDETRTransformerv2']['num_points'], [2, 2, 3, 4])
        self.assertEqual(p2['RTDETRTransformerv2']['num_layers'], 3)
        self.assertEqual(p2['epoches'], 50)
        base['RTDETRTransformerv2']['num_levels'] = 4
        with self.assertRaises(ValueError):
            add_p2(base)


@unittest.skipUnless(os.environ.get('RTDETR_TEST_REPO'), 'Set RTDETR_TEST_REPO for official repo integration.')
class OfficialRepoIntegrationTests(unittest.TestCase):
    def test_export_evaluate_and_render(self):
        import torch
        import yaml
        from PIL import Image

        repo = Path(os.environ['RTDETR_TEST_REPO']).resolve()
        YAMLConfig = import_config(repo)
        config = YAMLConfig(str(repo / 'configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml')).yaml_cfg
        config.pop('__include__', None)
        config['num_classes'] = 11
        config['remap_mscoco_category'] = False
        config['PResNet']['pretrained'] = False
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
                     'tricycle', 'awning-tricycle', 'bus', 'motor']
            images = [dict(id=1, file_name='a.jpg', width=120, height=80),
                      dict(id=2, file_name='empty.jpg', width=120, height=80)]
            annotation = ann(1, category=4, occ=2)
            gt = dict(info={}, images=images,
                      categories=[dict(id=i+1, name=n) for i, n in enumerate(names)],
                      annotations=[annotation])
            gt_path = root / 'instances_val.json'
            gt_path.write_text(json.dumps(gt), encoding='utf-8')
            for image in images:
                Image.new('RGB', (120, 80), '#506050').save(root / image['file_name'])
            config['val_dataloader']['dataset'].update(img_folder=str(root), ann_file=str(gt_path))
            config_path = root / 'config.yml'
            config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
            checkpoint = os.environ.get('RTDETR_TEST_CHECKPOINT')
            if not checkpoint:
                checkpoint = str(root / 'checkpoint.pth')
                model = YAMLConfig(str(config_path)).model
                torch.save({'model': model.state_dict()}, checkpoint)
            map_path = root / 'map.json'
            map_path.write_text(json.dumps({i: i if i else None for i in range(11)}))
            scripts = Path(__file__).parent.resolve()
            export = root / 'export'
            result = subprocess.run([sys.executable, str(scripts / 'export_predictions.py'),
                '--repo', str(repo), '--config', str(config_path), '--checkpoint', checkpoint,
                '--annotations', str(gt_path), '--label-map', str(map_path),
                '--device', 'cpu', '--output', str(export)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            coverage = json.loads((export / 'coverage.json').read_text())
            self.assertEqual(coverage['image_ids'], [1, 2])
            self.assertEqual(coverage['export']['score_filter'], None)
            predictions = json.loads((export / 'predictions.json').read_text())
            self.assertTrue(predictions)
            self.assertTrue(all(1 <= p['category_id'] <= 10 for p in predictions))
            evaluation = root / 'evaluation'
            result = subprocess.run([sys.executable, str(scripts / 'evaluate.py'),
                '--annotations', str(gt_path), '--predictions', str(export / 'predictions.json'),
                '--coverage', str(export / 'coverage.json'), '--images', str(root),
                '--output', str(evaluation)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            metrics = json.loads((evaluation / 'metrics.json').read_text())
            self.assertEqual(metrics['protocol']['coverage_images'], 2)
            heavy = next(r for r in metrics['groups'] if r['scope'] == 'vehicles' and r['group'] == 'heavy')
            self.assertEqual(heavy['gt'], 1)
            self.assertTrue((evaluation / 'failures' / '000001.jpg').exists())
            self.assertTrue((evaluation / 'failure_images.csv').exists())


if __name__ == '__main__':
    unittest.main()
