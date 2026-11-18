"""Full DIOR checkpoint evaluation. Owner: Maha Muthupalaniappan (testing module)."""

import argparse
import csv
import hashlib
import json
import sys
import time
import pathlib
from pathlib import Path

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'DIOR_Data'))
from dataset import DIORDataset
from classes import CLASS_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='Smoke test only; 0 evaluates the full test split')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    # Trusted local training checkpoint; Python 3.13 renamed pathlib internals.
    sys.modules.setdefault('pathlib._local', pathlib)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert checkpoint['class_names'] == CLASS_NAMES, 'Checkpoint class ordering differs from dataset'
    config = checkpoint['config']
    backbone = checkpoint['backbone']
    size = config['image_size']
    options = {} if size == 800 else dict(min_size=size, max_size=size)
    factory = fasterrcnn_resnet50_fpn if backbone == 'resnet50' else fasterrcnn_mobilenet_v3_large_fpn
    model = factory(weights=None, weights_backbone=None, **options)
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, len(CLASS_NAMES) + 1)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    del checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    dataset = DIORDataset(args.data_root, 'test')
    for split in ('train', 'val'):
        assert not set(dataset.ids) & set(DIORDataset(args.data_root, split).ids), 'Test/train leakage'
    count = min(args.limit, len(dataset)) if args.limit else len(dataset)
    signature = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    manifest = dict(checkpoint=str(args.checkpoint.resolve()), sha256=signature,
                    epoch=10, backbone=backbone, config={k:str(v) if isinstance(v, Path) else v for k,v in config.items()},
                    test_images=count, full_test=count == len(dataset), device=str(device),
                    torch=torch.__version__, score_threshold=model.roi_heads.score_thresh,
                    detections_per_image=model.roi_heads.detections_per_img,
                    display_threshold=0.5, threads=args.threads)
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        assert previous == manifest, 'Output directory belongs to a different evaluation configuration'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    predictions_dir = args.output / 'predictions'
    predictions_dir.mkdir(exist_ok=True)
    metrics = MeanAveragePrecision(class_metrics=True)
    metric50 = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)
    timings = []
    quality = []
    started = time.perf_counter()
    with torch.inference_mode():
        for i in range(count):
            image, target = dataset[i]
            image_id = dataset.ids[i]
            path = predictions_dir / f'{image_id}.pt'
            if path.exists():
                record = torch.load(path, map_location='cpu', weights_only=True)
                prediction = record['prediction']
                seconds = record['seconds']
            else:
                image = image.to(device)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                prediction = model([image])[0]
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - t0
                prediction = {k:v.cpu() for k,v in prediction.items()}
                temporary = path.with_suffix('.tmp')
                torch.save(dict(prediction=prediction, seconds=seconds), temporary)
                temporary.replace(path)
            timings.append(dict(image_id=image_id, seconds=seconds))
            metrics.update([prediction], [target])
            metric50.update([prediction], [target])
            # At confidence 0.5, greedily match boxes of the same class at IoU 0.5.
            from torchvision.ops import box_iou
            keep = prediction['scores'] >= 0.5
            boxes, labels = prediction['boxes'][keep], prediction['labels'][keep]
            matched = set()
            tp = 0
            overlaps = box_iou(boxes, target['boxes'])
            for j, label in enumerate(labels):
                candidates = [(float(overlaps[j,k]), k) for k in range(len(target['labels']))
                              if k not in matched and target['labels'][k] == label]
                if candidates:
                    value, k = max(candidates)
                    if value >= 0.5:
                        tp += 1
                        matched.add(k)
            fp, fn = len(boxes)-tp, len(target['boxes'])-tp
            f1 = 2*tp/max(2*tp+fp+fn, 1)
            quality.append(dict(index=i, image_id=image_id, tp=tp, fp=fp, fn=fn, f1=f1))
            if (i+1) % 25 == 0 or i+1 == count:
                elapsed = time.perf_counter()-started
                print(f'Images {i+1}/{count}; elapsed {elapsed/60:.1f} min; last inference {seconds:.3f}s', flush=True)
                (args.output / 'progress.json').write_text(json.dumps(dict(completed=i+1,total=count,elapsed_seconds=elapsed)))
    print('Computing aggregate AP...', flush=True)
    result, result50 = metrics.compute(), metric50.compute()
    class_rows = []
    ids = result['classes'].tolist()
    for label, name in enumerate(CLASS_NAMES, 1):
        j = ids.index(label) if label in ids else None
        class_rows.append(dict(class_id=label, class_name=name,
                               ap50=float(result50['map_per_class'][j]) if j is not None else None,
                               ap50_95=float(result['map_per_class'][j]) if j is not None else None))
    summary = {k:float(v) for k,v in result.items() if v.numel()==1}
    summary.update(ap50=float(result50['map']), test_images=count,
                   inference_seconds=sum(x['seconds'] for x in timings),
                   images_per_second=count/sum(x['seconds'] for x in timings),
                   speed_note='Model forward only; batch size 1; includes first-image warmup; excludes loading and metrics',
                   per_class=class_rows, manifest=manifest)
    (args.output / 'results.json').write_text(json.dumps(summary, indent=2))
    for name, rows in [('per_class.csv', class_rows), ('timings.csv', timings), ('image_quality.csv', quality)]:
        with (args.output / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11,7))
    ax.barh(CLASS_NAMES, [x['ap50'] or 0 for x in class_rows])
    ax.set(xlabel='AP at IoU 0.5', title='DIOR test set: AP per class', xlim=(0,1))
    fig.tight_layout()
    fig.savefig(args.output / 'per_class_ap50.png', dpi=180)
    plt.close(fig)
    from torchvision.utils import draw_bounding_boxes
    from torchvision.transforms.functional import to_pil_image
    examples = args.output / 'examples'
    examples.mkdir(exist_ok=True)
    eligible = [q for q in quality if q['tp']+q['fn'] > 0]
    for kind, selected in [('success', sorted(eligible, key=lambda q:-q['f1'])[:5]),
                           ('failure', sorted(eligible, key=lambda q:q['f1'])[:5])]:
        for q in selected:
            image, target = dataset[q['index']]
            record = torch.load(predictions_dir / f"{q['image_id']}.pt", weights_only=True)
            pred = record['prediction']
            keep = pred['scores'] >= 0.5
            labels = [f'{CLASS_NAMES[int(c)-1]} {float(s):.2f}' for c,s in zip(pred['labels'][keep],pred['scores'][keep])]
            pixels = (image*255).to(torch.uint8)
            drawn = draw_bounding_boxes(pixels, pred['boxes'][keep], labels=labels, colors='red', width=2)
            to_pil_image(drawn).save(examples / f"{kind}_{q['image_id']}_predictions.png")
            drawn = draw_bounding_boxes(pixels, target['boxes'], labels=[CLASS_NAMES[int(c)-1] for c in target['labels']], colors='green', width=2)
            to_pil_image(drawn).save(examples / f"{kind}_{q['image_id']}_ground_truth.png")
    print(json.dumps({k:v for k,v in summary.items() if k not in ('manifest','per_class')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
