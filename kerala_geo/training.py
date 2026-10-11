"""Frozen image features + supervised binary head; no GPS fine-tuning."""
import csv
import json
import math
from pathlib import Path

import numpy as np

from .core import ROOT, image_digest, threshold_labels, haversine_km


def read_dataset(manifest):
    manifest = Path(manifest).resolve()
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'path', 'label', 'group', 'split'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('CSV must contain path,label,group,split columns.')
        rows = list(reader)
    if not rows:
        raise ValueError('Dataset is empty.')
    groups, hashes, files = {}, {}, set()
    result = []
    for line, row in enumerate(rows, 2):
        label = row['label'].strip().lower()
        if label not in ('kerala', 'non_kerala'):
            raise ValueError(f'Row {line}: label must be kerala or non_kerala.')
        split, group = row['split'].strip().lower(), row['group'].strip()
        if split not in ('train', 'val', 'test') or not group:
            raise ValueError(f'Row {line}: supply a group and train/val/test split.')
        path = (manifest.parent / row['path']).resolve()
        if not path.is_file():
            raise ValueError(f'Row {line}: image does not exist: {path}')
        if path in files:
            raise ValueError(f'Row {line}: repeated image path.')
        files.add(path)
        if group in groups and groups[group] != split:
            raise ValueError(f'Group {group!r} occurs in multiple splits. Keep each location/session together.')
        groups[group] = split
        digest = image_digest(path)
        if digest in hashes:
            raise ValueError(f'Row {line}: duplicate image pixels (also row {hashes[digest]}).')
        hashes[digest] = line
        entry = {'path': str(path), 'label': int(label == 'kerala'), 'group': group, 'split': split, 'sha256': digest}
        if row.get('latitude') or row.get('longitude'):
            try:
                lat, lon = float(row['latitude']), float(row['longitude'])
                if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
                    raise ValueError()
            except (ValueError,TypeError,KeyError) as exc:
                raise ValueError(f'Row {line}: invalid GPS ground truth.') from exc
            entry.update(latitude=lat,longitude=lon)
        result.append(entry)
    located = [r for r in result if 'latitude' in r]
    for i, a in enumerate(located):
        for b in located[i+1:]:
            if a['split'] != b['split'] and float(haversine_km(a['latitude'],a['longitude'],b['latitude'],b['longitude'])) < 1:
                raise ValueError('Locations within 1 km occur in different splits. Increase geographic separation.')
    return result


def validate_training_counts(rows):
    for split, minimum in [('train', 20), ('val', 10), ('test', 10)]:
        for label in (0, 1):
            selected = [r for r in rows if r['split'] == split and r['label'] == label]
            if len(selected) < minimum or len({r['group'] for r in selected}) < 2:
                raise ValueError(f'{split}, class {label}: need at least {minimum} images from 2 distinct groups. These are software minimums, not sufficient evidence of general accuracy.')


def choose_thresholds(y, scores, target_precision=0.9, minimum_support=5):
    y, scores = np.asarray(y), np.asarray(scores)
    if not 0.5 < target_precision <= 1:
        raise ValueError('Target precision must be greater than 0.5 and at most 1.')
    low, high = 0.0, 1.0
    for threshold in sorted(set(scores[scores > 0.5])):
        selected = scores >= threshold
        if selected.sum() >= minimum_support and (y[selected] == 1).mean() >= target_precision:
            high = float(threshold)
            break
    for threshold in sorted(set(scores[scores < 0.5]), reverse=True):
        selected = scores <= threshold
        if selected.sum() >= minimum_support and (y[selected] == 0).mean() >= target_precision:
            low = float(threshold)
            break
    return low, high


def metrics(y, predictions):
    y, p = np.asarray(y), np.asarray(predictions)
    if len(y) == 0 or y.shape != p.shape:
        raise ValueError('Metrics need equally sized, nonempty labels and predictions.')
    accepted = p != -1
    result = {'images': len(y), 'accepted': int(accepted.sum()), 'uncertain': int((~accepted).sum()),
              'coverage': float(accepted.mean()),
              'accuracy_counting_uncertain_as_incorrect': float((p == y).mean()),
              'accuracy_on_accepted': float((p[accepted] == y[accepted]).mean()) if accepted.any() else None,
              'classes': {}}
    recalls = []
    for label, name in [(0, 'non_kerala'), (1, 'kerala')]:
        tp = int(((p == label) & (y == label)).sum())
        predicted, actual = int((p == label).sum()), int((y == label).sum())
        recall = tp / actual if actual else None
        if recall is not None:
            recalls.append(recall)
        result['classes'][name] = {'precision': tp / predicted if predicted else None,
                                  'recall_including_abstentions': recall, 'support': actual,
                                  'predicted': predicted, 'correct': tp}
    result['balanced_accuracy_including_abstentions'] = float(np.mean(recalls))
    return result


def train(manifest, progress=print):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, log_loss
    from .model import Predictor
    rows = read_dataset(manifest)
    validate_training_counts(rows)
    predictor = Predictor(progress=progress, use_classifier=False)
    features, geo_features = [], []
    feature_cache = ROOT / 'cache' / ('features-' + predictor.fingerprint[:16])
    feature_cache.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        progress(f'Extracting image features {i+1}/{len(rows)}')
        cached = feature_cache / (row['sha256'] + '.npz')
        if cached.exists():
            with np.load(cached, allow_pickle=False) as saved:
                raw, geo = saved['clip'], saved['geo']
        else:
            raw, tensor = predictor.features(row['path'])
            geo = tensor.cpu().numpy()[0]
            np.savez(cached, clip=raw, geo=geo)
        features.append(raw)
        geo_features.append(geo)
    feature_sets = {'clip': np.asarray(features), 'geoclip': np.asarray(geo_features)}
    feature_sets['combined'] = np.concatenate([feature_sets['clip'], feature_sets['geoclip']], axis=1)/np.sqrt(2)
    y = np.array([r['label'] for r in rows])
    splits = np.array([r['split'] for r in rows])
    val = splits == 'val'
    candidates, best = [], None
    for feature_type, X in feature_sets.items():
        for strength in (0.1, 1.0, 10.0, 100.0):
            head = LogisticRegression(C=strength, class_weight='balanced', max_iter=2000, random_state=42)
            head.fit(X[splits == 'train'], y[splits == 'train'])
            if int(head.n_iter_[0]) >= head.max_iter:
                raise RuntimeError('Classifier did not converge. No model saved.')
            validation_scores = head.predict_proba(X[val])[:, 1]
            balanced = float(balanced_accuracy_score(y[val], validation_scores >= .5))
            loss = float(log_loss(y[val], validation_scores, labels=[0,1]))
            candidates.append({'features':feature_type,'C':strength,'validation_balanced_accuracy':balanced,'validation_log_loss':loss})
            key = (balanced, -loss)
            if best is None or key > best[0]:
                best = (key, head, feature_type, strength)
    _, head, feature_type, strength = best
    scores = head.predict_proba(feature_sets[feature_type])[:, 1]
    low, high = choose_thresholds(y[val], scores[val])
    predictions = threshold_labels(scores, low, high)
    test = splits == 'test'
    report = {
        'method': 'Frozen visual features + logistic regression; GeoCLIP coordinate model unchanged',
        'selected_features': feature_type, 'selected_C': strength,
        'model_selection': '12 candidates selected using validation balanced accuracy, breaking ties by log loss. Test labels never used for selection.',
        'validation_candidates': candidates,
        'accuracy_claim': 'Measured only on this supplied dataset. No guarantee on new locations or interiors.',
        'threshold_selection': 'Validation only; target empirical precision 0.90 per class, at least 5 accepted examples per class. This is not calibrated confidence.',
        'low': low, 'high': high,
        'validation': metrics(y[val], predictions[val]),
        'held_out_test': metrics(y[test], predictions[test]),
        'held_out_test_forced_binary': metrics(y[test], (scores[test] >= .5).astype(int)),
        'model_fingerprint': predictor.fingerprint,
        'dataset': [{k: r[k] for k in ('sha256', 'label', 'group', 'split')} for r in rows],
    }
    # Compare against GeoCLIP on exactly the same held-out images, without new feature extraction.
    import torch
    predictor.prepare_gallery()
    with torch.inference_mode():
        test_geo = torch.from_numpy(feature_sets['geoclip'][test]).to(predictor.device)
        indices = (test_geo @ predictor.locations.T).argmax(dim=1).cpu().numpy()
    baseline = [int(predictor.boundary.contains(*map(float,predictor.gps[i]))) for i in indices]
    report['held_out_geoclip_top1_boundary_baseline'] = metrics(y[test], baseline)
    test_rows = [r for r in rows if r['split'] == 'test']
    report['held_out_by_location'] = {}
    for group in sorted({r['group'] for r in test_rows}):
        selected = np.array([r['group'] == group for r in test_rows])
        report['held_out_by_location'][group] = metrics(y[test][selected],predictions[test][selected])
    distances = [float(haversine_km(r['latitude'],r['longitude'],*predictor.gps[i])) for r,i in zip(test_rows,indices) if 'latitude' in r]
    if distances:
        report['held_out_geoclip_coordinate_error'] = {'images_with_coordinates':len(distances),'median_km':float(np.median(distances)),
                                                      'within_25km':float((np.asarray(distances)<=25).mean())}
    payload = {'schema_version': 1, 'model_fingerprint': predictor.fingerprint, 'low': low, 'high': high, 'feature_type': feature_type,
               'coef': head.coef_[0].tolist(), 'intercept': float(head.intercept_[0])}
    destination = ROOT / 'trained'
    destination.mkdir(exist_ok=True)
    with (destination/'test-predictions.csv').open('w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path','group','actual','classifier_score','prediction','geoclip_top1_prediction'])
        for row,score,prediction,base in zip(test_rows,scores[test],predictions[test],baseline):
            writer.writerow([row['path'],row['group'],row['label'],float(score),int(prediction),base])
    for name, data in [('report.json', report), ('classifier.json', payload)]:
        path = destination / name
        if path.exists():
            (destination / (name + '.previous')).write_bytes(path.read_bytes())
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
        tmp.replace(path)
    progress('Training complete. Read trained/report.json before relying on the model.')
    return report


def evaluate(manifest, progress=print):
    from .model import Predictor
    rows = [r for r in read_dataset(manifest) if r['split'] == 'test']
    if not rows:
        raise ValueError('Evaluation requires rows whose split is test.')
    predictor = Predictor(progress=progress)
    # Prevent reporting familiar training/validation samples as new test evidence.
    report_path = ROOT / 'trained/report.json'
    if predictor.classifier is not None and report_path.exists():
        previous = json.loads(report_path.read_text(encoding='utf-8'))
        seen = [r for r in previous['dataset'] if r['split'] != 'test']
        hashes, groups = {r['sha256'] for r in seen}, {r['group'] for r in seen}
        if any(r['sha256'] in hashes or r['group'] in groups for r in rows):
            raise ValueError('Evaluation overlaps saved training/validation images or groups.')
    results = []
    for i, row in enumerate(rows):
        progress(f'Evaluating {i+1}/{len(rows)}')
        result = predictor.predict(row['path'])
        results.append({'path': row['path'], 'actual': row['label'], **result})
    return {'metrics': metrics([r['label'] for r in rows], [r['decision'] for r in results]),
            'model_fingerprint': predictor.fingerprint, 'results': results}
