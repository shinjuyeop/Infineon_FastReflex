"""Strict episode-level latency and false-alert accounting for stability pilots."""
import numpy as np


def evaluate_latency(probability, labels, episodes, threshold, persistence=5, deadline=30):
    probability, labels = np.asarray(probability), np.asarray(labels)
    if probability.shape != labels.shape:
        raise ValueError('unaligned decisions and labels')
    count = 0
    alerts = np.zeros(len(labels), dtype=bool)
    for i, p in enumerate(probability):
        count = count + 1 if labels[i] >= 0 and np.isfinite(p) and p >= threshold else 0
        alerts[i] = count >= persistence
    rising = alerts & ~np.r_[False, alerts[:-1]]
    # An alarm already active before onset is not credited as new detection.
    rows = []
    for begin, end in episodes:
        candidates = np.flatnonzero(rising[begin:end]) + begin
        latency = None if not len(candidates) else int(candidates[0] - begin)
        rows.append(dict(onset=int(begin), end=int(end), latency_ms=latency,
                         within_30ms=latency is not None and latency <= deadline,
                         alert_active_before_onset=bool(begin and alerts[begin-1])))
    negative = labels == 0
    # Count alarm onsets in negative samples; retain lingering false duration separately.
    false_trace = alerts & negative
    false_events = int(np.sum(rising & negative))
    return dict(episodes=rows, eligible_samples=int(np.sum(labels >= 0)),
                negative_samples=int(np.sum(negative)), false_alert_events=false_events,
                false_alert_samples=int(np.sum(false_trace)),
                detected_within_30ms=sum(r['within_30ms'] for r in rows),
                episode_count=len(rows))


def verify_stability_pilot(root, config_path):
    """Verify saved pilot identities without retraining or reopening source runs."""
    import json
    import yaml
    from fastreflex.dataset.loader import sha256_file
    document=yaml.safe_load(config_path.read_text())
    output=root/document['output']
    manifest=json.loads((output/'dataset_manifest.json').read_text())
    freeze=json.loads((output/'candidate_freeze.json').read_text())
    result=json.loads((output/'results.json').read_text())
    checks={config_path:manifest['config_sha256'],
        output/'dataset_manifest.json':freeze['dataset_manifest_sha256'],
        output/'normalizer.json':freeze['normalizer_sha256'],
        output/'models'/f"{freeze['family']}.pt":freeze['model_sha256']}
    checks.update({output/'data'/row['derived_file']:row['sha256'] for row in manifest['runs']})
    if result['freeze']!=freeze or any(sha256_file(p)!=expected for p,expected in checks.items()):
        raise ValueError('pilot frozen artifact identity mismatch')
    return result
