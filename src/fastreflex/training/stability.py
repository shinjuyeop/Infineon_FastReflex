"""Small, run-disjoint stability relabel/train/evaluate pilot orchestration."""
from __future__ import annotations
import json
import hashlib
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import torch
import yaml

from fastreflex.dataset.loader import WindowSet, sha256_file
from fastreflex.dataset.stability import label_perturbation
from fastreflex.evaluation.stability import evaluate_latency
from fastreflex.features import extract_hazard_features
from fastreflex.training.trainer import train_model, save_checkpoint


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def _windows(records, mean, std, width, stride, dense, sampling="regular", seed=17):
    inputs, targets, ids, endpoints = [], [], [], []
    for row in records:
        features = (row['features']-mean)/std
        if sampling == "seeded_uniform_without_replacement":
            candidates = np.flatnonzero((row['labels'] >= 0) & (np.arange(len(features)) >= width-1))
            run_seed = int(hashlib.sha256(row['id'].encode()).hexdigest()[:8],16) + seed
            rng = np.random.default_rng(run_seed)
            selected = set(rng.choice(candidates, size=max(1,len(candidates)//stride),replace=False).tolist())
        elif sampling == "regular":
            selected = set(range(width-1, len(features), stride))
        else:
            raise ValueError('unknown endpoint sampling')
        for begin, end in row['episodes']:
            selected.update(range(max(width-1, begin-width), min(len(features), begin+dense)))
        for endpoint in sorted(selected):
            target = int(row['labels'][endpoint])
            if target < 0:
                continue
            inputs.append(features[endpoint-width+1:endpoint+1])
            targets.append(target); ids.append(row['id']); endpoints.append(endpoint)
    y=np.asarray(targets, dtype=np.int64)
    return WindowSet(np.asarray(inputs,dtype=np.float32),y,np.asarray(ids),
                     np.asarray(endpoints),tuple(np.bincount(y,minlength=2)))


def _probabilities(model, row, mean, std, width):
    features = ((row['features']-mean)/std).astype(np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(features,width,axis=0).transpose(0,2,1)
    output = np.full(len(features), np.nan)
    model.eval()
    with torch.no_grad():
        for start in range(0,len(windows),256):
            batch=torch.from_numpy(windows[start:start+256].copy())
            output[width-1+start:width-1+start+len(batch)] = torch.softmax(model(batch),dim=1)[:,1].numpy()
    return output


def _score(records, probabilities, threshold, document):
    results={r['id']:evaluate_latency(probabilities[r['id']],r['labels'],r['episodes'],
        threshold, document['operating_point']['persistence_samples'],
        document['acceptance']['deadline_ms']) for r in records}
    episodes=sum(r['episode_count'] for r in results.values())
    hits=sum(r['detected_within_30ms'] for r in results.values())
    false=sum(r['false_alert_events'] for r in results.values())
    normal_ms=sum(r['negative_samples'] for r in results.values())
    normal_false=sum(results[r['id']]['false_alert_events'] for r in records if not r['episodes'])
    recall=hits/episodes if episodes else None
    false_rate=false/(normal_ms/60000) if normal_ms else None
    gates=document['acceptance']
    passed=(recall is not None and recall>=gates['episode_recall_within_deadline'] and
            false_rate is not None and false_rate<=gates['false_alerts_per_minute_max'] and
            normal_false<=gates['normal_run_false_alerts_max'])
    return dict(episode_count=episodes,detected_within_30ms=hits,recall_30ms=recall,
        false_alert_events=false,false_alerts_per_minute=false_rate,
        no_episode_run_false_alerts=normal_false, passed=passed, runs=results)


def run_stability_pilot(root: Path, config_path: Path):
    from fastreflex.visualization import resolve_visualization_run, reconstruct_simulation_config, compare_stored_runtime, require_parity
    from fastreflex.simulation.g1 import run_simulation
    document=yaml.safe_load(config_path.read_text())
    output=root/document['output']
    output.mkdir(parents=True,exist_ok=False)
    (output/'data').mkdir()
    (output/'models').mkdir()
    _write(output/'experiment.json',document)
    train_settings=document['training'];split=document['split']
    records=[]
    reuse_root = root/document['reuse_physical_from'] if document.get('reuse_physical_from') else None
    reuse_manifest = json.loads((reuse_root/'dataset_manifest.json').read_text()) if reuse_root else None
    reuse_rows = {r['run_id']:r for r in reuse_manifest['runs']} if reuse_manifest else {}
    for role in ('train','development','pilot_test'):
        for group in split['groups']:
            for suffix in split[f'{role}_suffixes']:
                run_id=f'uhr_{group}_{suffix}'
                resolved=resolve_visualization_run(root,run_id)
                config=reconstruct_simulation_config(resolved)
                if run_id in reuse_rows:
                    reused = reuse_rows[run_id]
                    path = reuse_root/'data'/reused['derived_file']
                    if sha256_file(path) != reused['sha256'] or reused['source']['file_sha256'] != resolved.manifest_row['file_sha256']:
                        raise ValueError('reused physical provenance mismatch')
                    with np.load(path,allow_pickle=False) as data:
                        imu, tilt, angular = data['imu'].copy(),data['tilt'].copy(),data['angular_speed'].copy()
                else:
                    result=run_simulation(config)
                    require_parity(compare_stored_runtime(resolved,result))
                    imu=result.runtime.pelvis_imu
                    tilt=result.diagnostics.pelvis_tilt_rad
                    angular=np.linalg.norm(result.diagnostics.pelvis_angular_velocity_rad_s,axis=1)
                row=dict(id=run_id,role=role,group=group,imu=imu,
                    features=extract_hazard_features(imu),tilt=tilt,angular_speed=angular,
                    censor=resolved.run.censor_sample,
                    source=dict(dataset_id=resolved.manifest['dataset_id'],
                                **dict(resolved.manifest_row)))
                records.append(row)
                print(f'Physical replay {len(records)}: {run_id} ({role})',flush=True)
    ids=[r['id'] for r in records]
    if len(ids)!=len(set(ids)):
        raise ValueError('run overlap in pilot roles')
    normal=[r for r in records if r['role']=='train' and r['group']=='hard_n']
    start=document['label']['baseline_start_sample']
    if any(r['censor'] < len(r['imu']) for r in normal):
        raise ValueError('normal calibration contains censored run')
    limits=[max(float(np.max(r[key][start:])) for r in normal) for key in ('tilt','angular_speed')]
    manifest=dict(dataset_id=document.get('dataset_id','walking_stability_latency_pilot_20260908'),schema_version=1,
        created_at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        source_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)),
        code_sha256={str(p.relative_to(root)):sha256_file(p) for p in
            [Path(__file__),root/'src/fastreflex/dataset/stability.py',root/'src/fastreflex/evaluation/stability.py',root/'src/fastreflex/simulation/g1.py']},
        config_sha256=sha256_file(config_path),limits=limits,
        calibration_run_ids=[r['id'] for r in normal],label_role=document['label']['role'],runs=[])
    for row in records:
        labels,episodes=label_perturbation(row['tilt'],row['angular_speed'],limits,
            document['label']['persistence_samples'],start,row['censor'])
        row.update(labels=labels,episodes=episodes)
        path=output/'data'/f"{row['id']}.npz"
        np.savez_compressed(path,imu=row['imu'],tilt=row['tilt'],angular_speed=row['angular_speed'],
                            labels=labels,episodes=np.asarray(episodes,dtype=np.int64).reshape(-1,2))
        manifest['runs'].append(dict(run_id=row['id'],role=row['role'],source=row['source'],
                                    derived_file=path.name,sha256=sha256_file(path)))
    _write(output/'dataset_manifest.json',manifest)
    train=[r for r in records if r['role']=='train']
    dev=[r for r in records if r['role']=='development']
    test=[r for r in records if r['role']=='pilot_test']
    values=np.concatenate([r['features'][r['labels']>=0] for r in train])
    mean=values.mean(axis=0,dtype=np.float64).astype(np.float32)
    std=values.std(axis=0,dtype=np.float64).astype(np.float32)
    std=np.maximum(std,1e-6)
    _write(output/'normalizer.json',dict(mean=mean.tolist(),std=std.tolist(),source='train_only'))
    width=train_settings['window_samples']
    train_windows=_windows(train,mean,std,width,train_settings['stride_samples'],train_settings['onset_dense_samples'],
        train_settings.get('endpoint_sampling','regular'),train_settings['seed'])
    dev_windows=_windows(dev,mean,std,width,train_settings['stride_samples'],train_settings['onset_dense_samples'],
        train_settings.get('endpoint_sampling','regular'),train_settings['seed'])
    candidates=[]
    selected=None
    for family in train_settings['families']:
        print(f'Training {family}: {len(train_windows)} windows',flush=True)
        model,result=train_model(family,width,train_windows,dev_windows,train_settings['seed'],
            batch_size=train_settings['batch_size'],max_epochs=train_settings['max_epochs'],
            patience=train_settings['patience'],learning_rate=train_settings['learning_rate'],
            class_names=('NORMAL','PERTURBED'),selection_metric=train_settings['selection_metric'])
        path=output/'models'/f'{family}.pt'
        save_checkpoint(path,model,family,width,train_settings['seed'],result,
                        input_channels=80,class_names=('NORMAL','PERTURBED'))
        probabilities={r['id']:_probabilities(model,r,mean,std,width) for r in dev}
        for threshold in document['operating_point']['thresholds']:
            score=_score(dev,probabilities,threshold,document)
            rate=score['false_alerts_per_minute']
            # Prefer passing false-alarm gates, then deadline recall, then fewer false alarms.
            key=(rate is not None and rate<=document['acceptance']['false_alerts_per_minute_max']
                 and score['no_episode_run_false_alerts']==0,
                 score['recall_30ms'] or 0,-(rate if rate is not None else float('inf')))
            entry=dict(family=family,threshold=threshold,training=asdict(result),development=score)
            candidates.append(entry)
            if selected is None or key>selected[0]:
                selected=(key,model,entry)
        print(f'Completed {family}',flush=True)
    _,model,chosen=selected
    freeze=dict(family=chosen['family'],threshold=chosen['threshold'],
        model_sha256=sha256_file(output/'models'/f"{chosen['family']}.pt"),
        normalizer_sha256=sha256_file(output/'normalizer.json'),
        dataset_manifest_sha256=sha256_file(output/'dataset_manifest.json'),
        before_pilot_test_model_inference=True)
    _write(output/'candidate_freeze.json',freeze)
    _write(output/'development.json',candidates)
    probabilities={r['id']:_probabilities(model,r,mean,std,width) for r in test}
    result=_score(test,probabilities,chosen['threshold'],document)
    summary=dict(verdict='PILOT_30MS_PASS' if result['passed'] else 'PILOT_30MS_NOT_ACHIEVED',
        role='historical_run_disjoint_pilot_not_fresh_generalization_or_e84_validation',
        label_role=document['label']['role'],limits=limits,freeze=freeze,
        development=chosen['development'],pilot_test=result)
    _write(output/'results.json',summary)
    for row in test:
        np.savez_compressed(output/'data'/f"{row['id']}_predictions.npz",probability=probabilities[row['id']])
    print(json.dumps({k:v for k,v in summary.items() if k not in ('development','pilot_test')},indent=2))
    return summary
