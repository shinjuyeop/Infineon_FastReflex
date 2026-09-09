"""Label boundaries and latency gates must not turn early alarms into fast hits."""
import numpy as np
from fastreflex.dataset.stability import label_perturbation
from fastreflex.evaluation.stability import evaluate_latency


def test_labels_backdate_only_confirmed_episodes_and_censor_unresolved():
    tilt=np.zeros(60); speed=np.zeros(60)
    tilt[5:24]=2  # 19 samples, transient remains normal
    tilt[25:45]=2
    tilt[55:]=2  # unresolved end
    labels,episodes=label_perturbation(tilt,speed,[1,1],start=0)
    assert episodes==[(25,45)]
    assert np.all(labels[5:24]==0)
    assert np.all(labels[25:45]==1)
    assert np.all(labels[55:]==-1)
    labels,episodes=label_perturbation(tilt,speed,[1,1],start=0,censor=40)
    assert episodes==[] and np.all(labels[25:]==-1)


def test_latency_is_inclusive_and_early_persistent_alarm_is_not_a_hit():
    labels=np.zeros(100,dtype=np.int8);labels[20:90]=1
    p=np.zeros(100);p[46:]=1
    r=evaluate_latency(p,labels,[(20,90)],.5,persistence=5)
    assert r['episodes'][0]['latency_ms']==30
    assert r['detected_within_30ms']==1
    p[:]=1
    r=evaluate_latency(p,labels,[(20,90)],.5,persistence=5)
    assert r['detected_within_30ms']==0
    assert r['episodes'][0]['alert_active_before_onset']
    assert r['false_alert_events']==1


def test_missing_detection_is_not_zero_latency():
    labels=np.r_[np.zeros(20),np.ones(80)].astype(np.int8)
    r=evaluate_latency(np.zeros(100),labels,[(20,100)],.5)
    assert r['episodes'][0]['latency_ms'] is None
    assert r['detected_within_30ms']==0


def test_uniform_training_samples_cover_all_sensor_phases_reproducibly():
    from fastreflex.training.stability import _windows
    row=dict(id='normal_control',features=np.arange(8000,dtype=np.float32).reshape(100,80),
             labels=np.zeros(100,dtype=np.int8),episodes=[])
    args=([row],np.zeros(80),np.ones(80),20,3,60,'seeded_uniform_without_replacement',17)
    first=_windows(*args); second=_windows(*args)
    np.testing.assert_array_equal(first.endpoint_samples,second.endpoint_samples)
    assert len(set(first.endpoint_samples%10)) > 7
    for window,endpoint in zip(first.inputs,first.endpoint_samples):
        np.testing.assert_array_equal(window,row['features'][endpoint-19:endpoint+1])
