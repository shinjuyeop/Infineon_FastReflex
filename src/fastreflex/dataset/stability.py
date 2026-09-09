"""Derived body-perturbation annotations; frozen Hazard labels are never edited."""
from __future__ import annotations
import numpy as np


def label_perturbation(tilt, angular_speed, limits, persistence=20, start=1000, censor=None):
    """Offline confirm exceedance episodes and backdate onset, excluding censor."""
    tilt, angular_speed = np.asarray(tilt), np.asarray(angular_speed)
    if tilt.shape != angular_speed.shape or tilt.ndim != 1:
        raise ValueError('unaligned physical signals')
    if persistence <= 0 or start < 0 or not np.all(np.isfinite(limits)):
        raise ValueError('invalid annotation contract')
    stop = len(tilt) if censor is None else min(int(censor), len(tilt))
    if not np.all(np.isfinite(tilt[:stop])) or not np.all(np.isfinite(angular_speed[:stop])):
        raise ValueError('invalid physical trace')
    labels = np.full(len(tilt), -1, dtype=np.int8)
    labels[start:stop] = 0
    outside = (tilt > limits[0]) | (angular_speed > limits[1])
    episodes = []
    cursor = start
    while cursor < stop:
        if not outside[cursor]:
            cursor += 1
            continue
        begin = cursor
        while cursor < stop and outside[cursor]:
            cursor += 1
        if cursor - begin >= persistence:
            labels[begin:cursor] = 1
            episodes.append((begin, cursor))
        elif cursor == stop:
            labels[begin:cursor] = -1  # unresolved at censor/end, never negative
    return labels, episodes
