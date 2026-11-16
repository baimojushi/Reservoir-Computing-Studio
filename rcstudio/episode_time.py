"""Temporal contract for offline episode playback.

Every stored ESN row describes an input/state at episode-local step ``s`` and
an observed-input prediction for that same episode's target step ``s+h``.
Rows belonging to distinct episodes must never be joined by a path, interpolated
or reassigned to each other when predictions are drawn at their target times.
"""
from __future__ import annotations

import numpy as np


def episode_breaks(episode_id):
    """Break *before* the first displayed row of each subsequent episode."""
    ids = np.asarray(episode_id, dtype=np.int64).reshape(-1)
    result = np.zeros(len(ids), dtype=bool)
    result[1:] = ids[1:] != ids[:-1]
    return result


def align_to_target(source_values, episode_id, local_step, horizon, target_local_step=None):
    """Place a source-time y^(s+h) value on that episode's target-time row.

    Samples without a matching target row remain NaN: this is intentional near
    recording boundaries and under strided/downsampled windows. Never interpolate
    across an episode boundary to fill a gap. NaN washout values remain NaN.
    """
    values = np.asarray(source_values, dtype=np.float64).reshape(-1)
    episodes = np.asarray(episode_id, dtype=np.int64).reshape(-1)
    source = np.asarray(local_step, dtype=np.int64).reshape(-1)
    targets = (source + int(horizon) if target_local_step is None
               else np.asarray(target_local_step, dtype=np.int64).reshape(-1))
    if not (len(values) == len(episodes) == len(source) == len(targets)):
        raise ValueError("source, episode and local-step arrays must have the same length")
    if horizon <= 0:
        raise ValueError("forecast horizon must be positive")
    lookup = {(int(episode), int(step)): i for i, (episode, step)
              in enumerate(zip(episodes, source))}
    aligned = np.full(len(values), np.nan, dtype=np.float64)
    for i, (episode, target_step) in enumerate(zip(episodes, targets)):
        j = lookup.get((int(episode), int(target_step)))
        if j is not None and np.isfinite(values[i]):
            aligned[j] = values[i]
    return aligned
