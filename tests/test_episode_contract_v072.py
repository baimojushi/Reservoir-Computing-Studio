"""V0.7.2: real EBrush episode metadata and source-to-target time alignment.

These tests deliberately use the production train_motion path, not a hand-drawn
UI fixture. Node tests execute the same prepareRun implementation as the Web UI.
"""
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rcstudio.core import reservoir_frames
from rcstudio.ebrush import make_demo, train_motion, normalize_ebrush_config
from rcstudio.episode_time import align_to_target, episode_breaks
from rcstudio.lab_web import page_path, run_payload
from rcstudio.sketch_math import interpolate_stroke
from rcstudio.storage import Workspace


class TestEpisodeAlignment(unittest.TestCase):
    def test_signed_readout_target_keeps_episode_identity(self):
        values = [2., 3., 4., 5., 11., np.nan, 13., 14.]
        episodes = [0, 0, 0, 0, 1, 1, 1, 1]
        local = [0, 1, 2, 3, 0, 1, 2, 3]
        result = align_to_target(values, episodes, local, horizon=2)
        np.testing.assert_allclose(result[:4], [np.nan, np.nan, 2, 3], equal_nan=True)
        np.testing.assert_allclose(result[4:], [np.nan, np.nan, 11, np.nan], equal_nan=True)
        self.assertFalse(np.isfinite(result[4]))   # prior episode's last two predictions don't cross
        np.testing.assert_array_equal(episode_breaks(episodes), [0, 0, 0, 0, 1, 0, 0, 0])

    def test_strided_window_missing_target_is_missing_not_shifted(self):
        result = align_to_target([5, 7, 9, 11, 13], [0]*5, [0, 3, 6, 9, 12], horizon=4)
        self.assertTrue(np.isnan(result).all())
        result = align_to_target([5, 7, 9, 11], [0]*4, [0, 3, 6, 9], horizon=6)
        np.testing.assert_allclose(result, [np.nan, np.nan, 5, 7], equal_nan=True)

    def test_reverse_mouse_stroke_preserves_actual_endpoints(self):
        forward = np.zeros(11)
        interpolate_stroke(forward, 2, -0.3, 8, 0.9)
        backward = np.zeros(11)
        interpolate_stroke(backward, 8, 0.9, 2, -0.3)
        np.testing.assert_allclose(forward, backward)
        self.assertAlmostEqual(backward[2], -0.3)
        self.assertAlmostEqual(backward[8], 0.9)


class TestTrainedEbrushEpisodes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ws = Workspace(cls.root / 'workspace')
        info = make_demo(cls.ws, count=5, length=260)
        cfg = normalize_ebrush_config({
            'dataset': {'session_ids': info['session_ids']},
            'model': {'units': 18, 'washout': 16, 'horizon': 10, 'seed': 9},
        })
        cls.run_dir = cls.root / 'run'
        cls.summary = train_motion(cfg, cls.ws, cls.run_dir)
        with np.load(cls.run_dir / 'traces.npz', allow_pickle=False) as z:
            cls.full_ids = z['episode_id'].copy()
            cls.full_local = z['local_step'].copy()
            cls.full_prediction = z['prediction'].copy()
            cls.full_recordings = z['episode_recording_ids'].copy()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_full_trace_episode_starts_and_prediction_masks(self):
        ids = self.full_ids
        self.assertEqual(len(set(ids)), 5)
        starts = np.flatnonzero(episode_breaks(ids))
        self.assertEqual(starts.tolist(), [250, 500, 750, 1000])
        self.assertEqual(len(self.full_recordings), 5)
        for ep in range(5):
            where = np.flatnonzero(ids == ep)
            self.assertEqual(self.full_local[where].tolist(), list(range(250)))
            self.assertTrue(np.isnan(self.full_prediction[where[:16]]).all())
            self.assertTrue(np.isfinite(self.full_prediction[where[16:]]).all())

    def test_http_contract_exposes_metadata_inside_window(self):
        frames = reservoir_frames(self.run_dir, start=225, count=80)
        self.assertTrue(frames['has_episode_metadata'])
        self.assertEqual(frames['count'], 80)
        self.assertEqual(frames['episode_id'][24:27], [0, 1, 1])
        self.assertEqual(frames['episode_start'][25], True)
        self.assertEqual(frames['local_step'][24:27], [249, 0, 1])
        self.assertEqual(frames['target_local_step'][25], 10)
        self.assertNotEqual(frames['recording_id'][24], frames['recording_id'][25])
        self.assertEqual(len(frames['states'][0]), 18)
        self.assertIsNone(frames['prediction'][25])  # per-episode Washout

    def test_old_ebrush_files_fail_closed(self):
        # Copy an older NPZ schema without inventing episode boundaries.
        old = self.root / 'legacy'
        old.mkdir(exist_ok=True)
        with np.load(self.run_dir / 'traces.npz', allow_pickle=False) as z:
            names = ('timestep', 'inputs', 'truth', 'prediction', 'split', 'states')
            np.savez_compressed(old / 'traces.npz', **{k: z[k] for k in names})
        frames = reservoir_frames(old, start=0, count=80)
        self.assertFalse(frames['has_episode_metadata'])
        self.assertNotIn('episode_id', frames)

    @unittest.skipUnless(shutil.which('node'), 'Node.js required to evaluate actual Web engine')
    def test_web_engine_does_not_move_forecast_into_next_recording(self):
        page = page_path().read_text(encoding='utf-8')
        engine = re.search(r'/\*ENGINE-BEGIN\*/(.*?)/\*ENGINE-END\*/', page, re.S).group(1)
        # 2 episodes in the same 350-frame window, preserving real train_motion values
        frames = reservoir_frames(self.run_dir, start=0, count=500)
        n = 18
        # Generic compact model weights are only for the standalone parity test;
        # target alignment itself uses real saved EBrush prediction/ground truth.
        structure = {
            'units': n, 'inputs': 6, 'edges': {'src': [], 'dst': [], 'weight': [], 'shown': 0, 'total': 0},
            'readout': {'nodes': [0.0]*n, 'bias': 0.0, 'bypass': [0.0]*6,
                        'input_mean': [0]*6, 'input_std': [1]*6,
                        'target_mean': 0.0, 'target_std': 1.0},
            'input_weights': [[0]*6 for _ in range(n)],
            'forecast': {'horizon': 10, 'washout': 16, 'project': 'ebrush_motion_v1'},
        }
        payload = run_payload(structure, frames, 'real-training')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'engine.js').write_text(engine, encoding='utf-8')
            (path/'payload.json').write_text(json.dumps(payload), encoding='utf-8')
            (path/'main.js').write_text('''const fs = require('fs');
const E = require('./engine.js'), p = JSON.parse(fs.readFileSync('payload.json'));
const d = E.prepareRun(p);
const b = d.episodeBoundaries.findIndex(Boolean);
console.log(JSON.stringify({b, before: d.outputAtSource[b-1], lastValid: d.outputAtSource[b-11],
 first: d.outputAtSource[b], firstIsNan: Number.isNaN(d.outPred[b+10]),
 aligned: Number.isFinite(d.outPred[b-1]), boundary: d.episodeBoundaries[b],
 sameEpisode: d.episodes[b] !== d.episodes[b-1], blank: !Number.isFinite(d.outPred[b]),
 local: d.localSteps[b], legacy: d.legacyUnbounded}));''', encoding='utf-8')
            result = subprocess.run(['node','main.js'], cwd=tmp, capture_output=True, text=True, check=True)
        got = json.loads(result.stdout)
        self.assertEqual(got['b'], 250)
        self.assertEqual(got['before'], -1)
        self.assertEqual(got['lastValid'], 249)
        self.assertEqual(got['first'], 260)
        self.assertTrue(got['firstIsNan'] and got['blank'] and got['boundary'] and got['sameEpisode'])
        self.assertEqual(got['local'], 0)
        self.assertFalse(got['legacy'])


if __name__ == '__main__':
    unittest.main()
