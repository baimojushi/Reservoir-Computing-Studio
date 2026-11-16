"""Pure numerical checks for the V0.6 intuitive reservoir sandbox."""
import numpy as np

from rcstudio.intuitive_lab import (
    make_signal, run_states, simulate_sandbox, stable_positions,
)


def test_layout_is_stable_and_independent_of_network_parameters():
    a = stable_positions(64, seed=17)
    b = stable_positions(64, seed=17)
    c = stable_positions(64, seed=18)
    assert a.shape == (64, 2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.all(np.sum(a * a, axis=1) <= 1.0 + 1e-12)


def test_signal_presets_and_hand_drawn_interpolation_are_bounded():
    for kind in ("sine", "pulse", "step", "chirp"):
        values = make_signal(kind, 180)
        assert values.shape == (180,)
        assert np.isfinite(values).all()
        assert np.max(np.abs(values)) <= 1.0
    hand = make_signal("handdrawn", 9, [-1.0, 0.0, 1.0])
    assert np.isclose(hand[0], -1.0) and abs(hand[15]) < 0.04 and np.isclose(hand[-1], 1.0)


def test_state_update_matches_leaky_tanh_equation():
    W = np.array([[0.0, 0.2], [-0.1, 0.0]])
    win = np.array([0.5, -0.25])
    bias = np.array([0.01, 0.02])
    inputs = np.array([0.4, -0.2])
    leak = 0.3
    got = run_states(W, win, bias, leak, inputs)
    state = np.zeros(2)
    expected = []
    for u in inputs:
        state = (1 - leak) * state + leak * np.tanh(W @ state + win * u + bias)
        expected.append(state.copy())
    assert np.allclose(got, expected, atol=1e-7)


def test_sandbox_is_reproducible_bounded_and_reports_probe_limits():
    a = simulate_sandbox(seed=91, spectral_radius=0.9, leak_rate=0.35,
                         input_scaling=0.8, units=40, steps=100)
    b = simulate_sandbox(seed=91, spectral_radius=0.9, leak_rate=0.35,
                         input_scaling=0.8, units=40, steps=100)
    assert a["states"].shape == (100, 40)
    assert a["delta_states"].shape == (100, 40)
    assert a["initial_gap"].shape == (100,)
    assert a["pulse_effect"].shape == (100,)
    assert np.array_equal(a["states"], b["states"])
    assert np.all(np.isfinite(a["states"]))
    assert np.max(np.abs(a["states"])) <= 1.0
    assert abs(a["spectral_radius"] - 0.9) < 1e-6
    assert a["pulse_effect"].max() > 0.0
    assert a["initial_gap"][0] > 0.0
    assert a["readout_nodes"].shape == (40,)
    assert a["bypass"].shape == (1,)
    assert a["horizon"] == 6
    assert np.isnan(a["prediction"][0])  # washout is omitted
    assert np.isfinite(a["truth"][0])
    assert np.isfinite(a["test_rmse"])
    assert abs(a["readout_train_fraction"] - 0.70) < 0.02
    valid = np.flatnonzero(np.isfinite(a["prediction"]))
    for index in valid[::7]:
        reconstructed = (a["output_offset"] + a["output_scale"] * (
            a["readout_bias"] + a["bypass"][0] *
            (a["inputs"][index] - a["input_mean"][0]) / a["input_std"][0]
            + a["readout_nodes"] @ a["states"][index]))
        assert np.isclose(reconstructed, a["prediction"][index], atol=1e-6)
    assert "不能单独证明" in a["status_note"]


def test_edge_sampling_is_explicit_and_does_not_change_the_full_network():
    payload = simulate_sandbox(seed=12, units=64, steps=80, max_edges=55)
    assert payload["edge_total"] > payload["edge_shown"] == 55
    assert payload["edge_method"] == "uniform_random_sample"
    assert len(payload["edge_src"]) == len(payload["edge_dst"]) == len(payload["edge_weights"]) == 55
    assert payload["W"].shape == (64, 64)
    assert np.count_nonzero(payload["W"]) == payload["edge_total"]
