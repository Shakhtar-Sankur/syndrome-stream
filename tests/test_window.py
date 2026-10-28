"""Sliding-window decoding matches whole-history decoding when the window is long enough."""

import numpy as np
import pymatching
import stim

from syndrome_stream import decode, from_dem, plan, pymatching_decoder

D, ROUNDS, P, SHOTS = 3, 12, 0.004, 4000


def setup():
    circ = stim.Circuit.generated("surface_code:rotated_memory_z", distance=D, rounds=ROUNDS,
                                  after_clifford_depolarization=P, after_reset_flip_probability=P,
                                  before_measure_flip_probability=P, before_round_data_depolarization=P)
    dem = circ.detector_error_model(decompose_errors=True)
    syn, obs = circ.compile_detector_sampler(seed=7).sample(SHOTS, separate_observables=True)
    return dem, from_dem(dem, split_components=True), syn, obs


def test_windows_cover_every_error_once():
    _, prob, _, _ = setup()
    for w, c in [(ROUNDS * 2, ROUNDS * 2), (6, 3), (3, 1), (4, 4)]:
        committed = np.concatenate([win.errs[win.commit] for win in plan(prob, w, c)])
        assert len(committed) == len(set(committed.tolist())) == prob.H.shape[1]


def test_one_window_equals_matching_on_the_same_graph():
    dem, prob, syn, obs = setup()
    ours = decode(prob, syn, plan(prob, 10**6, 10**6), pymatching_decoder)
    ref = pymatching.Matching.from_detector_error_model(dem).decode_batch(syn)
    # same graph up to how parallel edges are merged: logical error rates agree closely
    assert abs(np.mean(ours != obs) - np.mean(ref != obs)) < 0.01


def test_window_of_2d_rounds_matches_whole_history():
    _, prob, syn, obs = setup()
    whole = np.mean(decode(prob, syn, plan(prob, 10**6, 10**6), pymatching_decoder) != obs)
    streamed = np.mean(decode(prob, syn, plan(prob, 2 * D, D), pymatching_decoder) != obs)
    assert streamed <= whole * 1.3 + 0.005


def test_detector_rounds_from_the_circuit_match_stim_time():
    from syndrome_stream.window import detector_rounds

    for rounds in (3, 6):
        circ = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=rounds,
                                      after_clifford_depolarization=P)
        inferred = detector_rounds(circ)
        coords = circ.detector_error_model().get_detector_coordinates()
        t = np.array([int(coords[i][-1]) for i in range(len(inferred))])
        # the final data-qubit layer joins the last round of checks
        assert np.array_equal(inferred, np.minimum(t, rounds - 1))
