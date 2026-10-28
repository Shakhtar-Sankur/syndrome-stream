"""Sliding-window (streaming) decoding of a stim detector error model.

A memory experiment produces detectors round by round. Instead of waiting for
every round and decoding the whole history, a streaming decoder works on a
window of `w` rounds, keeps the corrections it is sure of (the oldest `c`
rounds of the window), applies them to the syndrome, and slides on by `c`.

Window k covers rounds [k*c, k*c + w). Its problem holds every error
mechanism whose earliest detector lies in the window; detectors outside the
window are dropped from that problem. Corrections whose earliest detector
lies in the commit region [k*c, k*c + c) are committed: they flip their
detectors (including later ones) and the logical observables. The last
window commits everything left.

The inner decoder is anything with `decode_batch(syndromes) -> corrections`
built from a check matrix and error probabilities, so the same windowing runs
around PyMatching, IBM's Relay-BP or the syndrome GPU kernel.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class Problem:
    H: sp.csc_matrix  # detectors x error mechanisms
    L: sp.csc_matrix  # observables x error mechanisms
    p: np.ndarray  # probability of each error mechanism
    det_round: np.ndarray  # round index of each detector


def from_dem(dem, round_coord=-1, split_components=False):
    """Check matrix, observable matrix, priors and detector rounds from a stim DetectorErrorModel.

    The round of a detector is its coordinate at `round_coord` (stim's convention: the last one).
    With split_components, each `^`-separated piece of a decomposed error becomes its own column
    with the error's probability, as PyMatching does for graph-like decoding; otherwise every error
    instruction is one column (for BP-type decoders such as Relay-BP)."""
    coords = dem.get_detector_coordinates()
    n_det, n_obs = dem.num_detectors, dem.num_observables
    det_round = np.array([int(round(coords[i][round_coord])) for i in range(n_det)], dtype=np.int64)
    rows, cols, orows, ocols, probs = [], [], [], [], []
    j = 0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        parts, cur = [], []
        for t in inst.targets_copy():
            if t.is_separator() and split_components:
                parts.append(cur)
                cur = []
            elif not t.is_separator():
                cur.append(t)
        parts.append(cur)
        for part in parts:
            dets = [t.val for t in part if t.is_relative_detector_id()]
            obs = [t.val for t in part if t.is_logical_observable_id()]
            if not dets:
                continue  # flips no detector: no decoder can see it
            rows += dets
            cols += [j] * len(dets)
            orows += obs
            ocols += [j] * len(obs)
            probs.append(inst.args_copy()[0])
            j += 1
    H = sp.csc_matrix((np.ones(len(rows), np.uint8), (rows, cols)), shape=(n_det, j))
    L = sp.csc_matrix((np.ones(len(orows), np.uint8), (orows, ocols)), shape=(n_obs, j))
    return Problem(H, L, np.asarray(probs), det_round)


@dataclass
class Window:
    dets: np.ndarray  # detector indices in this window
    errs: np.ndarray  # error mechanisms in this window's problem
    commit: np.ndarray  # bool per entry of `errs`: committed when chosen


def plan(problem, w, c):
    """Windows of `w` rounds that advance by `c` rounds (c <= w)."""
    assert 1 <= c <= w
    H = problem.H.tocsc()
    first = np.full(H.shape[1], np.iinfo(np.int64).max)
    for j in range(H.shape[1]):
        r = problem.det_round[H.indices[H.indptr[j]:H.indptr[j + 1]]]
        first[j] = r.min()
    r_min, r_max = int(problem.det_round.min()), int(problem.det_round.max())
    windows, start = [], r_min
    while True:
        end = start + w
        last = end > r_max
        dets = np.nonzero((problem.det_round >= start) & (problem.det_round < end))[0]
        errs = np.nonzero((first >= start) & (first < end))[0]
        commit = np.ones(len(errs), bool) if last else first[errs] < start + c
        windows.append(Window(dets, errs, commit))
        if last:
            return windows
        start += c


def decode(problem, syndromes, windows, make_decoder):
    """Decode a batch of shots window by window.

    syndromes: (shots, detectors) bool. Returns the predicted observables, (shots, observables) bool.
    make_decoder(H_sub, p_sub) must return an object with decode_batch(syndromes) -> corrections.
    """
    H = problem.H.tocsc()
    L = problem.L.tocsc()
    syn = np.array(syndromes, dtype=np.uint8, copy=True)
    obs = np.zeros((syn.shape[0], L.shape[0]), np.uint8)
    for win in windows:
        sub = H[win.dets][:, win.errs]
        dec = make_decoder(sub, problem.p[win.errs])
        corr = np.asarray(dec.decode_batch(syn[:, win.dets]), dtype=np.int64)[:, win.commit]
        kept = win.errs[win.commit]
        # committed errors flip their detectors (including later rounds) and the observables
        syn ^= (np.asarray(H[:, kept] @ corr.T).T & 1).astype(np.uint8)
        obs ^= (np.asarray(L[:, kept] @ corr.T).T & 1).astype(np.uint8)
    return obs.astype(bool)


def pymatching_decoder(H_sub, p_sub):
    import pymatching

    p = np.clip(p_sub, 1e-12, 0.5 - 1e-12)
    return pymatching.Matching.from_check_matrix(H_sub, weights=np.log((1 - p) / p))
