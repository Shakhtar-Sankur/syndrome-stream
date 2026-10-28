"""S1b: sliding-window Relay-BP on IBM's [[144,12,12]] gross code.

The inner decoder is syndrome's Relay-BP (CUDA on a GPU, or its CPU build), which gives the same
decodings as IBM's reference on every shot checked. Each window's check matrix is a sub-matrix of
the whole-history one, decoded with the same Relay-BP settings as IBM's published results.

    python bench/s1b_gross_windows.py --syndrome ../syndrome --relay ../relay --p 0.006 \
        --shots 2000 --windows whole,6:3,4:2,3:1 --backend cuda --out s1b.json
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import stim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syndrome_stream import decode, detector_rounds, from_dem, plan  # noqa: E402

NAME = ("circuit=bicycle_bivariate_144_12_12_memory_Z,distance=12,rounds=12,error_rate={p},"
        "noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim")


def relay_decoder_factory(backend, gamma_seed, timings):
    from syndrome import Problem, RelayParams, decode_cpu, decode_cuda

    par = RelayParams()

    class Inner:
        def __init__(self, H, p):
            self.prob = Problem(H.tocsr(), p)
            self.gam = self.prob.gammas(par, par.num_sets, seed=gamma_seed)

        def decode_batch(self, syn):
            t0 = time.perf_counter()
            syn = np.ascontiguousarray(syn, dtype=np.uint8)
            if backend == "cuda":
                dec = decode_cuda(self.prob, par, self.gam, syn, want_decodings=True, dtype="f32")[0]
            else:
                dec = decode_cpu(self.prob, par, self.gam, syn, dtype="f32")[0]
            timings.append({"vars": int(self.prob.n_vars), "checks": int(self.prob.n_checks),
                            "shots": int(syn.shape[0]), "s": time.perf_counter() - t0})
            return dec

    return Inner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syndrome", type=Path, required=True, help="path to the syndrome repository")
    ap.add_argument("--relay", type=Path, required=True, help="path to IBM's relay repository (circuits)")
    ap.add_argument("--p", default="0.006")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--windows", default="whole,6:3,4:2,3:1")
    ap.add_argument("--backend", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("s1b.json"))
    a = ap.parse_args()
    sys.path.insert(0, str(a.syndrome))

    circ = stim.Circuit.from_file(a.relay / "tests/testdata/bicycle_bivariate" / NAME.format(p=a.p))
    dem = circ.detector_error_model()
    prob = from_dem(dem, det_round=detector_rounds(circ))
    syn, obs = circ.compile_detector_sampler(seed=a.seed).sample(a.shots, separate_observables=True)
    print(f"gross code p={a.p}: {prob.H.shape[0]} detectors, {prob.H.shape[1]} error mechanisms, "
          f"rounds {prob.det_round.min()}..{prob.det_round.max()}, {a.shots} shots", flush=True)

    results = {"code": "[[144,12,12]] memory_Z, 12 rounds", "p": float(a.p), "shots": a.shots,
               "backend": a.backend, "machine": platform.platform(), "runs": []}
    for spec in a.windows.split(","):
        w, c = (10**6, 10**6) if spec == "whole" else map(int, spec.split(":"))
        timings = []
        t0 = time.perf_counter()
        pred = decode(prob, syn, plan(prob, w, c), relay_decoder_factory(a.backend, 2026, timings))
        wall = time.perf_counter() - t0
        fail = np.any(pred != obs, axis=1)
        run = {"window": spec, "logical_errors": int(fail.sum()), "ler": float(fail.mean()),
               "windows": len(timings), "wall_s": wall,
               "largest_window_vars": max(t["vars"] for t in timings),
               "decoder_s": [round(t["s"], 3) for t in timings]}
        results["runs"].append(run)
        print(f"{spec:>6}: {run['logical_errors']:4d} / {a.shots} logical errors ({run['ler']:.4f}), "
              f"{run['windows']} windows, largest {run['largest_window_vars']} error mechanisms, "
              f"{wall:.0f} s", flush=True)
        a.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
