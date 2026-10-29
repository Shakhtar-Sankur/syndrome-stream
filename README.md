# syndrome-stream

**Streaming decoding of IBM's quantum LDPC codes for quantum-centric supercomputers.**

A fault-tolerant quantum computer produces a round of error syndromes about every microsecond, and a
classical decoder must keep up or the backlog grows without bound. IBM's codes for its next machines
(bivariate bicycle codes such as the [[144,12,12]] gross code) are decoded with Relay-BP
([Müller et al., arXiv:2506.01779](https://arxiv.org/abs/2506.01779)). IBM has reported a preliminary
FPGA implementation, and NVIDIA ships a GPU Relay-BP in CUDA-Q QEC, but that library is proprietary.

[syndrome](https://github.com/Shakhtar-Sankur/syndrome) is an open CUDA Relay-BP that gives the same
decoding as IBM's reference implementation on every shot checked (256 of 256, float64 and float32), and
decodes a whole 12-round gross-code shot in a median 37.5 ms on a T4, 28.7× IBM's decoder on one CPU
thread. That is still far from a microsecond per round. syndrome-stream takes the next steps:

1. **Streaming.** Decode a window of rounds, commit the oldest, slide on, so each decoding problem stays
   small and latency per round stops growing with the length of the experiment.
2. **On HPC.** Run the decoder as a service on supercomputer GPU nodes, next to the quantum-resource
   interfaces being standardised for Slurm (IBM's QRMI).
3. **Head to head.** Report logical error rate and latency per round against whole-history decoding,
   IBM's reference, and published FPGA and CUDA-Q figures, with every machine and loss recorded.
4. **On real hardware.** Decode error-correction experiments run on IBM quantum processors.

## Status

| Stage | Scope | Status |
|---|---|---|
| S1 | Sliding-window decoding around any inner decoder, checked against whole-history decoding | done for the surface code with PyMatching (below) |
| S1b | The same windows around Relay-BP (IBM's reference, then the syndrome GPU kernel) on the gross code | first measurement done (below) |
| S2 | Decoding service on GPU nodes; throughput and latency per round | planned |
| S3 | Benchmarks against whole-history Relay-BP, IBM's FPGA figure and CUDA-Q | planned |
| S4 | Experiments on IBM quantum hardware, decoded through the service | planned |

## S1: sliding windows, checked on the surface code

[`syndrome_stream/window.py`](syndrome_stream/window.py) reads a stim detector error model. Window *k*
covers rounds [*kc*, *kc* + *w*): its problem holds every error mechanism whose earliest detector is in
the window. Corrections whose earliest detector lies in the first *c* rounds are committed: they flip
their detectors, including those in later rounds, and the logical observables. The last window commits
everything left. The inner decoder is any object with `decode_batch`, so the same windows run around
PyMatching, IBM's Relay-BP or the syndrome GPU kernel.

Rotated surface code, distance 5, 25 rounds, circuit noise p = 0.4%, 20,000 shots, PyMatching as the
inner decoder ([raw output](results/surface_d5_window_2026-10-28.txt)):

| Decoding | Logical error rate | Time |
|---|---|---|
| PyMatching on the whole history (reference) | 3.89% | |
| This pipeline, one window | 3.97% | 53 s |
| **Window 10 rounds (2d), commit 5** | **3.99%** | **12 s** |
| Window 5, commit 3 | 4.66% | 11 s |
| Window 3, commit 1 | 5.73% | 20 s |

A window of about 2d rounds matches whole-history decoding; shorter windows trade accuracy for smaller
problems. The same trade-off for Relay-BP on the gross code is S1b's measurement. The small gap between
the reference and "one window" comes from how parallel graph edges are merged, not from windowing.

## S1b: sliding windows around GPU Relay-BP on the gross code

The same windows around the [syndrome](https://github.com/Shakhtar-Sankur/syndrome) CUDA Relay-BP, on
IBM's [[144,12,12]] gross-code memory circuit: 12 rounds, 1728 detectors and 67,752 error mechanisms,
uniform circuit noise. Every decoding sees the same shots. One Kaggle Tesla T4
([public run](https://www.kaggle.com/code/sankurkundu/notebook6fe8dc7118);
[raw output](results/gross_s1b_kaggle_2026-10-29.txt), [JSON](results/gross_s1b_kaggle_2026-10-29.json)).
The brackets are 95% Wilson intervals.

| Decoding (window : commit, rounds) | Largest problem | p = 0.3%, 1000 shots | p = 0.6%, 200 shots |
|---|---|---|---|
| Whole history | 67,752 | **0 errors** (< 0.38%) | 38.5% [32.0, 45.4] |
| 6 : 3 | 35,424 | 0.1% [0.02, 0.56] | **30.5%** [24.5, 37.2] |
| 4 : 2 | 23,616 | 0.6% [0.28, 1.30] | 35.0% [28.7, 41.8] |
| 3 : 1 | 18,720 | 0.4% [0.16, 1.02] | 31.5% [25.5, 38.2] |

- **At p = 0.3%, windows cost accuracy.** The whole-history decoder made no error in 1000 shots. A 6-round
  window made one, and shorter windows four to six. This is the same trade-off as on the surface code, and
  the gross code needs a window of at least half its 12 rounds.
- **At p = 0.6%, the code is past its useful range.** Every decoder fails on about a third of shots. The
  windows come out no worse than whole-history decoding, and possibly better, likely because on the largest
  problem Relay-BP more often stops at its iteration limit. 200 shots cannot separate them.
- **Size and time.** A 6 : 3 window halves the largest decoding problem, and 3 : 1 cuts it to 28%. The
  slowest single window takes 64 ms per shot on average at p = 0.3%, against 194 ms for the whole history
  (batch time divided by shots). Summed over all windows, the work is not yet smaller. Overlapping windows
  re-decode rounds, so S2 has to pipeline them.

Next: more shots at p = 0.1–0.3%, where IBM reports its operating point, and window 8 : 4.

```bash
pip install -r requirements.txt
python -m pytest -q tests
```

MIT licence. Copyright 2026 Sankur Kundu.
