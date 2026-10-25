# SpecDoctor

Lightweight Roofline profiler and break-even latency analyzer for LLM speculative decoding on Apple Silicon and unified-memory edge accelerators.

## Motivation

Speculative decoding is often assumed to deliver linear speedups. In practice on memory-bandwidth-bound unified memory architectures:
- The drafter forward pass ($t_d$) is pure added latency in a serial draft→verify loop.
- Target verification latency $t_v(M)$ is not flat in the batch size $M$: low-bit (dequantization-heavy) kernels turn compute-bound at small $M$, and custom kernels can have dispatch/tiling cliffs at specific column counts. `spec-doctor` measures the curve and flags shape anomalies without asserting their cause.
- When draft acceptance ($\alpha$) is low (e.g. in reasoning phases), speculation can fall below break-even and lower throughput. The acceptance rates used in the tables are assumptions, not measurements.

`spec-doctor` empirically profiles target model batch verification scaling across batch sizes $M = 1 \dots 8$, measures drafter latency, and computes the required tokens per cycle ($\tau^*$) and the corresponding acceptance threshold under a stated acceptance model.

## Analytical Formulation

One speculative cycle drafts $K$ tokens (cost $t_d(K)$), then verifies $K+1$ positions in a single target pass (cost $t_v(K+1)$), and yields $1 + (\text{accepted drafts})$ tokens. Baseline decode yields one token per $t_v(1)$. Speculation breaks even when the mean tokens per cycle equals

$$\tau^* = \frac{t_d(K) + t_v(K+1)}{t_v(1)}$$

and the net speedup is $\bar N/\tau^* - 1$, where $\bar N$ is the mean tokens per cycle. $\tau^*$ depends on no acceptance model.

To express $\tau^*$ as an acceptance rate we assume i.i.d. per-position acceptance $\alpha$ (geometric rollout), so $\bar N = 1 + \sum_{i=1}^{K}\alpha^i$. The break-even $\alpha$ is the unique root in $(0,1)$ of

$$\sum_{i=1}^{K}\alpha^i = \tau^* - 1,$$

found by bisection. A root exists iff $\tau^* < K+1$; otherwise speculation loses even at $\alpha = 1$.

$\alpha$ is the *per-position* acceptance probability, not the accepted/drafted ratio that llama.cpp prints; the break-even for the latter is $(\tau^*-1)/K$.

Assumptions: serial draft→verify with no overlap; the verification batch is exactly $K+1$ tokens (a truncated draft lands on a different $M$); no per-cycle overhead beyond $t_d$ and $t_v$; $t_v(M)$ measured with `llama-bench -p M` at zero context depth, which computes logits for the last position only.

## Installation

```bash
git clone https://github.com/nogeonwoo/SpecDoctor.git
cd SpecDoctor
pip install -e .
```

Requires Python 3.9+ and macOS with `llama-bench` / `llama-cli` built locally or available in `$PATH`. Automatic drafter-cost measurement for DFlash drafters additionally needs a build that supports `--spec-type draft-dflash` and prints `dflash-timer ... total_td=... ms` under `LLAMA_SPEC_DEBUG=1`; otherwise pass `--td` (the tool never substitutes a default).

## Usage

```bash
# Profile target model with drafter model
spec-doctor -m models/target.gguf -md models/draft.gguf

# Profile target with known drafter latency (td in ms)
spec-doctor -m models/target.gguf --td 23.5 --k-max 4

# Export reports
spec-doctor -m models/target.gguf -md models/draft.gguf --export-md report.md --export-json report.json
```

### Options

| Flag | Description | Default |
|---|---|---|
| `-m, --target` | Path to target GGUF model (required) | - |
| `-md, --draft` | Path to drafter GGUF model | None |
| `--td` | Drafter latency in ms per draft step (treated as independent of $K$ unless the drafter is an autoregressive GGUF passed via `-md`) | None |
| `--k-max` | Maximum draft length $K$ to scan | 6 |
| `-r, --repetitions`| Number of repetitions per batch size | 2 |
| `-t, --threads` | Number of CPU threads | 4 |
| `--bench-bin` | Custom path to `llama-bench` | `./bin/llama-bench` |
| `--cli-bin` | Custom path to `llama-cli` | `./bin/llama-cli` |
| `--export-md` | Path to export Markdown summary | None |
| `--export-json` | Path to export JSON metrics | None |

## Example Output

```text
================================================================================
[SpecDoctor] Speculative Decoding Roofline & Hardware Diagnostic
================================================================================
[1. Hardware & Physical Memory Bandwidth]
  • Device / Chip     : Apple M3 (10 GPU Cores)
  • Total Unified RAM : 24.0 GB
  • Theoretical Bus   : 100.0 GB/s Roofline ceiling
  • OS / Platform     : Darwin (26.3.1)

[2. Evaluated Models & Baseline Latency]
  • Target Model      : Ternary-Bonsai-2-27B-PQ2_0.gguf
    - Type & Size     : qwen35 27B PQ2_0 - 2.13 bpw (group 128) | 7.19 GB (6.70 GiB)
    - Baseline M=1    : 128.94 ms/token (7.76 tok/s)
  • Drafter Model     : Ternary-Bonsai-2-27B-DFlash2-ft5-asym-Q2_K-FC-FullRank.gguf
    - Drafter Type    : dflash_parallel
    - Drafter Cost td : 24.20 ms

[3. Target Batch Verification Cost, M = K+1 for each analysed K]
  Batch (M)  | Latency (ms)   | Verify tok/s | Eff. BW (S/t)  | Delta vs M=1   | Overhead (%)
  -----------|----------------|--------------|----------------|----------------|-------------
  M=2        | 139.52         | 14.33        | 51.6 GB/s      | +10.58 ms      | +8.2%
  M=3        | 279.49         | 10.73        | 25.7 GB/s      | +150.55 ms     | +116.8%
  M=4        | 225.32         | 17.75        | 31.9 GB/s      | +96.38 ms      | +74.7%
  M=5        | 801.85         | 6.24         | 9.0 GB/s       | +672.91 ms     | +521.9%
  Eff. BW = model_bytes / t_v(M): assumes one weight pass per verify step; a lower bound on bus traffic, not a counter reading.

  [Anomaly] Non-monotonic verification latency: M=4 (225.3 ms) is 19% faster than M=3 (279.5 ms). Cause not identified by this tool; confirm with a same-architecture control and per-op timing.
  [Anomaly] M=5 (801.9 ms) is slower than 5 independent M=1 passes (644.7 ms): a per-column fallback would beat this path. Cause not identified by this tool.

[4. Speculative Break-Even & Net Acceleration Matrix]
  Draft (K)  | Cycle Time   | Req Tokens (tau*)  | Break-Even alpha   | Speedup (alpha=70%) | Speedup (alpha=80%) | Think Mode (alpha=15%)
  -----------|--------------|--------------------|--------------------|--------------------|--------------------|---------------------
  K=1         | 163.72    ms | 1.27 tok/cyc       | 27.0%              | +33.9%             | +41.8%             | -9.4%
  K=2         | 303.69    ms | 2.36 tok/cyc       | 76.7%              | -7.0%              | +3.6%              | -50.2%
  K=3       * | 249.52    ms | 1.94 tok/cyc       | 52.1%              | +30.9%             | +52.5%             | -39.2%
  K=4         | 826.05    ms | 6.41 tok/cyc       | IMPOSSIBLE         | -56.7%             | -47.5%             | -81.6%
  (* = best tested K at the 75% reference alpha, shown only if net-positive there. tau* = cycle_time / t_v(1) needs no acceptance model; alpha = per-position acceptance under a geometric rollout)

--------------------------------------------------------------------------------
[5. Diagnostic Summary & Guidelines]
--------------------------------------------------------------------------------
  • Break-even gate: requires >= 1.94 mean tokens per cycle for K=3 (geometric per-position alpha >= 52.1%; equivalently >= 31.2% accepted/drafted, the ratio llama.cpp prints).
  • Recommended draft length: --spec-draft-n-max 3 (projected +52.5% at assumed alpha=80%; best tested K at the 75% reference alpha).
  • Disallowed draft lengths: K=4 (required tokens per cycle exceed 1+K: net slowdown at any acceptance rate).
  • Reasoning (<think>) mode: at an assumed, unmeasured alpha of 15%, K=3 yields -39.2%. Disable speculation (K=0) or use a runtime acceptance circuit-breaker while thinking.

[Recommended CLI Command]
  ./bin/llama-cli -m models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
                  -md models/Ternary-Bonsai-2-27B-DFlash2-ft5-asym-Q2_K-FC-FullRank.gguf \
                  --spec-draft-n-max 3 \
                  -ngl 999 -ngld 999 -t 4 --temp 0.0
================================================================================
```

Generated from a single measurement run (Apple M3, 24 GB).

## License

MIT
