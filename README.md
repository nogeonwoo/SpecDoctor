# SpecDoctor

Lightweight Roofline profiler and break-even latency analyzer for LLM speculative decoding on Apple Silicon and unified-memory edge accelerators.

## Motivation

Speculative decoding is often assumed to deliver linear speedups. In practice on memory-bandwidth-bound unified memory architectures:
- The drafter model forward pass ($t_d$) competes for memory bus bandwidth.
- Target verification latency ($t_v$) scales non-linearly with batch size, frequently encountering register-spill and non-power-of-2 SIMD penalties (e.g. 3-wide vectorization stalls).
- In high-entropy generation or reasoning (`<think>`) phases, draft acceptance ($\alpha$) frequently falls below the physical break-even threshold ($\alpha_{\text{break-even}}$), resulting in net-negative throughput.

`spec-doctor` empirically profiles target model batch verification scaling across batch sizes $M = 1 \dots 8$, measures drafter latency, and computes the exact acceptance threshold required for net-positive acceleration.

## Analytical Formulation

Speculative decoding provides acceleration ($\text{TPS}_{\text{spec}} > \text{TPS}_{\text{base}}$) if and only if:

$$\alpha > \alpha_{\text{break-even}} = \frac{t_d(K) + \Delta t_v(K)}{K \cdot t_v(1)}$$

Where:
- $t_v(1)$: Target model single-token decode latency ($M=1$).
- $t_d(K)$: Latency to draft $K$ tokens.
- $\Delta t_v(K) = t_v(K+1) - t_v(1)$: Incremental target latency to verify $K$ tokens in batch.
- $K \cdot t_v(1)$: Theoretical upper bound on time saved under 100% acceptance.

If $\alpha < \alpha_{\text{break-even}}$, speculation actively degrades generation speed. If $\alpha_{\text{break-even}} \ge 1.0$, speculative acceleration is mathematically impossible on that hardware configuration.

## Installation

```bash
git clone https://github.com/nogeonwoo/SpecDoctor.git
cd SpecDoctor
pip install -e .
```

Requires Python 3.9+ and macOS with `llama-bench` / `llama-cli` built locally or available in `$PATH`.

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
| `--td` | Fixed drafter latency in ms | None |
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
    - Type & Size     : qwen35 27B PQ2_0 | 6.70 GB
    - Baseline M=1    : 117.35 ms/token (8.52 tok/s)
  • Drafter Model     : Ternary-Bonsai-2-27B-DFlash2-ft5-asym-Q2_K-FC-FullRank.gguf
    - Drafter Type    : dflash_parallel
    - Drafter Cost td : 23.97 ms

[3. Target Batch Verification Scaling Cost (M=1..8)]
  Batch (M)  | Latency (ms)   | TPS (tok/s)  | Delta vs M=1   | Overhead (%) 
  -----------|----------------|--------------|----------------|-------------
  M=2        | 137.78         | 7.26         | +20.43 ms      | +17.4%      
  M=3        | 284.79         | 3.51         | +167.44 ms     | +142.7%     
  M=4        | 228.11         | 4.38         | +110.76 ms     | +94.4%      
  M=5        | 809.03         | 1.24         | +691.68 ms     | +589.4%     

  [SIMD Warning] Apple Silicon 3-Wide SIMD Penalty: M=3 (284.8ms) is SLOWER than M=4 (228.1ms). Avoid K=2!

[4. Speculative Break-Even & Net Acceleration Matrix]
  Draft (K)  | Cycle Time   | Break-Even α   | Speedup (α=70%)  | Speedup (α=80%)  | Think Mode (α=15%)
  -----------|--------------|----------------|------------------|------------------|-------------------
  K=1         | 161.75    ms | 37.8%          | +23.3%           | +30.6%           | -16.6%            
  K=2         | 308.76    ms | 81.6%          | -16.8%           | -7.3%            | -55.4%            
  K=3       * | 252.08    ms | 38.3%          | +17.9%           | +37.4%           | -45.3%            
  K=4         | 833.00    ms | IMPOSSIBLE     | -60.9%           | -52.6%           | -83.4%            
  (* denotes recommended optimal configuration)

--------------------------------------------------------------------------------
[5. Diagnostic Summary & Guidelines]
--------------------------------------------------------------------------------
  • Break-even acceptance rate: Speculative decoding requires alpha > 38.3% to be net-positive.
  • Recommended draft length: Set --spec-draft-n-max 3 (Projected speedup: +37.4% at alpha=80%).
  • SIMD Vectorization constraint: Avoid K=2 due to Apple Silicon 3-wide register penalty. Prefer K=3 or K=4.
  • Reasoning (<think>) Mode: At alpha ~ 10-15%, unmanaged speculative decoding causes a -45.3% slowdown. Lock K=0 or enable dynamic circuit breaker during thinking generation.

[Recommended CLI Command]
  ./bin/llama-cli -m models/target.gguf \
                  -md models/draft.gguf \
                  --spec-draft-n-max 3 \
                  -ngl 999 -ngld 999 -t 4 --temp 0.0
================================================================================
```

## License

MIT
