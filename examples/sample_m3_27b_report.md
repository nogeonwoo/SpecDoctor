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
    - Type & Size     : qwen35 27B PQ2_0 - 2.13 bpw (group 128) | 6.70 GB
    - Baseline M=1    : 128.94 ms/token (7.76 tok/s)
  • Drafter Model     : Ternary-Bonsai-2-27B-DFlash2-ft5-asym-Q2_K-FC-FullRank.gguf
    - Drafter Type    : dflash_parallel
    - Drafter Cost td : 24.20 ms

[3. Target Batch Verification Scaling Cost (M=1..8)]
  Batch (M)  | Latency (ms)   | TPS (tok/s)  | Achieved BW    | Delta vs M=1   | Overhead (%)
  -----------|----------------|--------------|----------------|----------------|-------------
  M=2        | 139.52         | 7.17         | 48.0 GB/s      | +10.58 ms      | +8.2%       
  M=3        | 279.49         | 3.58         | 24.0 GB/s      | +150.55 ms     | +116.8%     
  M=4        | 225.32         | 4.44         | 29.7 GB/s      | +96.39 ms      | +74.8%      
  M=5        | 801.85         | 1.25         | 8.4 GB/s       | +672.92 ms     | +521.9%     

  [Anomaly] Non-monotonic verification latency: M=3 (279.5ms, 24.0 GB/s) is SLOWER than M=4 (225.3ms, 29.7 GB/s). Kernel multi-column dispatch/tiling inefficiency.
  [Anomaly] Throughput collapse at M=5: achieved bandwidth fell to 8.4 GB/s (801.9ms), suggesting kernel fallback or register spill.

[4. Speculative Break-Even & Net Acceleration Matrix]
  Draft (K)  | Cycle Time   | Req Tokens (tau*)  | Break-Even alpha   | Speedup (alpha=70%) | Speedup (alpha=80%) | Think Mode (alpha=15%)
  -----------|--------------|--------------------|--------------------|--------------------|--------------------|---------------------
  K=1         | 163.72    ms | 1.27 tok/cyc       | 27.0%              | +33.9%             | +41.8%             | -9.4%               
  K=2         | 303.69    ms | 2.36 tok/cyc       | 76.7%              | -7.0%              | +3.6%              | -50.2%              
  K=3       * | 249.52    ms | 1.94 tok/cyc       | 52.1%              | +30.9%             | +52.5%             | -39.2%              
  K=4         | 826.05    ms | 6.41 tok/cyc       | IMPOSSIBLE         | -56.7%             | -47.5%             | -81.6%              
  (* denotes recommended optimal configuration; tau* = cycle_time / t_v(1) is model-free)

--------------------------------------------------------------------------------
[5. Diagnostic Summary & Guidelines]
--------------------------------------------------------------------------------
  • Break-even gate: Requires mean tokens per cycle >= 1.94 (geometric alpha >= 52.1%) for K=3.
  • Recommended draft length: Set --spec-draft-n-max 3 (Projected speedup: +52.5% at alpha=80%).
  • Disallowed draft lengths: Avoid K=[4] where target batch latency causes net slowdown regardless of acceptance rate.
  • Reasoning (<think>) Mode: At alpha ~ 10-15%, unmanaged speculative decoding causes a -39.2% slowdown. Lock K=0 or enable a dynamic circuit breaker during thinking generation.

[Recommended CLI Command]
  ./bin/llama-cli -m /Users/ani/Ai_Agent/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
                  -md /Users/ani/Ai_Agent/models/Ternary-Bonsai-2-27B-DFlash2-ft5-asym-Q2_K-FC-FullRank.gguf \
                  --spec-draft-n-max 3 \
                  -ngl 999 -ngld 999 -t 4 --temp 0.0
================================================================================
```
