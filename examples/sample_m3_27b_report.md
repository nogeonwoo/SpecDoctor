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
    - Type & Size     : qwen35 27B PQ2_0 - 2.13 bpw (group 128) | 7.20 GB (6.70 GiB)
    - Baseline M=1    : 113.93 ms/token (8.78 tok/s)
  • Drafter Model     : Ternary-Bonsai-2-27B-DFlash2-ft5-asym-Q2_K-FC-FullRank.gguf
    - Drafter Type    : dflash_parallel
    - Drafter Cost td : 23.85 ms

[3. Target Batch Verification Cost, M = K+1 for each analysed K]
  Batch (M)  | Latency (ms)   | Verify tok/s | Eff. BW (S/t)  | Delta vs M=1   | Overhead (%)
  -----------|----------------|--------------|----------------|----------------|-------------
  M=2        | 137.73         | 14.52        | 52.2 GB/s      | +23.80 ms      | +20.9%      
  M=3        | 266.38         | 11.26        | 27.0 GB/s      | +152.46 ms     | +133.8%     
  M=4        | 220.59         | 18.13        | 32.6 GB/s      | +106.67 ms     | +93.6%      
  M=5        | 788.85         | 6.34         | 9.1 GB/s       | +674.92 ms     | +592.4%     
  Eff. BW = model_bytes / t_v(M): assumes one weight pass per verify step; a lower bound on bus traffic, not a counter reading.

  [Anomaly] Non-monotonic verification latency: M=4 (220.6 ms) is 17% faster than M=3 (266.4 ms). Cause not identified by this tool; confirm with a same-architecture control and per-op timing.
  [Anomaly] M=5 (788.8 ms) is slower than 5 independent M=1 passes (569.6 ms): a per-column fallback would beat this path. Cause not identified by this tool.

[4. Speculative Break-Even & Net Acceleration Matrix]
  Draft (K)  | Cycle Time   | Req Tokens (tau*)  | Break-Even alpha   | Speedup (alpha=70%) | Speedup (alpha=80%) | Think Mode (alpha=15%)
  -----------|--------------|--------------------|--------------------|--------------------|--------------------|---------------------
  K=1         | 161.58    ms | 1.42 tok/cyc       | 41.8%              | +19.9%             | +26.9%             | -18.9%              
  K=2         | 290.23    ms | 2.55 tok/cyc       | 84.1%              | -14.0%             | -4.2%              | -54.0%              
  K=3       * | 244.44    ms | 2.15 tok/cyc       | 59.1%              | +18.1%             | +37.6%             | -45.2%              
  K=4         | 812.70    ms | 7.13 tok/cyc       | IMPOSSIBLE         | -61.1%             | -52.9%             | -83.5%              
  (* = best tested K at the 75% reference alpha, shown only if net-positive there. tau* = cycle_time / t_v(1) needs no acceptance model; alpha = per-position acceptance under a geometric rollout)

--------------------------------------------------------------------------------
[5. Diagnostic Summary & Guidelines]
--------------------------------------------------------------------------------
  • Break-even gate: requires >= 2.15 mean tokens per cycle for K=3 (geometric per-position alpha >= 59.1%; equivalently >= 38.2% accepted/drafted, the ratio llama.cpp prints).
  • Recommended draft length: --spec-draft-n-max 3 (projected +37.6% at assumed alpha=80%; best tested K at the 75% reference alpha).
  • Disallowed draft lengths: K=4 (required tokens per cycle exceed 1+K: net slowdown at any acceptance rate).
  • Reasoning (<think>) mode: at an assumed, unmeasured alpha of 15%, K=3 yields -45.2%. Disable speculation (K=0) or use a runtime acceptance circuit-breaker while thinking.

[Recommended CLI Command]
  ./bin/llama-cli -m /Users/ani/Ai_Agent/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
                  -md /Users/ani/Ai_Agent/models/Ternary-Bonsai-2-27B-DFlash2-ft5-asym-Q2_K-FC-FullRank.gguf \
                  --spec-draft-n-max 3 \
                  -ngl 999 -ngld 999 -t 4 --temp 0.0
================================================================================
```
