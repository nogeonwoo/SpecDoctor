```text
================================================================================
🏥 [SpecDoctor] Speculative Decoding Roofline & Hardware Diagnostic Report
================================================================================
[1. Hardware & Physical Memory Bandwidth]
  • Device / Chip     : Apple M3 (10 GPU Cores)
  • Total Unified RAM : 24.0 GB
  • Theoretical Bus   : 100.0 GB/s Roofline ceiling
  • OS / Platform     : Darwin (26.3.1)

[2. Evaluated Models & Baseline Latency]
  • Target Model      : Ternary-Bonsai-2-27B-PQ2_0.gguf
    - Type & Size     : qwen35 27B PQ2_0 - 2.13 bpw (group 128) | 6.70 GB
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

  ⚡ [SIMD WARNING] Apple Silicon 3-Wide SIMD Penalty: M=3 (284.8ms) is SLOWER than M=4 (228.1ms). Avoid K=2!

[4. Speculative Break-Even & Net Acceleration Matrix]
  Draft (K)  | Cycle Time   | Break-Even α   | Speedup (α=70%)  | Speedup (α=80%)  | Think Mode (α=15%)
  -----------|--------------|----------------|------------------|------------------|-------------------
  K=1         | 161.75    ms | 37.8%          | +23.3%           | +30.6%           | -16.6%            
  K=2         | 308.76    ms | 81.6%          | -16.8%           | -7.3%            | -55.4%            
  K=3       * | 252.08    ms | 38.3%          | +17.9%           | +37.4%           | -45.3%            
  K=4         | 833.00    ms | IMPOSSIBLE     | -60.9%           | -52.6%           | -83.4%            
  (* denotes recommended optimal configuration)

--------------------------------------------------------------------------------
📋 [Prescription & Actionable Optimization Guidelines]
--------------------------------------------------------------------------------
  • ✅ Break-Even Gate: Only engage speculative decoding when live acceptance α > 38.3%.
  • 🎯 Recommended Draft Length: Set --spec-draft-n-max 3 (Peak net speedup: +37.4% at α=80%).
  • ⚠️ Apple Silicon Vectorization Rule: Strictly avoid K=2 due to 3-wide SIMD register stall. Use K=3 or K=4.
  • 🧠 Thinking Mode Protection: During <think> blocks (α ≈ 10~15%), speculative decoding causes a -45.3% SLOWDOWN. Lock K=0 or employ an adaptive arbiter/circuit breaker to prevent degradation.

🚀 [Recommended Production Command Line]
  ./bin/llama-cli -m /Users/ani/Ai_Agent/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
                  -md /Users/ani/Ai_Agent/models/Ternary-Bonsai-2-27B-DFlash2-ft5-asym-Q2_K-FC-FullRank.gguf \
                  --spec-draft-n-max 3 \
                  -ngl 999 -ngld 999 -t 4 --temp 0.0
================================================================================
```
