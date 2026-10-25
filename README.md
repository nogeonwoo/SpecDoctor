# 🏥 SpecDoctor: On-Device Speculative Decoding Roofline Diagnostic Engine

> **Empirical Roofline & Hardware Break-Even Analyzer for LLM Speculative Decoding on Apple Silicon & Edge Devices**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform: macOS Metal](https://img.shields.io/badge/Platform-Apple%20Silicon%20(M1--M4)-black.svg)]()
[![Hardware Bound: Memory Bandwidth](https://img.shields.io/badge/Roofline-Memory%20Bound-orange.svg)]()

---

## 📌 Why SpecDoctor?

Speculative decoding is often advertised as a *"free 2×–3× speedup"*. In reality, on unified memory systems (such as Apple Silicon Macs) or memory-bandwidth-constrained edge devices:
1. **Drafter forward latency ($t_d$)** consumes precious memory bus cycles.
2. **Target verification latency ($t_v(K)$)** scales non-linearly with batch size $K$, frequently hitting non-power-of-2 SIMD register stalls.
3. During **reasoning monologues (`<think>`)** or high-entropy outputs ($\text{temperature} > 0.6$), drafter acceptance ($\alpha$) drops below the physical break-even point, **causing speculative decoding to run SLOWER than single-token baseline decoding!**

**SpecDoctor solves this blind spot.** In under 60 seconds, it profiles your physical hardware, measures your model's exact batch verification curve ($M=1\dots 8$), and computes the mathematical break-even threshold ($\alpha_{\text{break-even}}$) below which speculative decoding must be throttled or disengaged.

---

## 🧮 Mathematical Foundation

Speculative decoding yields net-positive acceleration ($\text{TPS}_{\text{spec}} > \text{TPS}_{\text{base}}$) if and only if:

$$\alpha > \alpha_{\text{break-even}} = \frac{t_d(K) + \Delta t_v(K)}{K \cdot t_v(1)}$$

Where:
- $t_v(1)$: Target model single-token decode latency ($M=1$).
- $t_d(K)$: Latency to draft $K$ tokens.
- $\Delta t_v(K) = t_v(K+1) - t_v(1)$: Additional target latency to verify $K$ tokens in batch.
- $K \cdot t_v(1)$: Theoretical maximum time saved if all $K$ tokens are accepted.

If $\alpha < \alpha_{\text{break-even}}$, speculative decoding **actively slows down your machine**. If $\alpha_{\text{break-even}} \ge 1.0$, acceleration is **physically impossible** on that hardware configuration.

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/nogeonwoo/Ai_Agent.git
cd Ai_Agent
pip install -e .
```

### 2. Run Diagnostics
```bash
# Full diagnostic with target and drafter
spec-doctor -m models/target.gguf -md models/draft.gguf

# Fast diagnostic with user-specified drafter latency (td)
spec-doctor -m models/target.gguf --td 22.5 --k-max 4

# Export Markdown and JSON reports
spec-doctor -m models/target.gguf -md models/draft.gguf --export-md report.md --export-json report.json
```

---

## 📋 Example Terminal Output

```text
================================================================================
🏥 [SpecDoctor] Speculative Decoding Roofline & Hardware Diagnostic Report
================================================================================
[1. Hardware & Physical Memory Bandwidth]
  • Device / Chip     : Apple M3 (10 GPU Cores)
  • Total Unified RAM : 24.0 GB
  • Theoretical Bus   : 100.0 GB/s Roofline ceiling
  • OS / Platform     : Darwin (15.5)

[2. Evaluated Models & Baseline Latency]
  • Target Model      : Ternary-Bonsai-2-27B-PQ2_0.gguf
    - Type & Size     : qwen35 27B PQ2_0 | 6.70 GB
    - Baseline M=1    : 115.55 ms/token (8.67 tok/s)
  • Drafter Model     : Ternary-Bonsai-2-27B-DFlash2-ft5-asym-Q2_K-FC-FullRank.gguf
    - Drafter Type    : dflash_parallel
    - Drafter Cost td : 23.11 ms

[3. Target Batch Verification Scaling Cost (M=1..8)]
  Batch (M)  | Latency (ms)   | TPS (tok/s)  | Delta vs M=1   | Overhead (%)
  -----------|----------------|--------------|----------------|-------------
  M=1        | 115.55         | 8.67         | +0.00 ms       | +0.0%       
  M=2        | 139.14         | 7.19         | +23.59 ms      | +20.4%      
  M=3        | 275.11         | 3.63         | +159.56 ms     | +138.1%     
  M=4        | 224.14         | 4.46         | +108.59 ms     | +94.0%      
  M=5        | 256.40         | 3.90         | +140.85 ms     | +121.9%     

  ⚡ [SIMD WARNING] Apple Silicon 3-Wide SIMD Penalty: M=3 (275.1ms) is SLOWER than M=4 (224.1ms). Avoid K=2!

[4. Speculative Break-Even & Net Acceleration Matrix]
  Draft (K)  | Cycle Time   | Break-Even α   | Speedup (α=70%)  | Speedup (α=80%)  | Think Mode (α=15%)
  -----------|--------------|----------------|------------------|------------------|-------------------
  K=1        | 162.25 ms    | 40.4%          | +18.2%           | +24.1%           | -12.4%            
  K=2        | 298.22 ms    | 79.0%          | -6.3%            | +1.2%            | -41.2%            
  K=3 *      | 247.25 ms    | 38.0%          | +21.4%           | +31.8%           | -14.6%            
  K=4        | 279.51 ms    | 35.5%          | +19.8%           | +28.9%           | -17.2%            
  (* denotes recommended optimal configuration)

--------------------------------------------------------------------------------
📋 [Prescription & Actionable Optimization Guidelines]
--------------------------------------------------------------------------------
  • ✅ Break-Even Gate: Only engage speculative decoding when live acceptance α > 38.0%.
  • 🎯 Recommended Draft Length: Set --spec-draft-n-max 3 (Peak net speedup: +31.8% at α=80%).
  • ⚠️ Apple Silicon Vectorization Rule: Strictly avoid K=2 due to 3-wide SIMD register stall. Use K=3 or K=4.
  • 🧠 Thinking Mode Protection: During <think> blocks (α ≈ 10~15%), speculative decoding causes a -14.6% SLOWDOWN.
    Lock K=0 or employ an adaptive arbiter/circuit breaker to prevent degradation.
================================================================================
```
