"""
Analyzer: Mathematical Roofline & Speculative Break-Even Diagnostic Engine
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from .profiler import ProfilingResult

@dataclass
class KAnalysisPoint:
    k: int
    batch_m_used: int
    t_d_ms: float
    t_v_ms: float
    cycle_time_ms: float
    delta_tv_ms: float
    alpha_break_even: float
    can_ever_accelerate: bool
    speedup_at_50_pct: float
    speedup_at_70_pct: float
    speedup_at_80_pct: float
    speedup_at_90_pct: float
    speedup_at_thinking_15_pct: float

@dataclass
class DiagnosticReport:
    hardware_info: Dict[str, Any]
    target_info: Dict[str, Any]
    drafter_info: Dict[str, Any]
    k_analysis: Dict[int, KAnalysisPoint] = field(default_factory=dict)
    optimal_k_at_75_pct: int = 3
    recommended_k_max: int = 3
    critical_alpha_threshold: float = 0.55
    thinking_mode_speedup_loss_pct: float = -15.0
    simd_anomalies: List[str] = field(default_factory=list)
    prescriptions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hardware_info": self.hardware_info,
            "target_info": self.target_info,
            "drafter_info": self.drafter_info,
            "k_analysis": {
                k: {
                    "k": v.k,
                    "batch_m_used": v.batch_m_used,
                    "t_d_ms": round(v.t_d_ms, 2),
                    "t_v_ms": round(v.t_v_ms, 2),
                    "cycle_time_ms": round(v.cycle_time_ms, 2),
                    "delta_tv_ms": round(v.delta_tv_ms, 2),
                    "alpha_break_even": round(v.alpha_break_even, 4),
                    "alpha_break_even_pct": round(v.alpha_break_even * 100, 1),
                    "can_ever_accelerate": v.can_ever_accelerate,
                    "speedup_at_50_pct": round(v.speedup_at_50_pct * 100, 1),
                    "speedup_at_70_pct": round(v.speedup_at_70_pct * 100, 1),
                    "speedup_at_80_pct": round(v.speedup_at_80_pct * 100, 1),
                    "speedup_at_90_pct": round(v.speedup_at_90_pct * 100, 1),
                    "speedup_at_thinking_15_pct": round(v.speedup_at_thinking_15_pct * 100, 1),
                }
                for k, v in self.k_analysis.items()
            },
            "optimal_k_at_75_pct": self.optimal_k_at_75_pct,
            "recommended_k_max": self.recommended_k_max,
            "critical_alpha_threshold": round(self.critical_alpha_threshold, 4),
            "critical_alpha_threshold_pct": round(self.critical_alpha_threshold * 100, 1),
            "thinking_mode_speedup_loss_pct": round(self.thinking_mode_speedup_loss_pct, 1),
            "simd_anomalies": self.simd_anomalies,
            "prescriptions": self.prescriptions,
        }

def calculate_expected_tokens(k: int, alpha: float, model: str = "geometric") -> float:
    """
    Computes expected accepted tokens E[N_acc].
    geometric: sum_{i=1}^k alpha^i
    linear: k * alpha
    """
    if model == "geometric":
        # Realistic autoregressive decay
        return sum(alpha ** i for i in range(1, k + 1))
    return k * alpha

def analyze_speculative_dynamics(
    profile: ProfilingResult,
    hardware_dict: Dict[str, Any],
    custom_td_ms: Optional[float] = None,
    k_range: List[int] = [1, 2, 3, 4, 5, 6, 7],
) -> DiagnosticReport:
    """
    Executes full Roofline break-even and speculative scaling analysis.
    """
    t_v1 = profile.target_m1_latency_ms
    if t_v1 <= 0:
        raise ValueError("Invalid target M=1 latency (<= 0)")

    tps_base = 1000.0 / t_v1
    td_base = custom_td_ms if custom_td_ms is not None else (profile.drafter_latency_td_ms or 23.0)
    is_autoregressive = (profile.drafter_type == "autoregressive")

    k_points: Dict[int, KAnalysisPoint] = {}
    best_k_75 = 1
    max_speedup_75 = -999.0

    for k in k_range:
        # Batch size for verifying K tokens is typically M = K + 1 (K draft tokens + 1 target continuation)
        # or M = K if last token is not pipelined.
        # We clamp to maximum measured batch size
        m_used = k + 1 if (k + 1) in profile.batch_curve else k
        if m_used not in profile.batch_curve:
            # Fallback to closest available batch size
            avail = sorted(profile.batch_curve.keys())
            m_used = avail[-1]

        t_v_m = profile.batch_curve[m_used].latency_ms
        delta_tv = t_v_m - t_v1

        # Drafter latency for K tokens
        td_k = (td_base * k) if is_autoregressive else td_base
        cycle_time = td_k + t_v_m

        # Exact Break-Even acceptance rate
        # alpha_break_even * K * t_v1 = td_k + delta_tv
        alpha_be = (td_k + delta_tv) / (k * t_v1)
        can_accel = (alpha_be < 1.0)

        # Helper to compute net speedup for a given alpha
        def get_speedup(alpha_val: float) -> float:
            e_tokens = calculate_expected_tokens(k, alpha_val, model="geometric")
            total_tokens = 1.0 + e_tokens
            tps_spec = (total_tokens / cycle_time) * 1000.0
            return (tps_spec / tps_base) - 1.0

        sp_50 = get_speedup(0.50)
        sp_70 = get_speedup(0.70)
        sp_75 = get_speedup(0.75)
        sp_80 = get_speedup(0.80)
        sp_90 = get_speedup(0.90)
        sp_think = get_speedup(0.15)  # 15% typical in <think> monologue

        if sp_75 > max_speedup_75 and can_accel:
            max_speedup_75 = sp_75
            best_k_75 = k

        k_points[k] = KAnalysisPoint(
            k=k,
            batch_m_used=m_used,
            t_d_ms=td_k,
            t_v_ms=t_v_m,
            cycle_time_ms=cycle_time,
            delta_tv_ms=delta_tv,
            alpha_break_even=alpha_be,
            can_ever_accelerate=can_accel,
            speedup_at_50_pct=sp_50,
            speedup_at_70_pct=sp_70,
            speedup_at_80_pct=sp_80,
            speedup_at_90_pct=sp_90,
            speedup_at_thinking_15_pct=sp_think,
        )

    # Determine recommended K
    # If Apple Silicon and K=2 exhibits 3-wide penalty, filter out K=2
    rec_k = best_k_75
    if rec_k == 2 and profile.simd_anomaly_flags:
        # Prefer K=3 if K=2 suffers anomaly
        rec_k = 3 if 3 in k_points and k_points[3].can_ever_accelerate else 1

    crit_alpha = k_points[rec_k].alpha_break_even if rec_k in k_points else 0.55
    think_loss = k_points[rec_k].speedup_at_thinking_15_pct * 100.0 if rec_k in k_points else -15.0

    # Build actionable prescriptions
    prescriptions = []
    if crit_alpha >= 1.0:
        prescriptions.append(
            "🛑 CRITICAL: Drafter latency + verification overhead exceed target decode time. "
            "Speculative decoding CANNOT accelerate this model on this hardware (all α produce negative speedup)."
        )
    else:
        prescriptions.append(
            f"✅ Break-Even Gate: Only engage speculative decoding when live acceptance α > {crit_alpha * 100:.1f}%."
        )
        prescriptions.append(
            f"🎯 Recommended Draft Length: Set --spec-draft-n-max {rec_k} (Peak net speedup: +{k_points[rec_k].speedup_at_80_pct * 100:.1f}% at α=80%)."
        )
        if any("M=3" in a for a in profile.simd_anomaly_flags):
            prescriptions.append(
                "⚠️ Apple Silicon Vectorization Rule: Strictly avoid K=2 due to 3-wide SIMD register stall. Use K=3 or K=4."
            )
        prescriptions.append(
            f"🧠 Thinking Mode Protection: During <think> blocks (α ≈ 10~15%), speculative decoding causes a {think_loss:.1f}% SLOWDOWN. "
            "Lock K=0 or employ an adaptive arbiter/circuit breaker to prevent degradation."
        )

    return DiagnosticReport(
        hardware_info=hardware_dict,
        target_info={
            "model_path": profile.target_model_path,
            "model_type": profile.target_model_type,
            "model_size_gb": profile.target_model_size_gb,
            "m1_latency_ms": profile.target_m1_latency_ms,
            "m1_tps": profile.target_m1_tps,
        },
        drafter_info={
            "model_path": profile.drafter_model_path,
            "drafter_type": profile.drafter_type,
            "td_ms": td_base,
        },
        k_analysis=k_points,
        optimal_k_at_75_pct=best_k_75,
        recommended_k_max=rec_k,
        critical_alpha_threshold=crit_alpha,
        thinking_mode_speedup_loss_pct=think_loss,
        simd_anomalies=profile.simd_anomaly_flags,
        prescriptions=prescriptions,
    )
