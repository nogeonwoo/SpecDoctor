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
    tau_star: float  # Model-free break-even tokens per cycle: cycle_time / t_v1
    alpha_break_even: Optional[float]  # Solved under geometric rollout model
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
    critical_tau_star: float = 2.0
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
                    "tau_star": round(v.tau_star, 3),
                    "alpha_break_even": round(v.alpha_break_even, 4) if v.alpha_break_even is not None else None,
                    "alpha_break_even_pct": round(v.alpha_break_even * 100, 1) if v.alpha_break_even is not None else None,
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
            "critical_tau_star": round(self.critical_tau_star, 3),
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

def solve_geometric_alpha_break_even(k: int, tau_star: float) -> Optional[float]:
    """
    Solves for alpha in [0.0, 1.0] under the geometric rollout model:
      1 + sum_{i=1}^k alpha^i = tau_star
    i.e.:
      sum_{i=1}^k alpha^i = tau_star - 1.0

    Returns:
      float in [0.0, 1.0] if break-even is physically achievable.
      None if tau_star - 1.0 > k (impossible even at alpha = 1.0).
    """
    target = tau_star - 1.0
    if target <= 0.0:
        return 0.0
    if target > float(k):
        return None  # Mathematically impossible

    low, high = 0.0, 1.0
    for _ in range(40):
        mid = (low + high) / 2.0
        val = sum(mid ** i for i in range(1, k + 1))
        if val < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0

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
        m_used = k + 1 if (k + 1) in profile.batch_curve else k
        if m_used not in profile.batch_curve:
            avail = sorted(profile.batch_curve.keys())
            m_used = avail[-1]

        t_v_m = profile.batch_curve[m_used].latency_ms
        delta_tv = t_v_m - t_v1

        # Drafter latency for K tokens
        td_k = (td_base * k) if is_autoregressive else td_base
        cycle_time = td_k + t_v_m

        # Model-free break-even required tokens per cycle
        tau_star = cycle_time / t_v1

        # Break-Even acceptance rate under identical geometric rollout model
        alpha_be = solve_geometric_alpha_break_even(k, tau_star)
        can_accel = (alpha_be is not None and alpha_be < 1.0)

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
            tau_star=tau_star,
            alpha_break_even=alpha_be,
            can_ever_accelerate=can_accel,
            speedup_at_50_pct=sp_50,
            speedup_at_70_pct=sp_70,
            speedup_at_80_pct=sp_80,
            speedup_at_90_pct=sp_90,
            speedup_at_thinking_15_pct=sp_think,
        )

    # Determine recommended K among achievable configurations
    rec_k = best_k_75
    crit_alpha = k_points[rec_k].alpha_break_even if (rec_k in k_points and k_points[rec_k].alpha_break_even is not None) else 0.55
    crit_tau = k_points[rec_k].tau_star if rec_k in k_points else 2.0
    think_loss = k_points[rec_k].speedup_at_thinking_15_pct * 100.0 if rec_k in k_points else -15.0

    # Build actionable, non-contradictory prescriptions
    prescriptions = []
    if not any(pt.can_ever_accelerate for pt in k_points.values()):
        prescriptions.append(
            "Critical: Drafter latency and target batch verification overhead exceed baseline single-token decode time. "
            "Speculative decoding cannot accelerate this model on this hardware configuration (all draft lengths yield negative throughput)."
        )
    else:
        prescriptions.append(
            f"Break-even gate: Requires mean tokens per cycle >= {crit_tau:.2f} (geometric alpha >= {crit_alpha * 100:.1f}%) for K={rec_k}."
        )
        prescriptions.append(
            f"Recommended draft length: Set --spec-draft-n-max {rec_k} (Projected speedup: +{k_points[rec_k].speedup_at_80_pct * 100:.1f}% at alpha=80%)."
        )
        impossible_ks = [k for k, pt in k_points.items() if not pt.can_ever_accelerate]
        if impossible_ks:
            prescriptions.append(
                f"Disallowed draft lengths: Avoid K={impossible_ks} where target batch latency causes net slowdown regardless of acceptance rate."
            )
        prescriptions.append(
            f"Reasoning (<think>) Mode: At alpha ~ 10-15%, unmanaged speculative decoding causes a {think_loss:.1f}% slowdown. "
            "Lock K=0 or enable a dynamic circuit breaker during thinking generation."
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
        critical_tau_star=crit_tau,
        thinking_mode_speedup_loss_pct=think_loss,
        simd_anomalies=profile.simd_anomaly_flags,
        prescriptions=prescriptions,
    )
