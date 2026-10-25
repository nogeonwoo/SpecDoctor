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
    optimal_k_at_75_pct: int = 0
    recommended_k_max: int = 0  # 0 == do not speculate
    critical_alpha_threshold: Optional[float] = None
    critical_tau_star: Optional[float] = None
    thinking_mode_speedup_loss_pct: Optional[float] = None
    simd_anomalies: List[str] = field(default_factory=list)
    prescriptions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

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
            "critical_alpha_threshold": None if self.critical_alpha_threshold is None else round(self.critical_alpha_threshold, 4),
            "critical_alpha_threshold_pct": None if self.critical_alpha_threshold is None else round(self.critical_alpha_threshold * 100, 1),
            "critical_tau_star": None if self.critical_tau_star is None else round(self.critical_tau_star, 3),
            "thinking_mode_speedup_loss_pct": None if self.thinking_mode_speedup_loss_pct is None else round(self.thinking_mode_speedup_loss_pct, 1),
            "simd_anomalies": self.simd_anomalies,
            "prescriptions": self.prescriptions,
            "warnings": self.warnings,
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
      None if tau_star - 1.0 >= k (needs alpha = 1.0 exactly for zero gain: impossible).
    """
    target = tau_star - 1.0
    if target <= 0.0:
        return 0.0
    if target >= float(k):
        return None  # tau* >= K+1: even alpha = 1 gives no net gain

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
    td_base = custom_td_ms if custom_td_ms is not None else profile.drafter_latency_td_ms
    if td_base is None or td_base < 0:
        raise ValueError(
            "Drafter latency t_d is unknown. Provide a drafter model or --td; "
            "refusing to substitute an unmeasured default."
        )
    is_autoregressive = (profile.drafter_type == "autoregressive")

    k_points: Dict[int, KAnalysisPoint] = {}
    best_k_75 = 0          # 0 == do not speculate
    max_speedup_75 = 0.0   # a K is recommended only if it beats baseline at the 75% reference alpha
    warnings: List[str] = []

    for k in k_range:
        # Batch size for verifying K tokens is typically M = K + 1 (K draft tokens + 1 target continuation)
        m_used = k + 1  # verifying K drafts + 1 bonus position
        if m_used not in profile.batch_curve:
            warnings.append(f"K={k} skipped: no M={m_used} measurement (refusing to substitute a different batch size).")
            continue

        t_v_m = profile.batch_curve[m_used].latency_ms
        delta_tv = t_v_m - t_v1

        # Drafter latency for K tokens
        td_k = (td_base * k) if is_autoregressive else td_base
        cycle_time = td_k + t_v_m

        # Model-free break-even required tokens per cycle
        tau_star = cycle_time / t_v1
        if tau_star < 1.0:
            warnings.append(
                f"K={k}: tau*={tau_star:.3f} < 1 means t_d + t_v(M={m_used}) < t_v(1); non-physical, "
                "likely a noisy or warm-up-contaminated M=1 measurement. Re-run before trusting this K."
            )

        # Break-Even acceptance rate under identical geometric rollout model
        alpha_be = solve_geometric_alpha_break_even(k, tau_star)
        can_accel = alpha_be is not None

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

        if can_accel and sp_75 > max_speedup_75:
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
    rec_pt = k_points.get(rec_k)
    crit_alpha = rec_pt.alpha_break_even if rec_pt else None
    crit_tau = rec_pt.tau_star if rec_pt else None
    think_loss = rec_pt.speedup_at_thinking_15_pct * 100.0 if rec_pt else None

    # Build actionable, non-contradictory prescriptions
    prescriptions = []
    if rec_pt is None:
        feasible = {k: pt.alpha_break_even for k, pt in k_points.items() if pt.can_ever_accelerate}
        if not feasible:
            prescriptions.append(
                "Critical: no tested draft length can break even (required tokens per cycle exceed 1+K for every K). "
                "Do not enable speculative decoding (K=0)."
            )
        else:
            kb = min(feasible, key=feasible.get)
            prescriptions.append(
                f"No tested draft length is net-positive at the 75% reference acceptance. Lowest break-even: K={kb} "
                f"at alpha >= {feasible[kb] * 100:.1f}%. Keep speculation off (K=0) unless measured acceptance exceeds that."
            )
    else:
        prescriptions.append(
            f"Break-even gate: requires >= {crit_tau:.2f} mean tokens per cycle for K={rec_k} "
            f"(geometric per-position alpha >= {crit_alpha * 100:.1f}%; equivalently >= {(crit_tau - 1) / rec_k * 100:.1f}% "
            "accepted/drafted, the ratio llama.cpp prints)."
        )
        prescriptions.append(
            f"Recommended draft length: --spec-draft-n-max {rec_k} "
            f"(projected {rec_pt.speedup_at_80_pct * 100:+.1f}% at assumed alpha=80%; best tested K at the 75% reference alpha)."
        )
        impossible_ks = [k for k, pt in k_points.items() if not pt.can_ever_accelerate]
        if impossible_ks:
            prescriptions.append(
                f"Disallowed draft lengths: K={', '.join(map(str, impossible_ks))} "
                "(required tokens per cycle exceed 1+K: net slowdown at any acceptance rate)."
            )
        prescriptions.append(
            f"Reasoning (<think>) mode: at an assumed, unmeasured alpha of 15%, K={rec_k} yields {think_loss:+.1f}%. "
            "Disable speculation (K=0) or use a runtime acceptance circuit-breaker while thinking."
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
        warnings=warnings,
    )
