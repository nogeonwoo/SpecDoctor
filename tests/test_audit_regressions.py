"""Regression tests for the adversarial-audit findings. Plain asserts + unittest.mock (no pytest fixtures)."""
import json, random, types
from unittest import mock

from spec_doctor.profiler import (ProfilingResult, BatchVerifyPoint, SpecProfiler,
                                  detect_batch_anomalies, _parse_json_array)
import spec_doctor.profiler as P
from spec_doctor.analyzer import analyze_speculative_dynamics, solve_geometric_alpha_break_even
from spec_doctor.reporter import SpecReporter

HW = {"chip_name": "Test", "gpu_cores": 10, "total_ram_gb": 24.0, "theoretical_bw_gbps": 100.0}

def _profile(lat, td=20.0, size_gb=7.19, dtype="dflash_parallel", sd=0.5):
    curve = {m: BatchVerifyPoint(m, l, m * 1000.0 / l if l else 0.0, l - lat[1], 0.0, 0.0, sd) for m, l in lat.items()}
    return ProfilingResult("t.gguf", "x", size_gb, lat[1], 1000.0 / lat[1] if lat[1] else 0.0, curve,
                           "d.gguf", td, dtype)

SAMPLE = {1: 128.94, 2: 139.52, 3: 279.49, 4: 225.32, 5: 801.85}

def test_solver_boundaries_and_residual():
    assert solve_geometric_alpha_break_even(3, 4.0) is None          # tau* = K+1 -> needs alpha = 1 for zero gain
    assert solve_geometric_alpha_break_even(3, 0.9) == 0.0           # tau* < 1
    rnd = random.Random(1)
    for k in range(1, 17):
        for _ in range(50):
            tau = 1.0 + rnd.uniform(1e-6, k - 1e-6)
            a = solve_geometric_alpha_break_even(k, tau)
            assert abs(sum(a ** i for i in range(1, k + 1)) - (tau - 1.0)) < 1e-9

def test_no_viable_k_recommends_zero_and_never_a_spec_command():
    rep = analyze_speculative_dynamics(_profile({1: 100.0, 2: 250.0, 3: 400.0}, td=150.0), HW, k_range=[1, 2])
    assert rep.recommended_k_max == 0
    assert rep.critical_alpha_threshold is None and rep.to_dict()["critical_alpha_threshold_pct"] is None
    txt = SpecReporter.render_terminal(rep)
    assert "--spec-draft-n-max" not in txt and "not recommended" in txt
    assert " *" not in "\n".join(l for l in txt.splitlines() if l.strip().startswith("K="))

def test_never_recommend_a_k_that_loses_at_reference_alpha():
    rep = analyze_speculative_dynamics(_profile({1: 100.0, 2: 155.0}, td=30.0), HW, k_range=[1])  # alpha_be = 85%
    assert rep.k_analysis[1].can_ever_accelerate is True
    assert rep.recommended_k_max == 0
    assert "+-" not in "\n".join(rep.prescriptions) + SpecReporter.render_terminal(rep)

def test_unknown_td_is_an_error_not_a_default():
    p = _profile({1: 100.0, 2: 110.0}, td=None)
    try:
        analyze_speculative_dynamics(p, HW, k_range=[1]); assert False, "expected ValueError"
    except ValueError as e:
        assert "t_d" in str(e)

def test_missing_batch_size_is_skipped_with_warning():
    rep = analyze_speculative_dynamics(_profile({1: 100.0, 2: 105.0, 3: 110.0, 4: 115.0, 5: 120.0}), HW, k_range=[3, 4, 5, 6])
    assert sorted(rep.k_analysis) == [3, 4]
    assert any("K=5 skipped" in w for w in rep.warnings) and any("K=6 skipped" in w for w in rep.warnings)

def test_tau_below_one_is_warned():
    rep = analyze_speculative_dynamics(_profile({1: 100.0, 2: 99.0}), HW, custom_td_ms=0.5, k_range=[1])
    assert any("non-physical" in w for w in rep.warnings)

def test_detector_matches_sample_and_is_cause_agnostic():
    curve = {m: BatchVerifyPoint(m, l, 0, 0, 0, 0, 0.5) for m, l in SAMPLE.items()}
    flags = detect_batch_anomalies(curve)
    assert any("M=4" in f and "faster than M=3" in f for f in flags)           # the real inversion
    assert any(f.startswith("M=5") and "independent M=1 passes" in f for f in flags)  # worse than serial
    assert not any("register" in f or "tiling" in f.split("Cause")[0] for f in flags)  # no asserted cause

def test_detector_quiet_on_healthy_curves():
    rnd = random.Random(7); fp = 0
    for _ in range(2000):                                   # flat memory-bound curve, 1% noise
        curve = {m: BatchVerifyPoint(m, 100.0 * (1 + rnd.gauss(0, 0.01)), 0, 0, 0, 0, 1.0) for m in range(1, 9)}
        fp += bool(detect_batch_anomalies(curve))
    assert fp / 2000 < 0.005, f"{fp}/2000 false positives on a healthy flat curve (old detector: ~49%)"
    linear = {m: BatchVerifyPoint(m, 100.0 * m, 0, 0, 0, 0, 0.5) for m in range(1, 9)}  # no weight reuse, no bug
    assert detect_batch_anomalies(linear) == []

def _fake_bench(rows, trailer=""):
    def run(cmd, *a, **k):
        if cmd and cmd[0] == "which":
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        data = [{"model_type": "fake", "model_size": 7_194_000_000, "n_prompt": m, "avg_ns": l * 1e6,
                 "stddev_ns": 1e5, "avg_ts": m * 1000.0 / l} for m, l in rows]
        return types.SimpleNamespace(returncode=0, stdout="metal init [noise]\n" + json.dumps(data) + trailer, stderr="")
    return run

def test_profiler_m1_handling_and_parsing():
    with mock.patch.object(P.subprocess, "run", _fake_bench([(2, 110.0), (3, 120.0), (1, 100.0)], "\ntrailing log\n")):
        r = SpecProfiler().profile_target_batch_curve("m.gguf")
    assert abs(r.batch_curve[2].delta_vs_m1_ms - 10.0) < 1e-9 and abs(r.batch_curve[3].delta_vs_m1_ms - 20.0) < 1e-9
    assert abs(r.target_model_size_gb - 7.194) < 1e-9                       # decimal GB
    with mock.patch.object(P.subprocess, "run", _fake_bench([(2, 110.0), (3, 120.0)])):
        try:
            SpecProfiler().profile_target_batch_curve("m.gguf"); assert False, "expected ValueError"
        except ValueError as e:
            assert "M=1" in str(e)

def test_td_probe_does_not_fabricate():
    with mock.patch.object(P.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="no timers here", stderr="")):
        try:
            SpecProfiler()._probe_dflash_td("t.gguf", "d.gguf"); assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "--td" in str(e)

def test_reporter_units_and_no_crash_on_zero_latency():
    rep = analyze_speculative_dynamics(_profile(SAMPLE, td=24.20), HW, k_range=[1, 2, 3, 4])
    txt = SpecReporter.render_terminal(rep)
    assert "14.33" in txt and "10.73" in txt and "6.24" in txt              # tokens/s == llama-bench pp-M t/s
    assert "GiB" in txt
    rep0 = analyze_speculative_dynamics(_profile({1: 100.0, 2: 0.0}), HW, k_range=[1])
    SpecReporter.render_terminal(rep0)                                       # must not raise ZeroDivisionError

def test_detector_quiet_at_realistic_noise_with_reported_stddev():
    rnd = random.Random(5); fp = 0
    for _ in range(2000):                                   # 3% per-point noise, stddev_ms reported honestly
        curve = {m: BatchVerifyPoint(m, 100.0 * (1 + rnd.gauss(0, 0.03)), 0, 0, 0, 0, 3.0) for m in range(1, 9)}
        fp += bool(detect_batch_anomalies(curve))
    assert fp / 2000 < 0.01, f"{fp}/2000 false positives at 3% noise"
