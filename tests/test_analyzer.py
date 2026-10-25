import pytest
from spec_doctor.profiler import ProfilingResult, BatchVerifyPoint
from spec_doctor.analyzer import (
    analyze_speculative_dynamics,
    calculate_expected_tokens,
    solve_geometric_alpha_break_even,
)

def test_expected_tokens_geometric():
    # K=3, alpha=0.5
    # expected = 0.5^1 + 0.5^2 + 0.5^3 = 0.5 + 0.25 + 0.125 = 0.875
    val = calculate_expected_tokens(3, 0.5, model="geometric")
    assert pytest.approx(val, 0.001) == 0.875

def test_break_even_speedup_consistency():
    """
    CRITICAL REGRESSION TEST (Defect 1.1):
    Assert that the break-even acceptance rate alpha_be MUST yield EXACTLY
    0.0% net speedup under the model's own evaluation function.
    """
    batch_curve = {
        1: BatchVerifyPoint(1, 100.0, 10.0, 0.0, 0.0, 70.0, 0.5),
        2: BatchVerifyPoint(2, 120.0, 8.33, 20.0, 20.0, 58.3, 0.5),
        3: BatchVerifyPoint(3, 140.0, 7.14, 40.0, 40.0, 50.0, 0.5),
        4: BatchVerifyPoint(4, 150.0, 6.67, 50.0, 50.0, 46.7, 0.5),
    }

    mock_profile = ProfilingResult(
        target_model_path="dummy.gguf",
        target_model_type="test-model",
        target_model_size_gb=7.0,
        target_m1_latency_ms=100.0,
        target_m1_tps=10.0,
        batch_curve=batch_curve,
        drafter_latency_td_ms=20.0,
        drafter_type="dflash_parallel",
    )

    report = analyze_speculative_dynamics(
        profile=mock_profile,
        hardware_dict={"chip_name": "Test M3", "gpu_cores": 10},
        custom_td_ms=20.0,
        k_range=[1, 2, 3],
    )

    tps_base = 1000.0 / 100.0

    for k, pt in report.k_analysis.items():
        if pt.can_ever_accelerate:
            assert pt.alpha_break_even is not None
            # Evaluate speedup at alpha_break_even
            e_tok = calculate_expected_tokens(k, pt.alpha_break_even, model="geometric")
            tps_at_be = ((1.0 + e_tok) / pt.cycle_time_ms) * 1000.0
            net_speedup = (tps_at_be / tps_base) - 1.0
            # Net speedup must be exactly 0.0
            assert pytest.approx(net_speedup, abs=1e-3) == 0.0

def test_m3_27b_fixture_regression():
    """
    Regression test against the exact M3 27B physical measurement fixture.
    Resolves Sonnet 5.5 Defect 1.1 and validates tau* vs alpha_be.
    """
    tv1 = 117.35
    td = 23.97
    batch_curve = {
        1: BatchVerifyPoint(1, 117.35, 8.52, 0.0, 0.0, 57.1, 0.5),
        2: BatchVerifyPoint(2, 137.78, 7.26, 20.43, 17.4, 48.6, 0.5),
        3: BatchVerifyPoint(3, 284.79, 3.51, 167.44, 142.7, 23.5, 0.5),
        4: BatchVerifyPoint(4, 228.11, 4.38, 110.76, 94.4, 29.4, 0.5),
        5: BatchVerifyPoint(5, 809.03, 1.24, 691.68, 589.4, 8.3, 0.5),
    }

    mock_profile = ProfilingResult(
        target_model_path="Ternary-Bonsai-2-27B-PQ2_0.gguf",
        target_model_type="qwen35 27B",
        target_model_size_gb=6.70,
        target_m1_latency_ms=tv1,
        target_m1_tps=8.52,
        batch_curve=batch_curve,
        drafter_latency_td_ms=td,
        drafter_type="dflash_parallel",
    )

    report = analyze_speculative_dynamics(
        profile=mock_profile,
        hardware_dict={"chip_name": "Apple M3", "gpu_cores": 10},
        custom_td_ms=td,
        k_range=[1, 2, 3, 4],
    )

    # K=1: M=2 used. Cycle = 23.97 + 137.78 = 161.75ms. tau* = 161.75 / 117.35 = 1.378
    pt1 = report.k_analysis[1]
    assert pytest.approx(pt1.tau_star, 0.01) == 1.378
    assert pytest.approx(pt1.alpha_break_even, 0.01) == 0.378
    assert pt1.can_ever_accelerate is True

    # K=3: M=4 used. Cycle = 23.97 + 228.11 = 252.08ms. tau* = 252.08 / 117.35 = 2.148
    # alpha_be under geometric model is ~59.2% (NOT the buggy 38.3% from linear model!)
    pt3 = report.k_analysis[3]
    assert pytest.approx(pt3.tau_star, 0.01) == 2.148
    assert pytest.approx(pt3.alpha_break_even, 0.01) == 0.592
    assert pt3.can_ever_accelerate is True

    # Validate that at alpha=38.3% (the old erroneous value), speedup is indeed ~ -26%
    e_tok_old = calculate_expected_tokens(3, 0.383, model="geometric")
    tps_old = ((1.0 + e_tok_old) / 252.08) * 1000.0
    speedup_old = (tps_old / 8.52) - 1.0
    assert pytest.approx(speedup_old, 0.02) == -0.26

    # K=4: M=5 used. Cycle = 23.97 + 809.03 = 833.00ms. tau* = 833.00 / 117.35 = 7.098
    # Since max possible tokens per cycle is 1 + 4 = 5 < 7.098, it is IMPOSSIBLE!
    pt4 = report.k_analysis[4]
    assert pt4.tau_star > 5.0
    assert pt4.can_ever_accelerate is False
    assert pt4.alpha_break_even is None

def test_impossible_acceleration():
    batch_curve = {
        1: BatchVerifyPoint(1, 100.0, 10.0, 0.0, 0.0, 70.0, 0.5),
        2: BatchVerifyPoint(2, 250.0, 4.0, 150.0, 150.0, 28.0, 0.5),
    }
    mock_profile = ProfilingResult(
        target_model_path="dummy.gguf",
        target_model_type="test-model",
        target_model_size_gb=7.0,
        target_m1_latency_ms=100.0,
        target_m1_tps=10.0,
        batch_curve=batch_curve,
        drafter_latency_td_ms=150.0,
        drafter_type="dflash_parallel",
    )
    report = analyze_speculative_dynamics(
        profile=mock_profile,
        hardware_dict={"chip_name": "Test M3"},
        custom_td_ms=150.0,
        k_range=[1],
    )
    assert report.k_analysis[1].can_ever_accelerate is False
    assert report.k_analysis[1].alpha_break_even is None

def test_anomaly_detection():
    batch_curve = {
        1: BatchVerifyPoint(1, 100.0, 10.0, 0.0, 0.0, 70.0, 0.5),
        2: BatchVerifyPoint(2, 120.0, 8.33, 20.0, 20.0, 58.3, 0.5),
        3: BatchVerifyPoint(3, 250.0, 4.0, 150.0, 150.0, 28.0, 0.5),
        4: BatchVerifyPoint(4, 200.0, 5.0, 100.0, 100.0, 35.0, 0.5),
    }
    mock_profile = ProfilingResult(
        target_model_path="dummy.gguf",
        target_model_type="test-model",
        target_model_size_gb=7.0,
        target_m1_latency_ms=100.0,
        target_m1_tps=10.0,
        batch_curve=batch_curve,
        drafter_latency_td_ms=20.0,
        drafter_type="dflash_parallel",
        simd_anomaly_flags=["Non-monotonic verification latency: M=3 is SLOWER than M=4."],
    )
    report = analyze_speculative_dynamics(
        profile=mock_profile,
        hardware_dict={"chip_name": "Apple M3"},
        custom_td_ms=20.0,
        k_range=[1, 2, 3],
    )
    assert len(report.simd_anomalies) > 0
