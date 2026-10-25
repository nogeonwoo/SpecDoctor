import pytest
from spec_doctor.profiler import ProfilingResult, BatchVerifyPoint
from spec_doctor.analyzer import analyze_speculative_dynamics, calculate_expected_tokens

def test_expected_tokens_geometric():
    # K=3, alpha=0.5
    # expected = 0.5^1 + 0.5^2 + 0.5^3 = 0.5 + 0.25 + 0.125 = 0.875
    val = calculate_expected_tokens(3, 0.5, model="geometric")
    assert pytest.approx(val, 0.001) == 0.875

def test_break_even_formula():
    # Setup mock profiling result
    # tv(1) = 100ms
    # tv(2) = 120ms (M=2) -> delta=20ms
    # tv(4) = 150ms (M=4) -> delta=50ms
    batch_curve = {
        1: BatchVerifyPoint(1, 100.0, 10.0, 0.0, 0.0, 0.5),
        2: BatchVerifyPoint(2, 120.0, 8.33, 20.0, 20.0, 0.5),
        4: BatchVerifyPoint(4, 150.0, 6.67, 50.0, 50.0, 0.5),
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
        k_range=[1, 3],
    )

    # For K=1: M=2 used. td=20, delta_tv=20.
    # alpha_be = (20 + 20) / (1 * 100) = 40 / 100 = 0.40 (40%)
    pt1 = report.k_analysis[1]
    assert pytest.approx(pt1.alpha_break_even, 0.01) == 0.40
    assert pt1.can_ever_accelerate is True

    # For K=3: M=4 used. td=20, delta_tv=50.
    # alpha_be = (20 + 50) / (3 * 100) = 70 / 300 = 0.2333 (23.3%)
    pt3 = report.k_analysis[3]
    assert pytest.approx(pt3.alpha_break_even, 0.001) == 0.2333
    assert pt3.can_ever_accelerate is True

def test_impossible_acceleration():
    # If drafter overhead is huge (e.g. td = 150ms > tv1 = 100ms)
    batch_curve = {
        1: BatchVerifyPoint(1, 100.0, 10.0, 0.0, 0.0, 0.5),
        2: BatchVerifyPoint(2, 130.0, 7.69, 30.0, 30.0, 0.5),
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
    # alpha_be = (150 + 30) / (1 * 100) = 1.8 > 1.0 -> Impossible!
    assert report.k_analysis[1].alpha_break_even > 1.0
    assert report.k_analysis[1].can_ever_accelerate is False

def test_simd_anomaly_detection():
    # M=3 latency is higher than M=4
    batch_curve = {
        1: BatchVerifyPoint(1, 100.0, 10.0, 0.0, 0.0, 0.5),
        2: BatchVerifyPoint(2, 120.0, 8.33, 20.0, 20.0, 0.5),
        3: BatchVerifyPoint(3, 250.0, 4.0, 150.0, 150.0, 0.5),
        4: BatchVerifyPoint(4, 200.0, 5.0, 100.0, 100.0, 0.5),
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
        simd_anomaly_flags=["Apple Silicon 3-Wide SIMD Penalty: M=3 is SLOWER than M=4"],
    )
    report = analyze_speculative_dynamics(
        profile=mock_profile,
        hardware_dict={"chip_name": "Apple M3"},
        custom_td_ms=20.0,
        k_range=[1, 2, 3],
    )
    assert len(report.simd_anomalies) > 0
    assert any("SIMD" in a for a in report.simd_anomalies)
