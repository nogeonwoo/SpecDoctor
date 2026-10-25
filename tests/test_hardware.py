from spec_doctor.hardware import detect_hardware, get_gpu_cores

def test_hardware_detection():
    hw = detect_hardware()
    assert hw.total_ram_gb > 0
    assert hw.platform_name in ["Darwin", "Linux", "Windows"]
    assert hw.theoretical_bw_gbps > 0
    if hw.is_apple_silicon:
        assert hw.gpu_cores > 0
