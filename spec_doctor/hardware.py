"""
Hardware Profiler: Apple Silicon & System Memory Bandwidth Diagnostics
"""

import os
import platform
import re
import subprocess
from dataclasses import dataclass
from typing import Optional, Dict, Any

APPLE_SILICON_BANDWIDTH_MAP = {
    # M1 Family
    r"Apple M1\b": 68.25,
    r"Apple M1 Pro": 200.0,
    r"Apple M1 Max": 400.0,
    r"Apple M1 Ultra": 800.0,
    # M2 Family
    r"Apple M2\b": 100.0,
    r"Apple M2 Pro": 200.0,
    r"Apple M2 Max": 400.0,
    r"Apple M2 Ultra": 800.0,
    # M3 Family
    r"Apple M3\b": 100.0,
    r"Apple M3 Pro": 150.0,
    r"Apple M3 Max": 400.0,
    # M4 Family
    r"Apple M4\b": 120.0,
    r"Apple M4 Pro": 273.0,
    r"Apple M4 Max": 546.0,
}

@dataclass
class HardwareProfile:
    platform_name: str
    chip_name: str
    is_apple_silicon: bool
    total_ram_gb: float
    gpu_cores: int
    theoretical_bw_gbps: float
    os_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform_name": self.platform_name,
            "chip_name": self.chip_name,
            "is_apple_silicon": self.is_apple_silicon,
            "total_ram_gb": round(self.total_ram_gb, 2),
            "gpu_cores": self.gpu_cores,
            "theoretical_bw_gbps": self.theoretical_bw_gbps,
            "os_version": self.os_version,
        }

def get_gpu_cores() -> int:
    try:
        out = subprocess.check_output(
            ["ioreg", "-r", "-c", "IOAccelerator"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="ignore")
        for line in out.splitlines():
            if "gpu-core-count" in line:
                val = int(line.split("=")[1].strip())
                if val > 0:
                    return val
    except Exception:
        pass
    return 10  # fallback default on base M3

def detect_hardware() -> HardwareProfile:
    plat = platform.system()
    chip = "Generic CPU/GPU"
    total_ram = 16.0
    is_as = False
    theoretical_bw = 100.0

    if plat == "Darwin":
        try:
            chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            ram_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip())
            total_ram = ram_bytes / (1024 ** 3)
            is_as = "Apple" in chip
        except Exception:
            pass

        gpu_cores = get_gpu_cores() if is_as else 0

        # Match theoretical memory bandwidth
        for pattern, bw in APPLE_SILICON_BANDWIDTH_MAP.items():
            if re.search(pattern, chip):
                theoretical_bw = bw
                break
    else:
        gpu_cores = 0

    return HardwareProfile(
        platform_name=plat,
        chip_name=chip,
        is_apple_silicon=is_as,
        total_ram_gb=total_ram,
        gpu_cores=gpu_cores,
        theoretical_bw_gbps=theoretical_bw,
        os_version=platform.mac_ver()[0] if plat == "Darwin" else platform.version(),
    )
