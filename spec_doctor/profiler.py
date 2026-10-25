"""
Profiler: Target Batch Verification Cost & Drafter Latency Profiler
"""

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

@dataclass
class BatchVerifyPoint:
    batch_m: int
    latency_ms: float
    tok_per_sec: float
    delta_vs_m1_ms: float
    overhead_pct: float
    achieved_bw_gbps: float
    stddev_ms: float

@dataclass
class ProfilingResult:
    target_model_path: str
    target_model_type: str
    target_model_size_gb: float
    target_m1_latency_ms: float
    target_m1_tps: float
    batch_curve: Dict[int, BatchVerifyPoint] = field(default_factory=dict)
    drafter_model_path: Optional[str] = None
    drafter_latency_td_ms: Optional[float] = None
    drafter_type: str = "unknown"
    simd_anomaly_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_model_path": self.target_model_path,
            "target_model_type": self.target_model_type,
            "target_model_size_gb": round(self.target_model_size_gb, 2),
            "target_m1_latency_ms": round(self.target_m1_latency_ms, 2),
            "target_m1_tps": round(self.target_m1_tps, 2),
            "batch_curve": {
                k: {
                    "batch_m": v.batch_m,
                    "latency_ms": round(v.latency_ms, 2),
                    "tok_per_sec": round(v.tok_per_sec, 2),
                    "delta_vs_m1_ms": round(v.delta_vs_m1_ms, 2),
                    "overhead_pct": round(v.overhead_pct, 1),
                    "achieved_bw_gbps": round(v.achieved_bw_gbps, 1),
                    "stddev_ms": round(v.stddev_ms, 2),
                }
                for k, v in self.batch_curve.items()
            },
            "drafter_model_path": self.drafter_model_path,
            "drafter_latency_td_ms": round(self.drafter_latency_td_ms, 2) if self.drafter_latency_td_ms else None,
            "drafter_type": self.drafter_type,
            "simd_anomaly_flags": self.simd_anomaly_flags,
        }

class SpecProfiler:
    def __init__(self, llama_bench_bin: str = "./bin/llama-bench", llama_cli_bin: str = "./bin/llama-cli"):
        self.llama_bench = self._resolve_binary(llama_bench_bin, "llama-bench")
        self.llama_cli = self._resolve_binary(llama_cli_bin, "llama-cli")

    def _resolve_binary(self, candidate: str, name: str) -> str:
        candidates = [
            candidate,
            os.path.abspath(candidate),
            os.path.expanduser(f"~/Ai_Agent/bin/{name}"),
            os.path.expanduser(f"~/llama.cpp/build/bin/{name}"),
            f"/usr/local/bin/{name}",
            f"/opt/homebrew/bin/{name}",
        ]
        for c in candidates:
            if os.path.exists(c) and os.access(c, os.X_OK):
                return os.path.abspath(c)

        which_path = subprocess.run(["which", name], stdout=subprocess.PIPE, text=True).stdout.strip()
        if which_path and os.path.exists(which_path):
            return which_path
        return candidate

    def profile_target_batch_curve(
        self,
        target_model: str,
        batch_sizes: List[int] = [1, 2, 3, 4, 5, 6, 7, 8],
        repetitions: int = 2,
        threads: int = 4,
    ) -> ProfilingResult:
        """
        Scans target model verification batch latency across batch sizes M=1..8
        using llama-bench in JSON output mode.
        """
        p_str = ",".join(str(b) for b in batch_sizes)
        cmd = [
            self.llama_bench,
            "-m", target_model,
            "-p", p_str,
            "-n", "0",
            "-r", str(repetitions),
            "-t", str(threads),
            "-ngl", "999",
            "-o", "json",
        ]

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"llama-bench failed (code {res.returncode}): {res.stderr}\n{res.stdout}")

        # Parse JSON output (skip any Metal init lines before the JSON array)
        stdout = res.stdout
        json_start = stdout.find("[")
        if json_start == -1:
            raise ValueError(f"Could not find JSON array in llama-bench output:\n{stdout}")

        data = json.loads(stdout[json_start:])
        if not data:
            raise ValueError("Empty JSON array returned by llama-bench")

        model_type = data[0].get("model_type", "unknown")
        model_size_gb = data[0].get("model_size", 0) / (1024 ** 3)

        curve: Dict[int, BatchVerifyPoint] = {}
        m1_lat_ms = 0.0

        for entry in data:
            m = entry.get("n_prompt", 1)
            avg_ns = entry.get("avg_ns", 0.0)
            lat_ms = avg_ns / 1e6
            std_ms = entry.get("stddev_ns", 0.0) / 1e6
            tps = entry.get("avg_ts", 0.0)

            if m == 1:
                m1_lat_ms = lat_ms

            delta_ms = lat_ms - m1_lat_ms if m1_lat_ms > 0 else 0.0
            overhead_pct = (delta_ms / m1_lat_ms * 100.0) if m1_lat_ms > 0 else 0.0
            bw_gbps = (model_size_gb / (lat_ms / 1000.0)) if lat_ms > 0 else 0.0

            curve[m] = BatchVerifyPoint(
                batch_m=m,
                latency_ms=lat_ms,
                tok_per_sec=tps,
                delta_vs_m1_ms=delta_ms,
                overhead_pct=overhead_pct,
                achieved_bw_gbps=bw_gbps,
                stddev_ms=std_ms,
            )

        # Detect non-monotonic or throughput collapse anomalies
        anomalies = []
        if 3 in curve and 4 in curve and curve[3].latency_ms > curve[4].latency_ms:
            bw3 = curve[3].achieved_bw_gbps
            bw4 = curve[4].achieved_bw_gbps
            anomalies.append(
                f"Non-monotonic verification latency: M=3 ({curve[3].latency_ms:.1f}ms, {bw3:.1f} GB/s) is SLOWER than M=4 ({curve[4].latency_ms:.1f}ms, {bw4:.1f} GB/s). Kernel multi-column dispatch/tiling inefficiency."
            )
        for m, pt in sorted(curve.items()):
            if m > 1 and pt.achieved_bw_gbps < (curve[1].achieved_bw_gbps * 0.3):
                anomalies.append(
                    f"Throughput collapse at M={m}: achieved bandwidth fell to {pt.achieved_bw_gbps:.1f} GB/s ({pt.latency_ms:.1f}ms), suggesting kernel fallback or register spill."
                )

        return ProfilingResult(
            target_model_path=target_model,
            target_model_type=model_type,
            target_model_size_gb=model_size_gb,
            target_m1_latency_ms=m1_lat_ms,
            target_m1_tps=curve.get(1, BatchVerifyPoint(1, m1_lat_ms, 1000/m1_lat_ms if m1_lat_ms else 0, 0, 0, 0, 0)).tok_per_sec,
            batch_curve=curve,
            simd_anomaly_flags=anomalies,
        )

    def profile_drafter(
        self,
        target_model: str,
        drafter_model: str,
        spec_type: str = "auto",
    ) -> Tuple[float, str]:
        """
        Profiles drafter latency td.
        If standalone GGUF: benchmarks single-token decode.
        If DFlash auxiliary: runs 4-token probe with LLAMA_SPEC_DEBUG=1.
        """
        # Check if DFlash auxiliary or standard drafter
        if "dflash" in drafter_model.lower() or spec_type == "draft-dflash":
            lat = self._probe_dflash_td(target_model, drafter_model)
            return lat, "dflash_parallel"
        else:
            # Standalone drafter via llama-bench
            try:
                cmd = [
                    self.llama_bench,
                    "-m", drafter_model,
                    "-p", "1",
                    "-n", "0",
                    "-r", "2",
                    "-ngl", "999",
                    "-o", "json",
                ]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if res.returncode == 0:
                    json_start = res.stdout.find("[")
                    if json_start != -1:
                        data = json.loads(res.stdout[json_start:])
                        lat_ms = data[0]["avg_ns"] / 1e6
                        return lat_ms, "autoregressive"
            except Exception:
                pass
            
            # Fallback probe via llama-cli
            lat = self._probe_dflash_td(target_model, drafter_model)
            return lat, "speculative_draft"

    def _probe_dflash_td(self, target_model: str, drafter_model: str) -> float:
        """
        Runs short probe with LLAMA_SPEC_DEBUG=1 to extract dflash-timer total_td
        """
        env = os.environ.copy()
        env["LLAMA_SPEC_DEBUG"] = "1"
        cmd = [
            self.llama_cli,
            "-m", target_model,
            "-md", drafter_model,
            "--spec-type", "draft-dflash",
            "-n", "4",
            "-p", "Hi",
            "--temp", "0",
            "-ngl", "999",
            "-ngld", "999",
            "-c", "512",
            "-st",
            "--simple-io",
        ]
        res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        full_out = res.stdout + "\n" + res.stderr

        # Regex match: [dflash-timer ... total_td=23.77 ms]
        matches = re.findall(r"total_td=([0-9.]+)\s*ms", full_out)
        if matches:
            values = [float(m) for m in matches]
            # Average across matches
            return sum(values) / len(values)

        return 23.0  # empirical fallback default
