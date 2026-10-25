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

def _parse_json_array(stdout: str) -> List[Dict[str, Any]]:
    """Return the first valid JSON array of objects in stdout, tolerating text before AND after it."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(stdout):
        if ch != "[":
            continue
        try:
            obj, _ = dec.raw_decode(stdout[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj
    raise ValueError(f"Could not find a JSON array of results in llama-bench output:\n{stdout[:500]}")


def detect_batch_anomalies(curve: Dict[int, "BatchVerifyPoint"], rel_tol: float = 0.05) -> List[str]:
    """
    Cause-agnostic, noise-aware shape checks on the verification curve t_v(M).
      1. Inversion: a larger batch finishes materially faster than the next-smaller one.
      2. Worse-than-serial: t_v(M) > M * t_v(1), i.e. M independent single-token passes would be faster.
    Both are physically implausible for a healthy kernel whatever its roofline position. The tool cannot
    identify the cause (dispatch/tiling for specific column counts, register pressure, noise, ...).
    """
    flags: List[str] = []
    if 1 not in curve:
        return flags
    t1 = curve[1].latency_ms
    ms = sorted(curve)
    for a, b in zip(ms, ms[1:]):
        pa, pb = curve[a], curve[b]
        if pa.latency_ms <= 0:
            continue
        noise = 4.0 * ((pa.stddev_ms ** 2 + pb.stddev_ms ** 2) ** 0.5) / pa.latency_ms  # 4 sigma of the difference
        if pb.latency_ms < pa.latency_ms * (1.0 - max(rel_tol, noise)):
            flags.append(
                f"Non-monotonic verification latency: M={b} ({pb.latency_ms:.1f} ms) is "
                f"{(1.0 - pb.latency_ms / pa.latency_ms) * 100:.0f}% faster than M={a} ({pa.latency_ms:.1f} ms). "
                "Cause not identified by this tool; confirm with a same-architecture control and per-op timing."
            )
    for m in ms:
        if m <= 1:
            continue
        pt = curve[m]
        if pt.latency_ms > m * t1 * (1.0 + rel_tol):
            flags.append(
                f"M={m} ({pt.latency_ms:.1f} ms) is slower than {m} independent M=1 passes ({m * t1:.1f} ms): "
                "a per-column fallback would beat this path. Cause not identified by this tool."
            )
    return flags


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

        data = _parse_json_array(res.stdout)

        model_type = data[0].get("model_type", "unknown")
        model_size_gb = data[0].get("model_size", 0) / 1e9  # decimal GB, comparable to vendor GB/s figures

        curve: Dict[int, BatchVerifyPoint] = {}
        m1_rows = [e for e in data if e.get("n_prompt", 1) == 1 and e.get("avg_ns", 0) > 0]
        if not m1_rows:
            raise ValueError("llama-bench returned no usable M=1 (n_prompt=1) row; the M=1 baseline is required (include 1 in batch_sizes).")
        m1_lat_ms = m1_rows[0]["avg_ns"] / 1e6

        for entry in data:
            m = entry.get("n_prompt", 1)
            avg_ns = entry.get("avg_ns", 0.0)
            lat_ms = avg_ns / 1e6
            std_ms = entry.get("stddev_ns", 0.0) / 1e6
            tps = entry.get("avg_ts", 0.0)

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

        anomalies = detect_batch_anomalies(curve)

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

        raise RuntimeError(
            "Could not parse 'total_td=<ms>' from llama-cli output. This needs a build that supports "
            "--spec-type draft-dflash and prints dflash-timer lines under LLAMA_SPEC_DEBUG=1. "
            "Pass the drafter cost explicitly with --td instead; no default is substituted."
        )
