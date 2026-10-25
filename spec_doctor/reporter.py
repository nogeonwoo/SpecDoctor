"""
Reporter: Terminal ASCII Dashboard & Markdown/JSON Report Generator
"""

import json
from pathlib import Path
from typing import Optional
from .analyzer import DiagnosticReport

class SpecReporter:
    @staticmethod
    def render_terminal(report: DiagnosticReport) -> str:
        hw = report.hardware_info
        tgt = report.target_info
        dft = report.drafter_info

        lines = []
        border = "=" * 80
        thin_border = "-" * 80

        lines.append(border)
        lines.append("[SpecDoctor] Speculative Decoding Roofline & Hardware Diagnostic")
        lines.append(border)

        # 1. Hardware Profile
        lines.append("[1. Hardware & Physical Memory Bandwidth]")
        lines.append(f"  • Device / Chip     : {hw.get('chip_name', 'Unknown')} ({hw.get('gpu_cores', 0)} GPU Cores)")
        lines.append(f"  • Total Unified RAM : {hw.get('total_ram_gb', 0):.1f} GB")
        lines.append(f"  • Theoretical Bus   : {hw.get('theoretical_bw_gbps', 0):.1f} GB/s Roofline ceiling")
        lines.append(f"  • OS / Platform     : {hw.get('platform_name', 'Darwin')} ({hw.get('os_version', '')})")
        lines.append("")

        # 2. Target & Drafter Profile
        lines.append("[2. Evaluated Models & Baseline Latency]")
        lines.append(f"  • Target Model      : {Path(tgt.get('model_path', '')).name}")
        lines.append(f"    - Type & Size     : {tgt.get('model_type', 'unknown')} | {tgt.get('model_size_gb', 0):.2f} GB ({tgt.get('model_size_gb', 0) * 1e9 / 1024**3:.2f} GiB)")
        lines.append(f"    - Baseline M=1    : {tgt.get('m1_latency_ms', 0):.2f} ms/token ({tgt.get('m1_tps', 0):.2f} tok/s)")
        if dft.get("model_path"):
            lines.append(f"  • Drafter Model     : {Path(dft.get('model_path', '')).name}")
        else:
            lines.append("  • Drafter Model     : (none; td supplied manually)")
        lines.append(f"    - Drafter Type    : {dft.get('drafter_type', 'unknown')}")
        lines.append(f"    - Drafter Cost td : {dft.get('td_ms', 0):.2f} ms")
        lines.append("")

        # 3. Target Verification Batch Scaling Table
        lines.append("[3. Target Batch Verification Cost, M = K+1 for each analysed K]")
        lines.append(
            f"  {'Batch (M)':<10} | {'Latency (ms)':<14} | {'Verify tok/s':<12} | "
            f"{'Eff. BW (S/t)':<14} | {'Delta vs M=1':<14} | {'Overhead (%)':<12}"
        )
        lines.append(
            f"  {'-'*10}-|-{'-'*14}-|-{'-'*12}-|-{'-'*14}-|-{'-'*14}-|-{'-'*12}"
        )

        model_sz = tgt.get('model_size_gb', 0.0)
        seen_batches = set()
        for k, pt in sorted(report.k_analysis.items()):
            if pt.batch_m_used in seen_batches:
                continue
            seen_batches.add(pt.batch_m_used)
            delta_str = f"{pt.delta_tv_ms:+.2f} ms"
            ovh_str = f"{(pt.delta_tv_ms / tgt.get('m1_latency_ms', 1.0) * 100):+.1f}%"
            bw_str = f"{(model_sz / (pt.t_v_ms / 1000.0)):.1f} GB/s" if pt.t_v_ms > 0 else "N/A"
            tps_str = f"{pt.batch_m_used * 1000.0 / pt.t_v_ms:.2f}" if pt.t_v_ms > 0 else "N/A"
            lines.append(
                f"  M={pt.batch_m_used:<8} | {pt.t_v_ms:<14.2f} | {tps_str:<12} | "
                f"{bw_str:<14} | {delta_str:<14} | {ovh_str:<12}"
            )

        lines.append("  Eff. BW = model_bytes / t_v(M): assumes one weight pass per verify step; a lower bound on bus traffic, not a counter reading.")
        if report.simd_anomalies:
            lines.append("")
            for anom in report.simd_anomalies:
                lines.append(f"  [Anomaly] {anom}")
        for w in report.warnings:
            lines.append(f"  [Warning] {w}")
        lines.append("")

        # 4. Break-Even Analysis Table
        lines.append("[4. Speculative Break-Even & Net Acceleration Matrix]")
        lines.append(
            f"  {'Draft (K)':<10} | {'Cycle Time':<12} | {'Req Tokens (tau*)':<18} | {'Break-Even alpha':<18} | "
            f"{'Speedup (alpha=70%)':<18} | {'Speedup (alpha=80%)':<18} | {'Think Mode (alpha=15%)':<20}"
        )
        lines.append(
            f"  {'-'*10}-|-{'-'*12}-|-{'-'*18}-|-{'-'*18}-|-{'-'*18}-|-{'-'*18}-|-{'-'*20}"
        )

        for k, pt in sorted(report.k_analysis.items()):
            be_str = f"{pt.alpha_break_even * 100:.1f}%" if pt.alpha_break_even is not None else "IMPOSSIBLE"
            tau_str = f"{pt.tau_star:.2f} tok/cyc"
            sp70_str = f"{pt.speedup_at_70_pct * 100:+.1f}%"
            sp80_str = f"{pt.speedup_at_80_pct * 100:+.1f}%"
            think_str = f"{pt.speedup_at_thinking_15_pct * 100:+.1f}%"

            marker = " *" if k == report.recommended_k_max else "  "
            lines.append(
                f"  K={k:<7}{marker} | {pt.cycle_time_ms:<9.2f} ms | {tau_str:<18} | {be_str:<18} | "
                f"{sp70_str:<18} | {sp80_str:<18} | {think_str:<20}"
            )
        lines.append("  (* = best tested K at the 75% reference alpha, shown only if net-positive there. tau* = cycle_time / t_v(1) needs no acceptance model; alpha = per-position acceptance under a geometric rollout)")
        lines.append("")

        # 5. Prescriptions & Actionable Verdict
        lines.append(thin_border)
        lines.append("[5. Diagnostic Summary & Guidelines]")
        lines.append(thin_border)
        for p in report.prescriptions:
            lines.append(f"  • {p}")
        lines.append("")

        # Recommended command line
        tgt_file = tgt.get('model_path', 'target.gguf')
        dft_file = dft.get('model_path', 'draft.gguf') or "draft.gguf"
        lines.append("[Recommended CLI Command]")
        if report.recommended_k_max <= 0:
            lines.append("  (speculative decoding not recommended for this configuration)")
            lines.append(f"  ./bin/llama-cli -m {tgt_file} -ngl 999 -t 4 --temp 0.0")
        else:
            lines.append(
                f"  ./bin/llama-cli -m {tgt_file} \\\n"
                f"                  -md {dft_file} \\\n"
                f"                  --spec-draft-n-max {report.recommended_k_max} \\\n"
                f"                  -ngl 999 -ngld 999 -t 4 --temp 0.0"
            )
        lines.append(border)

        return "\n".join(lines)

    @staticmethod
    def export_markdown(report: DiagnosticReport, output_path: str) -> None:
        text = SpecReporter.render_terminal(report)
        md_content = f"```text\n{text}\n```\n"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)

    @staticmethod
    def export_json(report: DiagnosticReport, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
