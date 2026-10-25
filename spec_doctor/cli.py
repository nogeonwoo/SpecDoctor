"""
SpecDoctor CLI: Command-Line Interface
"""

import argparse
import os
import sys
from pathlib import Path

from .hardware import detect_hardware
from .profiler import SpecProfiler
from .analyzer import analyze_speculative_dynamics
from .reporter import SpecReporter

def main():
    parser = argparse.ArgumentParser(
        prog="spec-doctor",
        description="SpecDoctor: On-Device Speculative Decoding Roofline & Hardware Diagnostic Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  spec-doctor -m models/target.gguf -md models/draft.gguf
  spec-doctor -m models/target.gguf --td 22.5
  spec-doctor -m models/target.gguf --export-md spec_doctor_report.md
        """,
    )

    parser.add_argument("-m", "--target", type=str, required=True, help="Path to target GGUF model")
    parser.add_argument("-md", "--draft", type=str, default=None, help="Path to drafter GGUF model (optional)")
    parser.add_argument("--td", type=float, default=None, help="Explicit drafter latency td in ms (skips drafter probe)")
    parser.add_argument("--k-max", type=int, default=6, help="Maximum draft length K to analyze (default: 6)")
    parser.add_argument("-r", "--repetitions", type=int, default=2, help="Number of repetitions per batch size (default: 2)")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Number of CPU threads (default: 4)")
    parser.add_argument("--bench-bin", type=str, default="./bin/llama-bench", help="Path to llama-bench binary")
    parser.add_argument("--cli-bin", type=str, default="./bin/llama-cli", help="Path to llama-cli binary")
    parser.add_argument("--export-md", type=str, default=None, help="Export diagnostic report to Markdown file")
    parser.add_argument("--export-json", type=str, default=None, help="Export diagnostic report to JSON file")
    parser.add_argument("--json", action="store_true", help="Print report as raw JSON to stdout")

    args = parser.parse_args()

    target_path = os.path.abspath(args.target)
    if not os.path.exists(target_path):
        print(f"❌ Error: Target model file not found: {target_path}", file=sys.stderr)
        sys.exit(1)

    draft_path = os.path.abspath(args.draft) if args.draft else None
    if draft_path and not os.path.exists(draft_path):
        print(f"❌ Error: Drafter model file not found: {draft_path}", file=sys.stderr)
        sys.exit(1)

    print("🔎 [1/3] Detecting hardware & physical memory subsystem...", file=sys.stderr)
    hw_profile = detect_hardware()

    print(f"⚡ [2/3] Profiling target verification batch curve (M=1..{args.k_max + 1})...", file=sys.stderr)
    profiler = SpecProfiler(llama_bench_bin=args.bench_bin, llama_cli_bin=args.cli_bin)

    batch_sizes = list(range(1, args.k_max + 2))
    prof_result = profiler.profile_target_batch_curve(
        target_model=target_path,
        batch_sizes=batch_sizes,
        repetitions=args.repetitions,
        threads=args.threads,
    )

    td_ms = args.td
    if td_ms is None and draft_path:
        print("🎯 [3/3] Profiling drafter latency (td)...", file=sys.stderr)
        td_val, dft_type = profiler.profile_drafter(target_path, draft_path)
        prof_result.drafter_model_path = draft_path
        prof_result.drafter_latency_td_ms = td_val
        prof_result.drafter_type = dft_type
    elif td_ms is not None:
        prof_result.drafter_latency_td_ms = td_ms
        prof_result.drafter_type = "user_specified"

    # Analyze dynamics
    k_range = list(range(1, args.k_max + 1))
    report = analyze_speculative_dynamics(
        profile=prof_result,
        hardware_dict=hw_profile.to_dict(),
        custom_td_ms=td_ms,
        k_range=k_range,
    )

    if args.json:
        import json
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(SpecReporter.render_terminal(report))

    if args.export_md:
        SpecReporter.export_markdown(report, args.export_md)
        print(f"\n📄 Saved Markdown report to: {args.export_md}", file=sys.stderr)

    if args.export_json:
        SpecReporter.export_json(report, args.export_json)
        print(f"📊 Saved JSON report to: {args.export_json}", file=sys.stderr)

if __name__ == "__main__":
    main()
