#!/usr/bin/env python3
"""
SRT Speculation Analysis CLI Tool.

A unified command-line interface for analyzing SRT speculation performance
from rollout data. Supports auto-detection of available data, running
simulation sweeps, generating figures, and creating analysis reports.

Usage:
    # Show info about a data directory (auto-detect ticks, structure)
    srt_analyze info /path/to/rollout_data

    # Run full analysis with auto-detection
    srt_analyze full /path/to/rollout_data -o ./results

    # Run sweep only, custom tick range
    srt_analyze sweep /path/to/rollout_data --tick-start 1 --tick-end 50 --tick-step 10

    # Generate figures from existing CSV data (no re-run needed)
    srt_analyze plot --data ./results/per_request_data.csv -o ./figures

    # Single tick simulation
    srt_analyze single /path/to/rollout_data --cache-tick 5 --sim-tick 6

    # Analyze within-prompt output length variance
    srt_analyze variance /path/to/rollout_data

    # Analyze runahead prediction with plots
    srt_analyze prediction /path/to/rollout_data --plot --compare-methods
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parents[4]))


def cmd_info(args):
    """Show information about a data directory."""
    from recipe.srt.scripts.rollout_analysis.data_discovery import (
        analyze_response_lengths,
        count_samples_in_tick,
        discover_data_directory,
        get_sample_data,
    )

    data_dir = Path(args.data_dir)
    info = discover_data_directory(data_dir)

    print(info.summary())

    if not info.valid:
        return 1

    # Show additional details if requested
    if args.detailed:
        print("\n--- Detailed Statistics ---")

        # Sample counts per tick
        if info.rollout_ticks:
            tick = info.rollout_ticks[0]
            count = count_samples_in_tick(info, tick, "rollout")
            print(f"\nSample count in rollout/{tick}.jsonl: {count}")

        if info.secondary_ticks:
            tick = info.secondary_ticks[0]
            count = count_samples_in_tick(info, tick, "secondary")
            print(f"Sample count in secondary/{tick}.jsonl: {count}")

        # Response length analysis
        if info.rollout_ticks:
            tick = info.rollout_ticks[0]
            stats = analyze_response_lengths(info, tick)
            if stats:
                print(f"\nResponse length analysis (tick {tick}, estimated tokens):")
                for k, v in stats.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.1f}")
                    else:
                        print(f"  {k}: {v}")

        # Sample data preview
        if args.preview:
            print("\n--- Sample Data Preview ---")
            if info.rollout_ticks:
                tick = info.rollout_ticks[0]
                samples = get_sample_data(info, tick, "rollout", max_samples=2)
                if samples:
                    print(f"\nRollout/{tick}.jsonl sample:")
                    for i, s in enumerate(samples):
                        print(f"  [{i}] keys: {list(s.keys())}")
                        if "input" in s:
                            print(f"      input: {s['input'][:100]}...")
                        if "output" in s:
                            print(f"      output length: {len(s.get('output', ''))}")

    return 0


def cmd_sweep(args):
    """Run simulation sweep across ticks."""
    from recipe.srt.scripts.rollout_analysis.analysis_config import SweepConfig
    from recipe.srt.scripts.rollout_analysis.sweep_runner import SweepRunner

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)

    # Determine modes to run
    if args.prefill_only:
        run_prefill, run_online, run_combined = True, False, False
    elif args.online_only:
        run_prefill, run_online, run_combined = False, True, False
    elif args.combined_only:
        run_prefill, run_online, run_combined = False, False, True
    else:
        # Default: run all modes
        run_prefill = args.run_prefill
        run_online = args.run_online
        run_combined = args.run_combined

    config = SweepConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        model_path=args.model,
        tick_start=args.tick_start,
        tick_end=args.tick_end,
        tick_step=args.tick_step,
        run_prefill_only=run_prefill,
        run_online_only=run_online,
        run_prefill_plus_online=run_combined,
        min_token_prob=args.min_token_prob,
        hash_token_count=args.hash_token_count,
        max_samples=args.max_samples,
        verbose=args.verbose,
    )

    runner = SweepRunner(config)
    results = runner.run()
    runner.save_results(results)
    runner.print_summary(results, min_response_len=args.min_response_len)

    print(f"\nSweep complete! Total simulations: {results.total_simulations}")
    print(f"Results saved to: {output_dir}")

    return 0


def cmd_plot(args):
    """Generate figures from existing CSV data."""
    from recipe.srt.scripts.rollout_analysis.analysis_config import FigureConfig
    from recipe.srt.scripts.rollout_analysis.figure_generator import FigureGenerator

    # Handle --list first (doesn't need data file)
    if args.list:
        print("Available figures:")
        for name in FigureConfig.available_figures():
            print(f"  - {name}")
        return 0

    data_csv = Path(args.data)
    output_dir = Path(args.output)

    if not data_csv.exists():
        print(f"Error: Data file not found: {data_csv}")
        return 1

    # Parse figure list
    figures = None
    if args.figures:
        figures = [f.strip() for f in args.figures.split(",")]

    config = FigureConfig(
        data_csv=data_csv,
        output_dir=output_dir,
        figures=figures,
        min_response_len=args.min_response_len,
        dpi=args.dpi,
        format=args.format,
    )

    generator = FigureGenerator(config)

    try:
        paths = generator.generate_all()
        print(f"\nGenerated {len(paths)} figures in {output_dir}")
    except Exception as e:
        print(f"Error generating figures: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


def cmd_full(args):
    """Run full analysis: sweep + plot + report."""
    from recipe.srt.scripts.rollout_analysis.analysis_config import FigureConfig, SweepConfig
    from recipe.srt.scripts.rollout_analysis.figure_generator import FigureGenerator
    from recipe.srt.scripts.rollout_analysis.sweep_runner import SweepRunner

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)

    # Step 1: Run sweep
    print("=" * 70)
    print("STEP 1: Running simulation sweep")
    print("=" * 70)

    sweep_config = SweepConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        model_path=args.model,
        tick_start=args.tick_start,
        tick_end=args.tick_end,
        tick_step=args.tick_step,
        run_prefill_only=True,
        run_online_only=True,
        run_prefill_plus_online=True,
        min_token_prob=args.min_token_prob,
        verbose=args.verbose,
    )

    runner = SweepRunner(sweep_config)
    results = runner.run()
    runner.save_results(results)
    runner.print_summary(results, min_response_len=args.min_response_len)

    # Step 2: Generate figures
    print("\n" + "=" * 70)
    print("STEP 2: Generating figures")
    print("=" * 70)

    csv_path = output_dir / "per_request_data.csv"
    figures_dir = output_dir / "figures"

    if csv_path.exists():
        fig_config = FigureConfig(
            data_csv=csv_path,
            output_dir=figures_dir,
            min_response_len=args.min_response_len,
        )

        generator = FigureGenerator(fig_config)
        try:
            paths = generator.generate_all()
            print(f"Generated {len(paths)} figures in {figures_dir}")
        except Exception as e:
            print(f"Warning: Error generating figures: {e}")
    else:
        print(f"Warning: No per-request data found at {csv_path}")

    # Step 3: Generate report
    print("\n" + "=" * 70)
    print("STEP 3: Generating analysis report")
    print("=" * 70)

    report_path = output_dir / "ANALYSIS_REPORT.md"
    _generate_report(output_dir, report_path, args.min_response_len)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results directory: {output_dir}")
    print(f"  - sweep_summary.json: Aggregated metrics")
    print(f"  - per_request_data.csv: Per-request data")
    print(f"  - figures/: Visualization figures")
    print(f"  - ANALYSIS_REPORT.md: Summary report")

    return 0


def cmd_variance(args):
    """Analyze within-prompt output length variance."""
    from recipe.srt.scripts.rollout_analysis.analyze_lengths import (
        analyze_within_prompt_variance,
        load_primary_data,
        print_variance_results,
    )
    from transformers import AutoTokenizer

    data_dir = Path(args.data_dir)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    print("\nLoading primary data...")
    primary_data = load_primary_data(data_dir, tokenizer, args.verbose)

    print("Analyzing within-prompt variance...")
    variance_analysis = analyze_within_prompt_variance(primary_data, args.min_samples)

    print_variance_results(variance_analysis)

    if args.output_json:
        if 'error' not in variance_analysis:
            results = {
                'cv_distribution': variance_analysis['cv_distribution'],
                'range_distribution': variance_analysis['range_distribution'],
                'variance_categories': variance_analysis['variance_categories'],
                'variance_accuracy': variance_analysis['variance_accuracy'],
                'cv_length_correlation': variance_analysis['cv_length_correlation'],
                'correct_vs_incorrect': variance_analysis['correct_vs_incorrect'],
            }

            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_path}")

    return 0


def cmd_prediction(args):
    """Analyze runahead prediction correlation."""
    from recipe.srt.scripts.rollout_analysis.analyze_runahead_prediction import (
        analyze_runahead_correlation,
        analyze_same_step_correlation,
        compare_correlation_methods,
        load_data_by_step,
        plot_all_steps_grid,
        plot_correlation_over_steps,
        plot_scatter_examples,
        print_results,
        MATPLOTLIB_AVAILABLE,
    )
    from transformers import AutoTokenizer

    data_dir = Path(args.data_dir)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    print("\nLoading data by step...")
    primary_by_step, secondary_by_step = load_data_by_step(data_dir, tokenizer, args.verbose)

    print("\nAnalyzing same-step correlation...")
    same_step_analysis = analyze_same_step_correlation(
        primary_by_step, secondary_by_step, min_samples=8
    )

    print("Analyzing runahead prediction (secondary[N] -> primary[N+1])...")
    runahead_analysis = analyze_runahead_correlation(
        primary_by_step, secondary_by_step, min_samples=args.min_samples
    )

    print_results(same_step_analysis, runahead_analysis)

    # Method comparison
    if args.compare_methods:
        print("\nComparing correlation methods...")
        method_comparison = compare_correlation_methods(
            primary_by_step, secondary_by_step, min_samples=args.min_samples
        )
        if 'error' not in method_comparison:
            print("\n" + "=" * 80)
            print("CORRELATION METHOD COMPARISON")
            print("=" * 80)
            print(f"\nMethod 1 (Mean per prompt): Overall avg r = {method_comparison['grouped_avg']:.3f}")
            print(f"Method 2 (Individual pairs): Overall avg r = {method_comparison['individual_avg']:.3f}")
            print(f"\nInterpretation:")
            print(f"  {method_comparison['interpretation']}")

    # Generate plots
    if args.plot:
        output_dir = Path(args.output_dir)
        print(f"\nGenerating plots in {output_dir}...")

        step_correlations = runahead_analysis.get('step_correlations', [])

        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available, cannot generate plots")
        elif step_correlations:
            path1 = plot_correlation_over_steps(step_correlations, output_dir, args.dpi)
            if path1:
                print(f"  Generated: {path1}")

            path2 = plot_scatter_examples(
                step_correlations, primary_by_step, secondary_by_step,
                output_dir, args.dpi, args.min_samples
            )
            if path2:
                print(f"  Generated: {path2}")

            path3 = plot_all_steps_grid(
                step_correlations, primary_by_step, secondary_by_step,
                output_dir, args.dpi, args.min_samples
            )
            if path3:
                print(f"  Generated: {path3}")
        else:
            print("  No step correlations available for plotting")

    return 0


def cmd_single(args):
    """Run single tick simulation."""
    import sys
    sys.path.insert(0, str(Path(__file__).parents[3]))
    from recipe.srt.replay_simulator import SimulatorConfig, print_results, run_simulation

    config = SimulatorConfig(
        model_path=args.model,
        data_dir=args.data_dir,
        mode="parallel",
        cache_tick=args.cache_tick,
        sim_tick=args.sim_tick,
        min_token_prob=args.min_token_prob,
        hash_token_count=args.hash_token_count,
        online_update=args.online_update,
        skip_prefill=args.skip_prefill,
        max_samples=args.max_samples,
        verbose=args.verbose,
    )

    result = run_simulation(config)
    print_results(result)

    # Save result if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save without the per-request details for summary
        summary = {k: v for k, v in result.items() if k != "requests"}
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved result to {output_path}")

    return 0


def _generate_report(output_dir: Path, report_path: Path, min_response_len: int = 4000):
    """Generate markdown analysis report."""
    import pandas as pd

    csv_path = output_dir / "per_request_data.csv"
    summary_path = output_dir / "sweep_summary.json"

    if not csv_path.exists():
        print(f"Cannot generate report: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    df_filtered = df[df["response_len"] >= min_response_len]

    # Calculate mode statistics
    mode_stats = {}
    for mode in ["prefill_only", "online_only", "prefill_plus_online"]:
        subset = df_filtered[df_filtered["mode"] == mode]
        if len(subset) > 0:
            mode_stats[mode] = {
                "count": len(subset),
                "tps": subset["tokens_per_step"].mean(),
                "hr": subset["hit_rate"].mean(),
                "ar": subset["acceptance_rate"].mean(),
                "tphs": subset["tokens_per_hit_step"].mean(),
                "dc": subset["draft_contribution"].mean(),
            }

    mode_labels = {
        "prefill_only": "Prefill Only",
        "online_only": "Online Only",
        "prefill_plus_online": "Prefill + Online",
    }

    # Generate report
    lines = [
        "# SRT Speculation Analysis Report",
        "",
        f"**Generated from:** `{output_dir}`",
        f"**Long sequence threshold:** {min_response_len} tokens",
        f"**Total samples:** {len(df)} ({len(df_filtered)} long sequences)",
        "",
        "## Summary Statistics",
        "",
        "| Mode | E2E Speedup | Hit Rate | Accept Rate | Draft Contrib |",
        "|------|-------------|----------|-------------|---------------|",
    ]

    for mode, label in mode_labels.items():
        if mode in mode_stats:
            s = mode_stats[mode]
            lines.append(
                f"| {label} | {s['tps']:.2f}x | {s['hr']:.1%} | {s['ar']:.1%} | {s['dc']:.1%} |"
            )

    lines.extend([
        "",
        "## Key Findings",
        "",
    ])

    # Add findings based on data
    if "prefill_only" in mode_stats and "online_only" in mode_stats:
        prefill_tps = mode_stats["prefill_only"]["tps"]
        online_tps = mode_stats["online_only"]["tps"]
        if online_tps > prefill_tps:
            improvement = (online_tps - prefill_tps) / prefill_tps * 100
            lines.append(f"1. **Online updates provide {improvement:.1f}% better speedup** than prefill alone for long sequences")
        lines.append(f"2. Prefill has higher hit rate ({mode_stats['prefill_only']['hr']:.1%}) but lower acceptance ({mode_stats['prefill_only']['ar']:.1%})")
        lines.append(f"3. Online has lower hit rate ({mode_stats['online_only']['hr']:.1%}) but higher acceptance ({mode_stats['online_only']['ar']:.1%})")

    if "prefill_plus_online" in mode_stats:
        lines.append(f"4. Combined mode achieves best speedup: {mode_stats['prefill_plus_online']['tps']:.2f}x")

    lines.extend([
        "",
        "## Recommendations",
        "",
        "- **Always enable online updates** for long sequences (> 4K tokens)",
        "- **Use combined mode** (prefill + online) for best results",
        "- **Focus on acceptance rate** over hit rate for E2E speedup",
        "",
        "## Figures",
        "",
    ])

    # List generated figures
    figures_dir = output_dir / "figures"
    if figures_dir.exists():
        for fig_file in sorted(figures_dir.glob("*.png")):
            lines.append(f"- [{fig_file.stem}](figures/{fig_file.name})")

    lines.extend([
        "",
        "---",
        "*Generated by srt_analyze CLI tool*",
    ])

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated report: {report_path}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="SRT Speculation Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show info about a data directory
  srt_analyze info /path/to/rollout_data

  # Run full analysis
  srt_analyze full /path/to/rollout_data -o ./results

  # Run sweep with custom tick range
  srt_analyze sweep /path/to/rollout_data --tick-start 1 --tick-end 50

  # Generate figures from existing CSV
  srt_analyze plot --data ./results/per_request_data.csv -o ./figures

  # Single tick simulation
  srt_analyze single /path/to/rollout_data --cache-tick 5 --sim-tick 6

  # Analyze within-prompt variance
  srt_analyze variance /path/to/rollout_data

  # Analyze runahead prediction with plots
  srt_analyze prediction /path/to/rollout_data --plot --compare-methods
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # =========================================================================
    # Info command
    # =========================================================================
    info_parser = subparsers.add_parser("info", help="Show data directory info")
    info_parser.add_argument("data_dir", type=str, help="Path to rollout data directory")
    info_parser.add_argument("-d", "--detailed", action="store_true",
                             help="Show detailed statistics")
    info_parser.add_argument("-p", "--preview", action="store_true",
                             help="Preview sample data")
    info_parser.set_defaults(func=cmd_info)

    # =========================================================================
    # Sweep command
    # =========================================================================
    sweep_parser = subparsers.add_parser("sweep", help="Run simulation sweep")
    sweep_parser.add_argument("data_dir", type=str, help="Path to rollout data directory")
    sweep_parser.add_argument("-o", "--output", type=str, default="./sweep_results",
                              help="Output directory (default: ./sweep_results)")
    sweep_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B",
                              help="Model path for tokenizer")
    sweep_parser.add_argument("--tick-start", type=int, default=None,
                              help="Start tick (auto-detect if not specified)")
    sweep_parser.add_argument("--tick-end", type=int, default=None,
                              help="End tick (auto-detect if not specified)")
    sweep_parser.add_argument("--tick-step", type=int, default=5,
                              help="Step between ticks (default: 5)")
    sweep_parser.add_argument("--min-token-prob", type=float, default=0.3,
                              help="Minimum probability for draft tokens")
    sweep_parser.add_argument("--hash-token-count", type=int, default=128,
                              help="Tokens to hash for tree sharing")
    sweep_parser.add_argument("--max-samples", type=int, default=0,
                              help="Maximum samples per tick (0 = all)")
    sweep_parser.add_argument("--min-response-len", type=int, default=0,
                              help="Minimum response length for summary (0 = all)")
    # Mode selection
    sweep_parser.add_argument("--prefill-only", action="store_true",
                              help="Only run prefill-only mode")
    sweep_parser.add_argument("--online-only", action="store_true",
                              help="Only run online-only mode")
    sweep_parser.add_argument("--combined-only", action="store_true",
                              help="Only run combined mode")
    sweep_parser.add_argument("--run-prefill", action="store_true", default=True,
                              help="Run prefill-only mode")
    sweep_parser.add_argument("--run-online", action="store_true", default=True,
                              help="Run online-only mode")
    sweep_parser.add_argument("--run-combined", action="store_true", default=True,
                              help="Run combined mode")
    sweep_parser.add_argument("-v", "--verbose", action="store_true",
                              help="Verbose output")
    sweep_parser.set_defaults(func=cmd_sweep)

    # =========================================================================
    # Plot command
    # =========================================================================
    plot_parser = subparsers.add_parser("plot", help="Generate figures from CSV data")
    plot_parser.add_argument("--data", "-d", type=str, required=True,
                             help="Path to per_request_data.csv")
    plot_parser.add_argument("-o", "--output", type=str, default="./figures",
                             help="Output directory for figures")
    plot_parser.add_argument("--figures", type=str, default=None,
                             help="Comma-separated list of figures to generate (default: all)")
    plot_parser.add_argument("--min-response-len", type=int, default=4000,
                             help="Minimum response length for filtering (default: 4000)")
    plot_parser.add_argument("--dpi", type=int, default=150,
                             help="Figure DPI (default: 150)")
    plot_parser.add_argument("--format", type=str, default="png",
                             choices=["png", "pdf", "svg"],
                             help="Figure format (default: png)")
    plot_parser.add_argument("--list", "-l", action="store_true",
                             help="List available figures and exit")
    plot_parser.add_argument("-v", "--verbose", action="store_true",
                             help="Verbose output")
    plot_parser.set_defaults(func=cmd_plot)

    # =========================================================================
    # Full command
    # =========================================================================
    full_parser = subparsers.add_parser("full", help="Run full analysis (sweep + plot + report)")
    full_parser.add_argument("data_dir", type=str, help="Path to rollout data directory")
    full_parser.add_argument("-o", "--output", type=str, default="./analysis_results",
                             help="Output directory")
    full_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B",
                             help="Model path for tokenizer")
    full_parser.add_argument("--tick-start", type=int, default=None,
                             help="Start tick (auto-detect if not specified)")
    full_parser.add_argument("--tick-end", type=int, default=None,
                             help="End tick (auto-detect if not specified)")
    full_parser.add_argument("--tick-step", type=int, default=5,
                             help="Step between ticks")
    full_parser.add_argument("--min-token-prob", type=float, default=0.3,
                             help="Minimum probability for draft tokens")
    full_parser.add_argument("--min-response-len", type=int, default=4000,
                             help="Minimum response length for analysis focus")
    full_parser.add_argument("-v", "--verbose", action="store_true",
                             help="Verbose output")
    full_parser.set_defaults(func=cmd_full)

    # =========================================================================
    # Variance command
    # =========================================================================
    variance_parser = subparsers.add_parser("variance", help="Analyze within-prompt output length variance")
    variance_parser.add_argument("data_dir", type=str, help="Path to rollout data directory")
    variance_parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                                 help="Tokenizer model name or path")
    variance_parser.add_argument("--min-samples", type=int, default=8,
                                 help="Minimum samples per prompt (default: 8)")
    variance_parser.add_argument("--output-json", type=str, default=None,
                                 help="Save results to JSON file")
    variance_parser.add_argument("-v", "--verbose", action="store_true",
                                 help="Verbose output")
    variance_parser.set_defaults(func=cmd_variance)

    # =========================================================================
    # Prediction command
    # =========================================================================
    prediction_parser = subparsers.add_parser("prediction", help="Analyze runahead prediction correlation")
    prediction_parser.add_argument("data_dir", type=str, help="Path to rollout data directory")
    prediction_parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                                   help="Tokenizer model name or path")
    prediction_parser.add_argument("--min-samples", type=int, default=4,
                                   help="Minimum samples per prompt (default: 4)")
    prediction_parser.add_argument("--plot", action="store_true",
                                   help="Generate visualization plots")
    prediction_parser.add_argument("--output-dir", type=str, default="./prediction_plots",
                                   help="Directory for saving plots")
    prediction_parser.add_argument("--dpi", type=int, default=150,
                                   help="Plot resolution (default: 150)")
    prediction_parser.add_argument("--compare-methods", action="store_true",
                                   help="Compare grouped vs individual correlation methods")
    prediction_parser.add_argument("-v", "--verbose", action="store_true",
                                   help="Verbose output")
    prediction_parser.set_defaults(func=cmd_prediction)

    # =========================================================================
    # Single command
    # =========================================================================
    single_parser = subparsers.add_parser("single", help="Run single tick simulation")
    single_parser.add_argument("data_dir", type=str, help="Path to rollout data directory")
    single_parser.add_argument("--cache-tick", type=int, required=True,
                               help="Tick to populate cache from")
    single_parser.add_argument("--sim-tick", type=int, required=True,
                               help="Tick to simulate")
    single_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B",
                               help="Model path for tokenizer")
    single_parser.add_argument("--min-token-prob", type=float, default=0.3,
                               help="Minimum probability for draft tokens")
    single_parser.add_argument("--hash-token-count", type=int, default=128,
                               help="Tokens to hash for tree sharing")
    single_parser.add_argument("--online-update", action="store_true",
                               help="Enable online updates")
    single_parser.add_argument("--skip-prefill", action="store_true",
                               help="Skip cache prefill (online-only mode)")
    single_parser.add_argument("--max-samples", type=int, default=0,
                               help="Maximum samples (0 = all)")
    single_parser.add_argument("-o", "--output", type=str, default=None,
                               help="Save result to JSON file")
    single_parser.add_argument("-v", "--verbose", action="store_true",
                               help="Verbose output")
    single_parser.set_defaults(func=cmd_single)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
