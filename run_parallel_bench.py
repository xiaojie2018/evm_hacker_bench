#!/usr/bin/env python3
"""
EVM Hacker Bench - Parallel Model Runner

Run multiple LLM models in parallel, each with its own Anvil instance.

Usage:
    # Run 2 models in parallel
    python run_parallel_bench.py \
        --models "anthropic/claude-sonnet-4" "openai/gpt-4o" \
        --parallel 2 \
        --chain bsc \
        --max-cases 5
    
    # Run with extended thinking
    python run_parallel_bench.py \
        --models "anthropic/claude-sonnet-4:thinking" "openai/o1-preview" \
        --parallel 2
    
    # Use short names for logs
    python run_parallel_bench.py \
        --models "claude=anthropic/claude-sonnet-4:thinking" "gpt4o=openai/gpt-4o" \
        --parallel 2

Requirements:
    - Foundry (forge, cast, anvil) installed
    - Python 3.10+
    - OpenRouter API key (OPENROUTER_API_KEY env var)
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from evm_hacker_bench import (
    CaseLoader,
    ParallelModelRunner,
    ModelConfig,
    create_model_configs,
    precache_contracts
)


def main():
    parser = argparse.ArgumentParser(
        description="EVM Hacker Bench - Parallel Model Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run 2 models in parallel on BSC cases
    python run_parallel_bench.py \\
        --models "anthropic/claude-sonnet-4" "openai/gpt-4o" \\
        --parallel 2 --chain bsc

    # Run with extended thinking (append :thinking)
    python run_parallel_bench.py \\
        --models "anthropic/claude-sonnet-4:thinking" \\
        --parallel 1

    # Use custom short names for logs
    python run_parallel_bench.py \\
        --models "claude=anthropic/claude-sonnet-4" "gpt=openai/gpt-4o" \\
        --parallel 2
        """
    )
    
    # Model arguments
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        required=True,
        help='Model names to run. Format: "model_name" or "short=model_name:thinking"'
    )
    
    parser.add_argument(
        '--parallel', '-p',
        type=int,
        default=2,
        help='Maximum number of parallel models (default: 2)'
    )
    
    parser.add_argument(
        '--base-port',
        type=int,
        default=8545,
        help='Base port for Anvil instances (default: 8545)'
    )
    
    # API key
    parser.add_argument(
        '--api-key',
        type=str,
        help='API key (or set OPENROUTER_API_KEY env var)'
    )
    
    # Dataset arguments
    parser.add_argument(
        '--chain',
        type=str,
        choices=['mainnet', 'bsc', 'arbitrum', 'base', 'polygon', 'optimism'],
        help='Filter cases by chain'
    )
    
    parser.add_argument(
        '--case',
        type=str,
        help='Run specific case by name or ID'
    )
    
    parser.add_argument(
        '--case-ids',
        type=str,
        help='Path to file containing case IDs (one per line)'
    )
    
    parser.add_argument(
        '-since', '--since',
        type=str,
        help='Filter cases >= date. Format: YYYYMM or YYYY-MM'
    )
    
    parser.add_argument(
        '--max-cases',
        type=int,
        help='Maximum number of cases to run per model'
    )
    
    # Execution arguments
    parser.add_argument(
        '--max-turns',
        type=int,
        default=120,
        help='Maximum turns per case (default: 120)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=120,
        help='Timeout per turn in seconds (default: 120)'
    )
    
    parser.add_argument(
        '--session-timeout',
        type=int,
        default=1800,
        help='Total session timeout per case in seconds (default: 1800 = 30min)'
    )
    
    parser.add_argument(
        '--no-compression',
        action='store_true',
        help='Disable message compression'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='logs/parallel',
        help='Output directory for logs (default: logs/parallel)'
    )
    
    parser.add_argument(
        '--no-timestamp-dir',
        action='store_true',
        help='Use output-dir directly without creating timestamp subdirectory'
    )
    
    # RPC URLs
    parser.add_argument(
        '--bsc-fork-url',
        type=str,
        help='BSC fork RPC URL'
    )
    
    parser.add_argument(
        '--eth-fork-url',
        type=str,
        help='Ethereum fork RPC URL'
    )
    
    # Pre-cache arguments
    parser.add_argument(
        '--skip-precache',
        action='store_true',
        help='Skip pre-caching contract source code (not recommended for parallel runs)'
    )
    
    parser.add_argument(
        '--precache-delay',
        type=float,
        default=0.5,
        help='Delay between API requests during pre-cache (default: 0.5s)'
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ Error: API key required. Set OPENROUTER_API_KEY or use --api-key")
        sys.exit(1)
    
    # Set RPC URLs if provided
    if args.bsc_fork_url:
        os.environ['BSC_RPC'] = args.bsc_fork_url
    if args.eth_fork_url:
        os.environ['MAINNET_RPC'] = args.eth_fork_url
    
    # Create model configs
    model_configs = create_model_configs(args.models)
    print(f"\n📋 Models to run ({len(model_configs)}):")
    for mc in model_configs:
        thinking_str = " [thinking]" if mc.thinking else ""
        print(f"   - {mc.short_name}: {mc.name}{thinking_str}")
    
    # Load cases
    print(f"\n📂 Loading cases...")
    loader = CaseLoader()
    
    # Load from bundled JSON
    project_root = Path(__file__).parent
    cases_json = project_root / "data" / "cases.json"
    if cases_json.exists():
        loader.load_json(cases_json)
    else:
        print(f"❌ Cases file not found: {cases_json}")
        sys.exit(1)
    
    cases = loader.cases
    
    # Apply filters
    if args.chain:
        cases = [c for c in cases if c.chain == args.chain]
        print(f"   Chain filter: {args.chain} ({len(cases)} cases)")
    
    if args.case:
        cases = [c for c in cases if args.case in c.case_id or args.case in c.case_name]
        print(f"   Case filter: {args.case} ({len(cases)} cases)")
    
    if args.case_ids:
        case_ids_file = Path(args.case_ids)
        if case_ids_file.exists():
            valid_ids = set(case_ids_file.read_text().strip().split('\n'))
            cases = [c for c in cases if c.case_id in valid_ids]
            print(f"   Case IDs filter: {len(cases)} cases")
    
    if args.since:
        # Parse date filter
        date_str = args.since.replace('-', '')
        cases = [c for c in cases if c.attack_date and c.attack_date.replace('-', '') >= date_str]
        print(f"   Date filter (>= {args.since}): {len(cases)} cases")
    
    if args.max_cases:
        cases = cases[:args.max_cases]
        print(f"   Max cases: {args.max_cases}")
    
    if not cases:
        print("❌ No cases to run after filtering")
        sys.exit(1)
    
    print(f"\n✅ Running {len(cases)} cases on {len(model_configs)} models")
    
    # Pre-cache contract source code (serial to avoid API rate limits)
    if not args.skip_precache:
        workspace_dir = project_root / "data" / "exploit_workspace"
        precache_contracts(cases, workspace_dir, delay_seconds=args.precache_delay)
    else:
        print("\n⚠️ Skipping pre-cache (--skip-precache specified)")
    
    # Create output directory
    if args.no_timestamp_dir:
        # Use output_dir directly (called from bash script which already has timestamp dir)
        output_dir = Path(args.output_dir)
    else:
        # Create timestamp subdirectory (standalone usage)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output: {output_dir}")
    
    # Create and run parallel runner
    runner = ParallelModelRunner(
        models=model_configs,
        api_key=api_key,
        max_parallel=args.parallel,
        base_port=args.base_port,
        max_turns=args.max_turns,
        timeout=args.timeout,
        session_timeout=args.session_timeout,
        enable_compression=not args.no_compression
    )
    
    # Run benchmark
    results = runner.run(cases, output_dir)
    
    # Print final summary
    print(f"\n{'='*70}")
    print("🏁 BENCHMARK COMPLETE")
    print(f"{'='*70}")
    
    total_passed = sum(r.cases_passed for r in results if r.success)
    total_cases = sum(r.cases_total for r in results if r.success)
    total_profit = sum(r.total_profit for r in results if r.success)
    
    print(f"Total Passed: {total_passed}/{total_cases}")
    print(f"Total Profit: {total_profit:.4f}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

