#!/usr/bin/env python3
"""
EVM Hacker Bench - Main Runner Script

A comprehensive benchmark for evaluating LLM ability to exploit 
vulnerable smart contracts on EVM-compatible blockchains.

Usage:
    # Run on all cases from SCONE-bench
    python run_hacker_bench.py --model anthropic/claude-sonnet-4 --dataset scone
    
    # Run on specific chain
    python run_hacker_bench.py --model openai/gpt-4o --chain bsc --max-cases 10
    
    # Run single case
    python run_hacker_bench.py --model anthropic/claude-sonnet-4 --case gamma
    
    # Resume from specific index
    python run_hacker_bench.py --model openai/gpt-4o --start-index 50

Requirements:
    - Foundry (forge, cast, anvil) installed
    - Python 3.10+
    - OpenRouter API key (OPENROUTER_API_KEY env var)
"""

import argparse
import json
import os
import sys
import time
import io
import signal
import atexit
from pathlib import Path
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Global list to track cleanup functions
_cleanup_functions = []

def register_cleanup(func):
    """Register a cleanup function to be called on exit"""
    _cleanup_functions.append(func)

def cleanup_all():
    """Run all registered cleanup functions"""
    for func in reversed(_cleanup_functions):
        try:
            func()
        except Exception as e:
            print(f"Cleanup error: {e}", file=sys.__stderr__)

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print(f"\n\n🛑 Received signal {signum}, cleaning up...", file=sys.__stderr__)
    cleanup_all()
    sys.exit(128 + signum)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_all)


class TeeWriter:
    """
    A class that writes to both stdout and a log file simultaneously.
    Used to capture console output to log files.
    """
    def __init__(self, log_file_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w', encoding='utf-8', buffering=1)  # Line buffered
        
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()


class TeeStderr:
    """
    A class that writes to both stderr and a log file simultaneously.
    """
    def __init__(self, log_file, original_stderr):
        self.terminal = original_stderr
        self.log_file = log_file
        
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from evm_hacker_bench.evm_env import EVMEnvironment
from evm_hacker_bench.case_loader import (
    CaseLoader, 
    AttackCase, 
    create_combined_dataset
)
from evm_hacker_bench.hacker_controller import HackerController, BatchHackerController
from evm_hacker_bench.exploit_validator import ExploitValidator, ScoringSystem
from evm_hacker_bench.contract_fetcher import ContractFetcher


def setup_dataset(args) -> CaseLoader:
    """Setup and load the attack case dataset"""
    loader = CaseLoader()
    
    # Define paths
    base_path = Path(__file__).parent
    bundled_data = base_path / "data" / "cases.json"
    
    # Priority: 
    # 1. Custom cases if provided
    # 2. Bundled JSON data (no external dependencies)
    
    # Load custom JSON if provided
    if args.custom_cases:
        custom_path = Path(args.custom_cases)
        if custom_path.exists():
            loader.load_json(custom_path)
            return loader
    
    # Load bundled data (self-contained, includes POC references)
    if bundled_data.exists():
        loader.load_bundled_cases()
        if loader.cases:
            return loader
    
    print("❌ No cases found. Please ensure data/cases.json exists.")
    return loader


# Cases known to be invalid (RPC issues, etc.)
# Note: Previously excluded cases have been re-enabled for testing
# Use --exclude-cases flag or case_ids file to filter specific cases
EXCLUDED_CASES = set()  # Empty - no hardcoded exclusions


def filter_cases(
    loader: CaseLoader, 
    args
) -> List[AttackCase]:
    """Filter cases based on arguments"""
    cases = loader.cases
    
    # Exclude known invalid cases
    original_count = len(cases)
    cases = [c for c in cases if c.case_name not in EXCLUDED_CASES]
    excluded_count = original_count - len(cases)
    if excluded_count > 0:
        print(f"✓ Excluded {excluded_count} known invalid cases")
    
    # Filter by case IDs file (from filter_valid_cases.py)
    if args.case_ids:
        case_ids_file = Path(args.case_ids)
        if case_ids_file.exists():
            valid_ids = set(case_ids_file.read_text().strip().split('\n'))
            cases = [c for c in cases if c.case_id in valid_ids]
            print(f"✓ Filtered to {len(cases)} cases from {args.case_ids}")
        else:
            print(f"⚠️  Case IDs file not found: {args.case_ids}")
    
    # Filter by chain
    if args.chain:
        cases = [c for c in cases if c.chain == args.chain]
    
    # Filter by minimum block number
    if args.min_block:
        original_count = len(cases)
        cases = [c for c in cases if c.fork_block >= args.min_block]
        filtered_count = original_count - len(cases)
        if filtered_count > 0:
            print(f"✓ Filtered out {filtered_count} cases with fork_block < {args.min_block}")
    
    # Filter by -since date (e.g., 202503 or 2025-03)
    if args.since:
        # Normalize date format (support YYYYMM or YYYY-MM)
        since_date = args.since.strip()
        if len(since_date) == 6 and since_date.isdigit():
            since_date = f"{since_date[:4]}-{since_date[4:]}"
        
        original_count = len(cases)
        cases = [c for c in cases if c.attack_date and c.attack_date >= since_date]
        filtered_count = original_count - len(cases)
        print(f"✓ Filtered to {len(cases)} cases with attack_date >= {since_date} ({filtered_count} older cases excluded)")
    
    # Filter by category
    if args.category:
        from evm_hacker_bench.case_loader import AttackCategory
        try:
            category = AttackCategory(args.category)
            cases = [c for c in cases if c.category == category]
        except ValueError:
            print(f"⚠️  Unknown category: {args.category}")
    
    # Filter by specific case
    if args.case:
        cases = [c for c in cases if args.case in c.case_name or args.case in c.case_id]
    
    # Apply start index
    if args.start_index:
        cases = cases[args.start_index:]
    
    # Apply max cases limit
    if args.max_cases:
        cases = cases[:args.max_cases]
    
    return cases


def get_fork_url_for_chain(chain: str, args) -> Optional[str]:
    """
    Get the appropriate fork RPC URL for the given chain.
    
    Priority:
    1. Generic --fork-url (backward compatibility, overrides all)
    2. Chain-specific URL (--bsc-fork-url, --eth-fork-url, etc.)
    3. None (use default RPC from EVMEnvironment)
    """
    # If generic fork-url is specified, use it for all chains
    if args.fork_url:
        return args.fork_url
    
    # Chain-specific RPC URLs
    chain_url_map = {
        'bsc': getattr(args, 'bsc_fork_url', None),
        'mainnet': getattr(args, 'eth_fork_url', None),
        'ethereum': getattr(args, 'eth_fork_url', None),
        'arbitrum': getattr(args, 'arbitrum_fork_url', None),
        'base': getattr(args, 'base_fork_url', None),
        'polygon': getattr(args, 'polygon_fork_url', None),
    }
    
    return chain_url_map.get(chain)


def run_single_case(
    case: AttackCase,
    model_name: str,
    api_key: Optional[str],
    args,
    output_dir: Path
) -> dict:
    """Run attack on a single case"""
    # Get the appropriate RPC URL for this chain
    rpc_url = get_fork_url_for_chain(case.chain, args)
    
    print(f"\n{'='*70}")
    print(f"🎯 TARGET: {case.case_name}")
    print(f"   Chain: {case.chain} | Block: {case.fork_block}")
    if case.attack_date:
        print(f"   Date: {case.attack_date} (contract age indicator)")
    print(f"   Address: {case.target_address}")
    if rpc_url:
        print(f"   RPC: {rpc_url[:50]}...")
    else:
        print(f"   RPC: default (from EVMEnvironment)")
    print(f"{'='*70}")
    
    result = {
        "case_id": case.case_id,
        "case_name": case.case_name,
        "chain": case.chain,
        "success": False,
        "error": None
    }
    
    # Skip if no RPC URL configured for this chain
    if not rpc_url:
        print(f"⚠️  No RPC URL configured for chain: {case.chain}, skipping...")
        result["error"] = f"No RPC URL configured for chain: {case.chain}"
        return result
    
    try:
        # Start environment
        with EVMEnvironment(
            chain=case.chain,
            fork_block=case.fork_block,
            evm_version=case.evm_version,
            rpc_url=rpc_url  # Use chain-specific fork URL
        ) as env:
            
            # Create controller
            # Determine compression setting (--no-compression overrides --enable-compression)
            enable_compression = args.enable_compression and not args.no_compression
            
            controller = HackerController(
                model_name=model_name,
                api_key=api_key,
                base_url=args.base_url,
                temperature=args.temperature,
                max_turns=args.max_turns,
                timeout_per_turn=args.timeout,
                session_timeout=args.session_timeout,
                enable_thinking=args.thinking,
                thinking_budget=args.thinking_budget,
                verbose=args.verbose,
                enable_compression=enable_compression,
                progress_mode=args.progress_mode,
                log_dir=output_dir  # Use same timestamped directory for raw_data storage
            )
            
            # Try to fetch contract source from block explorer
            contract_source = None
            if args.fetch_source:
                try:
                    print(f"📥 Fetching contract source from block explorer...")
                    fetcher = ContractFetcher(case.chain)
                    source = fetcher.fetch_source(case.target_address)
                    if source:
                        contract_source = source.source_code
                        # Also save to workspace
                        work_dir = controller.work_dir / case.case_id
                        contracts_dir = work_dir / "etherscan-contracts"
                        fetcher.save_to_directory(source, contracts_dir)
                        print(f"   ✓ Source code saved to workspace")
                except Exception as e:
                    print(f"   ⚠️ Could not fetch source: {e}")
            
            # Discovery Mode: LLM must find vulnerability independently
            print(f"   🔍 Discovery Mode: LLM must find vulnerability independently")
            
            # Run attack (Discovery Mode only)
            attack_result = controller.run_attack(case, env, contract_source)
            
            result.update(attack_result)
            
            # Sync success field with final_success
            if 'final_success' in attack_result:
                result['success'] = attack_result['final_success']
            
    except Exception as e:
        result["error"] = str(e)
        print(f"❌ Error: {e}")
    
    return result


def setup_explorer_api_keys(args):
    """Setup block explorer API keys from args or environment"""
    # Map of chain -> (arg_name, env_var_name)
    api_key_mapping = {
        'mainnet': ('etherscan_api_key', 'ETHERSCAN_API_KEY'),
        'bsc': ('bscscan_api_key', 'BSCSCAN_API_KEY'),
        'arbitrum': ('arbiscan_api_key', 'ARBISCAN_API_KEY'),
        'base': ('basescan_api_key', 'BASESCAN_API_KEY'),
        'polygon': ('polygonscan_api_key', 'POLYGONSCAN_API_KEY'),
    }
    
    configured_keys = []
    for chain, (arg_name, env_var) in api_key_mapping.items():
        # Get from args first, then from env
        arg_value = getattr(args, arg_name.replace('-', '_'), None)
        if arg_value:
            os.environ[env_var] = arg_value
            configured_keys.append(chain)
        elif os.getenv(env_var):
            configured_keys.append(chain)
    
    return configured_keys


def run_benchmark(args):
    """Main benchmark runner"""
    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(args.output_dir)
    
    # Create timestamp subdirectory unless --no-timestamp-dir is set
    # (bash script already creates timestamped directory)
    if getattr(args, 'no_timestamp_dir', False):
        output_dir = base_output_dir
    else:
        output_dir = base_output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_safe = args.model.replace("/", "_")
    
    # Setup console log file (tee output to both console and file)
    # Skip if --no-console-log is set (when bash script uses tee externally)
    console_log_file = None
    tee_writer = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    if not getattr(args, 'no_console_log', False):
        console_log_file = output_dir / f"{model_safe}.log"
        tee_writer = TeeWriter(console_log_file)
        sys.stdout = tee_writer
        sys.stderr = TeeStderr(tee_writer.log_file, original_stderr)
    
    try:
        print(f"\n{'='*70}")
        print("🔥 EVM HACKER BENCH")
        print(f"{'='*70}")
        print(f"Model: {args.model}")
        print(f"Dataset: {args.dataset}")
        if args.chain:
            print(f"Chain Filter: {args.chain}")
        print(f"Temperature: {args.temperature}")
        print(f"Extended Thinking: {'Enabled' if args.thinking else 'Disabled'}")
        if args.thinking:
            print(f"Thinking Budget: {args.thinking_budget} tokens")
        compression_enabled = args.enable_compression and not args.no_compression
        print(f"Message Compression: {'Enabled' if compression_enabled else 'Disabled'}")
        print(f"Mode: 🔍 Discovery Mode (LLM must find vulnerabilities independently)")
        print(f"Progress Mode: {args.progress_mode} ({'elapsed/total minutes' if args.progress_mode == 'time' else 'turn/max_turns'})")
        print(f"Max Turns: {args.max_turns}")
        print(f"Timeout per Turn: {args.timeout}s")
        print(f"Session Timeout: {args.session_timeout}s ({args.session_timeout//60} minutes)")
        if args.fork_url:
            print(f"Custom Fork URL: {args.fork_url}")
        if console_log_file:
            print(f"Console Log: {console_log_file}")
        
        # Setup explorer API keys
        configured_keys = setup_explorer_api_keys(args)
        if configured_keys:
            print(f"Explorer API Keys: {', '.join(configured_keys)}")
        
        # Get API key
        api_key = args.api_key or os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            print("\n❌ Error: No API key provided")
            print("   Set OPENROUTER_API_KEY environment variable or use --api-key")
            sys.exit(1)
        
        # Load dataset
        print(f"\n📚 Loading attack cases...")
        loader = setup_dataset(args)
        
        if not loader.cases:
            print("❌ No cases loaded")
            sys.exit(1)
        
        # Filter cases
        cases = filter_cases(loader, args)
        print(f"✓ {len(cases)} cases selected for testing")
        
        # Print statistics
        stats = loader.get_statistics()
        print(f"\n📊 Dataset Statistics:")
        print(f"   By Chain: {stats['by_chain']}")
        print(f"   By Source: {stats['by_source']}")
        
        # Run attacks
        results = []
        scoring = ScoringSystem()
        intermediate_file = None
        
        for i, case in enumerate(cases):
            print(f"\n[{i+1}/{len(cases)}]", end=" ")
            
            result = run_single_case(case, args.model, api_key, args, output_dir)
            results.append(result)
            
            # Delay to avoid RPC rate limiting (important for QuickNode and other RPCs)
            time.sleep(2.0)
            
            # Save intermediate results
            intermediate_file = output_dir / f"results_{model_safe}_partial.json"
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "model": args.model,
                    "timestamp": timestamp,
                    "completed": i + 1,
                    "total": len(cases),
                    "results": results
                }, f, indent=2, ensure_ascii=False)
        
        # Calculate final metrics
        successful = sum(1 for r in results if r.get('final_success', False))
        total_profit = sum(r.get('profit', 0) or 0 for r in results)
        
        # Calculate average duration and turns
        durations = [r.get('stats', {}).get('duration_seconds', 0) for r in results if r.get('stats')]
        turns_list = [r.get('stats', {}).get('total_turns', 0) for r in results if r.get('stats')]
        avg_duration = sum(durations) / len(durations) if durations else 0
        avg_turns = sum(turns_list) / len(turns_list) if turns_list else 0
        
        # Save final results
        final_results = {
            "benchmark": "EVM Hacker Bench",
            "version": "0.1.0",
            "model": args.model,
            "dataset": args.dataset,
            "chain_filter": args.chain,
            "timestamp": timestamp,
            "configuration": {
                "max_turns": args.max_turns,
                "timeout": args.timeout,
                "min_profit": 0.1
            },
            "summary": {
                "total_cases": len(cases),
                "successful": successful,
                "failed": len(cases) - successful,
                "success_rate": round(100 * successful / len(cases), 2) if cases else 0,
                "total_profit": round(total_profit, 4),
                "avg_duration_seconds": round(avg_duration, 1),
                "avg_turns": round(avg_turns, 1)
            },
            "results": results
        }
        
        final_file = output_dir / f"benchmark_results_{model_safe}.json"
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        # Generate summary files
        summary_score_file = output_dir / "summary_score.log"
        summary_detail_file = output_dir / "summary_detail.log"
        
        # Write summary_score.log (compact format)
        success_rate = final_results['summary']['success_rate']
        with open(summary_score_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*70}\n")
            f.write(f"EVM HACKER BENCH - SUMMARY SCORE\n")
            f.write(f"{'='*70}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"\n")
            f.write(f"{model_safe}: {successful}/{len(cases)} ({success_rate}%) | Profit: {total_profit:.4f} | Avg: {avg_duration:.0f}s, {avg_turns:.0f} turns\n")
        
        # Write summary_detail.log (detailed format)
        with open(summary_detail_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*70}\n")
            f.write(f"EVM HACKER BENCH - DETAILED SUMMARY\n")
            f.write(f"{'='*70}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Total Cases: {len(cases)}\n")
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {len(cases) - successful}\n")
            f.write(f"Success Rate: {success_rate}%\n")
            f.write(f"Total Profit: {total_profit:.4f}\n")
            f.write(f"Avg Duration: {avg_duration:.1f}s ({avg_duration/60:.1f} min)\n")
            f.write(f"Avg Turns: {avg_turns:.1f}\n")
            f.write(f"\n{'='*70}\n")
            f.write(f"CASE RESULTS\n")
            f.write(f"{'='*70}\n\n")
            
            for r in results:
                status = "✅ SUCCESS" if r.get('final_success') else "❌ FAILED"
                profit = r.get('profit', 0) or 0
                duration = r.get('stats', {}).get('duration_seconds', 0)
                turns = r.get('stats', {}).get('total_turns', 0)
                error = r.get('error', '')
                
                f.write(f"Case: {r.get('case_name', 'unknown')}\n")
                f.write(f"  Status: {status}\n")
                f.write(f"  Chain: {r.get('chain', 'unknown')}\n")
                f.write(f"  Profit: {profit:.4f}\n")
                f.write(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)\n")
                f.write(f"  Turns: {turns}\n")
                if error:
                    f.write(f"  Error: {error[:100]}...\n" if len(error) > 100 else f"  Error: {error}\n")
                f.write(f"\n")
        
        # Print final summary
        print(f"\n{'='*70}")
        print("📊 FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Model: {args.model}")
        print(f"Total Cases: {len(cases)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(cases) - successful}")
        print(f"Success Rate: {final_results['summary']['success_rate']}%")
        print(f"Total Profit: {total_profit:.4f}")
        print(f"Avg Duration: {avg_duration:.1f}s | Avg Turns: {avg_turns:.1f}")
        print(f"\nResults saved to: {final_file}")
        print(f"Summary score: {summary_score_file}")
        print(f"Summary detail: {summary_detail_file}")
        if console_log_file:
            print(f"Console log: {console_log_file}")
        print(f"{'='*70}\n")
        
        # Remove intermediate file
        if intermediate_file and intermediate_file.exists():
            intermediate_file.unlink()
    
    finally:
        # Restore original stdout/stderr and close log file
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if tee_writer:
            tee_writer.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="EVM Hacker Bench - LLM Smart Contract Exploitation Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='LLM model name (e.g., anthropic/claude-sonnet-4, openai/gpt-4o)'
    )
    
    # Dataset arguments
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='bundled',
        choices=['bundled', 'scone', 'defihacklabs', 'all', 'custom'],
        help='Dataset to use: bundled (built-in), scone, defihacklabs, all, or custom (default: bundled)'
    )
    
    parser.add_argument(
        '--custom-cases',
        type=str,
        help='Path to custom cases JSON file'
    )
    
    # Filter arguments
    parser.add_argument(
        '--chain',
        type=str,
        choices=['mainnet', 'bsc', 'arbitrum', 'base', 'polygon', 'optimism'],
        help='Filter cases by chain'
    )
    
    parser.add_argument(
        '--category',
        type=str,
        help='Filter cases by attack category'
    )
    
    parser.add_argument(
        '--case',
        type=str,
        help='Run specific case by name or ID'
    )
    
    parser.add_argument(
        '--case-ids',
        type=str,
        help='Path to file containing case IDs (one per line) - from filter_valid_cases.py output'
    )
    
    parser.add_argument(
        '--min-block',
        type=int,
        help='Minimum fork block number (filter out older blocks that RPC may not support)'
    )
    
    parser.add_argument(
        '-since', '--since',
        type=str,
        dest='since',
        help='Filter cases >= date. Format: YYYYMM or YYYY-MM (e.g., 202503 or 2025-03)'
    )
    
    # Execution arguments
    parser.add_argument(
        '--max-cases',
        type=int,
        help='Maximum number of cases to run'
    )
    
    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='Start from specific case index (for resuming)'
    )
    
    parser.add_argument(
        '--max-turns',
        type=int,
        default=50,
        help='Maximum LLM interaction turns per case (default: 50, for multi-tool workflow)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Timeout per turn in seconds (default: 300)'
    )
    
    parser.add_argument(
        '--session-timeout',
        type=int,
        default=3600,
        help='Maximum session duration in seconds (default: 3600 = 60 minutes)'
    )
    
    # API arguments
    parser.add_argument(
        '--api-key',
        type=str,
        help='API key (defaults to OPENROUTER_API_KEY env var)'
    )
    
    parser.add_argument(
        '--base-url',
        type=str,
        default='https://openrouter.ai/api/v1',
        help='API base URL (default: OpenRouter)'
    )
    
    # LLM parameters
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=0.7,
        help='LLM temperature (default: 0.7, lower=more deterministic, higher=more creative)'
    )
    
    parser.add_argument(
        '--thinking',
        action='store_true',
        help='Enable extended thinking mode for supported models (e.g., Claude)'
    )
    
    parser.add_argument(
        '--thinking-budget',
        type=int,
        default=10000,
        help='Max tokens for thinking when --thinking is enabled (default: 10000)'
    )
    
    parser.add_argument(
        '--enable-compression',
        action='store_true',
        default=True,
        help='Enable message compression using turn summaries (default: True)'
    )
    
    parser.add_argument(
        '--no-compression',
        action='store_true',
        help='Disable message compression (send full conversation history)'
    )
    
    parser.add_argument(
        '--progress-mode',
        type=str,
        default='time',
        choices=['time', 'turns'],
        help='Progress display mode: "time" (elapsed/total minutes) or "turns" (turn/max_turns). Default: time'
    )
    
    # Fork arguments
    parser.add_argument(
        '--fork-url',
        type=str,
        help='Custom fork RPC URL (overrides all chain RPCs, for backward compatibility)'
    )
    
    parser.add_argument(
        '--bsc-fork-url',
        type=str,
        help='Fork RPC URL for BSC chain'
    )
    
    parser.add_argument(
        '--eth-fork-url',
        type=str,
        help='Fork RPC URL for Ethereum mainnet'
    )
    
    parser.add_argument(
        '--arbitrum-fork-url',
        type=str,
        help='Fork RPC URL for Arbitrum'
    )
    
    parser.add_argument(
        '--base-fork-url',
        type=str,
        help='Fork RPC URL for Base chain'
    )
    
    parser.add_argument(
        '--polygon-fork-url',
        type=str,
        help='Fork RPC URL for Polygon'
    )
    
    parser.add_argument(
        '--fetch-source',
        action='store_true',
        help='Fetch contract source code from block explorer'
    )
    
    # Block explorer API keys
    parser.add_argument(
        '--etherscan-api-key',
        type=str,
        help='Etherscan API key for mainnet (also uses ETHERSCAN_API_KEY env var)'
    )
    
    parser.add_argument(
        '--bscscan-api-key',
        type=str,
        help='BSCScan API key for BSC chain (also uses BSCSCAN_API_KEY env var)'
    )
    
    parser.add_argument(
        '--arbiscan-api-key',
        type=str,
        help='Arbiscan API key for Arbitrum (also uses ARBISCAN_API_KEY env var)'
    )
    
    parser.add_argument(
        '--basescan-api-key',
        type=str,
        help='BaseScan API key for Base chain (also uses BASESCAN_API_KEY env var)'
    )
    
    parser.add_argument(
        '--polygonscan-api-key',
        type=str,
        help='PolygonScan API key for Polygon (also uses POLYGONSCAN_API_KEY env var)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='logs',
        help='Output directory for results (default: logs)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed LLM inputs and outputs for debugging'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Log directory for raw_data storage (default: logs)'
    )
    
    parser.add_argument(
        '--no-console-log',
        action='store_true',
        help='Disable separate console log file (use when calling from bash script with tee)'
    )
    
    parser.add_argument(
        '--no-timestamp-dir',
        action='store_true',
        help='Do not create timestamp subdirectory (use when bash script already created it)'
    )
    
    args = parser.parse_args()
    
    run_benchmark(args)


if __name__ == '__main__':
    main()

