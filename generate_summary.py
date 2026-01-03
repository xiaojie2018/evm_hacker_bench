#!/usr/bin/env python3
"""
Generate summary files from existing benchmark results.

Usage:
    python generate_summary.py logs/20260102_144112/benchmark_results_google_gemini-3-pro-preview.json
    python generate_summary.py logs/20260102_144112/  # Process all benchmark_results*.json in directory
"""

import json
import sys
from pathlib import Path


def generate_summary(results_file: Path):
    """Generate summary files from a benchmark results JSON file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_dir = results_file.parent
    model = data.get('model', 'unknown')
    model_safe = model.replace("/", "_")
    timestamp = data.get('timestamp', 'unknown')
    dataset = data.get('dataset', 'unknown')
    results = data.get('results', [])
    
    # Calculate metrics
    successful = sum(1 for r in results if r.get('final_success', False))
    total = len(results)
    total_profit = sum(r.get('profit', 0) or 0 for r in results)
    
    durations = [r.get('stats', {}).get('duration_seconds', 0) for r in results if r.get('stats')]
    turns_list = [r.get('stats', {}).get('total_turns', 0) for r in results if r.get('stats')]
    avg_duration = sum(durations) / len(durations) if durations else 0
    avg_turns = sum(turns_list) / len(turns_list) if turns_list else 0
    success_rate = round(100 * successful / total, 2) if total else 0
    
    # Write summary_score.log
    summary_score_file = output_dir / "summary_score.log"
    with open(summary_score_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*70}\n")
        f.write(f"EVM HACKER BENCH - SUMMARY SCORE\n")
        f.write(f"{'='*70}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {model}\n")
        f.write(f"\n")
        f.write(f"{model_safe}: {successful}/{total} ({success_rate}%) | Profit: {total_profit:.4f} | Avg: {avg_duration:.0f}s, {avg_turns:.0f} turns\n")
    
    # Write summary_detail.log
    summary_detail_file = output_dir / "summary_detail.log"
    with open(summary_detail_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*70}\n")
        f.write(f"EVM HACKER BENCH - DETAILED SUMMARY\n")
        f.write(f"{'='*70}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Total Cases: {total}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {total - successful}\n")
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
    
    print(f"✅ Generated summary files:")
    print(f"   - {summary_score_file}")
    print(f"   - {summary_detail_file}")
    
    # Print quick summary
    print(f"\n📊 Quick Summary:")
    print(f"   Model: {model}")
    print(f"   Results: {successful}/{total} ({success_rate}%)")
    print(f"   Profit: {total_profit:.4f}")
    print(f"   Avg: {avg_duration:.0f}s, {avg_turns:.0f} turns")


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_summary.py <results_file_or_directory>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if path.is_file() and path.suffix == '.json':
        generate_summary(path)
    elif path.is_dir():
        # Find all benchmark_results*.json files
        results_files = list(path.glob("benchmark_results*.json"))
        if not results_files:
            print(f"❌ No benchmark_results*.json files found in {path}")
            sys.exit(1)
        for f in results_files:
            print(f"\n📁 Processing: {f}")
            generate_summary(f)
    else:
        print(f"❌ Invalid path: {path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
