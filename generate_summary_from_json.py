#!/usr/bin/env python3
"""Generate summary files from parallel_results JSON file."""

import json
import sys
from pathlib import Path
from datetime import datetime


def generate_summary(log_dir: str):
    """Generate summary_score.log and summary_detail.log from JSON results."""
    log_path = Path(log_dir)
    
    # Find JSON result file
    json_files = list(log_path.glob("parallel_results_*.json"))
    if not json_files:
        # Also check for benchmark_results_*.json
        json_files = list(log_path.glob("benchmark_results_*.json"))
    
    if not json_files:
        print(f"❌ No results JSON file found in {log_dir}")
        return False
    
    # Use the most recent one
    json_file = sorted(json_files)[-1]
    print(f"📄 Reading: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    timestamp = data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Token prices (estimate based on common values)
    BNB_PRICE = 700
    ETH_PRICE = 3500
    
    # Generate summary_score.log
    score_file = log_path / "summary_score.log"
    with open(score_file, 'w') as f:
        f.write("=" * 69 + "\n")
        f.write("EVM Hacker Bench - Model Success Rate Summary\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Token Prices: BNB=${BNB_PRICE}, ETH=${ETH_PRICE}\n")
        f.write("=" * 69 + "\n\n")
        
        # Header
        f.write(f"{'Model':<40} | {'Success':<8} | {'Total':<8} | {'Rate':<10} | {'Profit (Token)':<15} | {'Profit (USD)':<15} | {'Avg Time':<10} | {'Avg Turns':<10}\n")
        f.write("-" * 41 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 12 + "|" + "-" * 16 + "|" + "-" * 16 + "|" + "-" * 12 + "|" + "-" * 10 + "\n")
        
        for result in data.get('results', []):
            model_name = result.get('short_name', result.get('model_name', 'unknown'))
            cases_passed = result.get('cases_passed', 0)
            cases_total = result.get('cases_total', 34)
            total_profit = result.get('total_profit', 0.0)
            duration = result.get('duration_seconds', 0)
            
            rate = (cases_passed / cases_total * 100) if cases_total > 0 else 0
            avg_time = duration / cases_total / 60 if cases_total > 0 else 0  # minutes per case
            
            # Estimate USD profit (assume mixed BNB/ETH, use average)
            avg_price = (BNB_PRICE + ETH_PRICE) / 2
            profit_usd = total_profit * avg_price
            
            # Parse log file to get avg turns
            avg_turns = 0
            log_file = result.get('log_file')
            if log_file and Path(log_file).exists():
                try:
                    with open(log_file, 'r') as lf:
                        content = lf.read()
                        # Count "Turn X" occurrences
                        import re
                        turns = re.findall(r'Turns: (\d+)', content)
                        if turns:
                            avg_turns = sum(int(t) for t in turns) / len(turns)
                except:
                    pass
            
            f.write(f"{model_name:<40} | {cases_passed:<8} | {cases_total:<8} | {rate:>6.1f}%    | {total_profit:>14.4f} | ${profit_usd:>13,.0f} | {avg_time:>8.1f}m  | {avg_turns:>8.1f}\n")
    
    print(f"✅ Generated: {score_file}")
    
    # Generate summary_detail.log
    detail_file = log_path / "summary_detail.log"
    with open(detail_file, 'w') as f:
        f.write("=" * 69 + "\n")
        f.write("EVM Hacker Bench - Detailed Results\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=" * 69 + "\n\n")
        
        for result in data.get('results', []):
            model_name = result.get('short_name', result.get('model_name', 'unknown'))
            full_name = result.get('model_name', model_name)
            cases_passed = result.get('cases_passed', 0)
            cases_total = result.get('cases_total', 34)
            total_profit = result.get('total_profit', 0.0)
            duration = result.get('duration_seconds', 0)
            error = result.get('error')
            log_file = result.get('log_file', '')
            
            rate = (cases_passed / cases_total * 100) if cases_total > 0 else 0
            duration_hours = duration / 3600
            
            f.write(f"Model: {full_name}\n")
            f.write(f"  Short Name: {model_name}\n")
            f.write(f"  Success: {cases_passed}/{cases_total} ({rate:.1f}%)\n")
            f.write(f"  Total Profit: {total_profit:.4f}\n")
            f.write(f"  Duration: {duration_hours:.2f} hours ({duration:.0f} seconds)\n")
            if error:
                f.write(f"  Error: {error}\n")
            f.write(f"  Log: {log_file}\n")
            f.write("\n")
            
            # Try to extract successful cases from log (only show successes)
            if log_file and Path(log_file).exists():
                try:
                    with open(log_file, 'r') as lf:
                        content = lf.read()
                        import re
                        # Find successful cases with profit
                        # Format: Case: xxx ... Success: ✅ Yes ... Profit: X.XX native tokens
                        summaries = re.findall(
                            r'Case: (\S+)\s+Model:.*?Success: (✅ Yes|❌ No)\s+(?:Profit: ([\d.]+) native tokens)?',
                            content, re.DOTALL
                        )
                        
                        # Only show successful cases
                        success_cases = [(s[0], s[2] if len(s) > 2 and s[2] else "0") 
                                        for s in summaries if "Yes" in s[1]]
                        
                        if success_cases:
                            f.write("  ✅ Successful Cases:\n")
                            for case_name, profit in success_cases:
                                f.write(f"    - {case_name}: Profit {profit} tokens\n")
                            f.write("\n")
                except Exception as e:
                    f.write(f"  (Could not parse log: {e})\n\n")
    
    print(f"✅ Generated: {detail_file}")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = "/home/yangpei/work/evm_hacker_bench/evm_hacker_bench/logs/20260103_164202"
    
    generate_summary(log_dir)
