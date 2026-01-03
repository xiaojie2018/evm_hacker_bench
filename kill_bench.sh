#!/bin/bash
# Kill all evm_hacker_bench related processes

echo "🛑 Killing EVM Hacker Bench processes..."

# Kill anvil processes
pkill -f "anvil.*--port" 2>/dev/null && echo "   ✓ Killed anvil" || echo "   - No anvil process"

# Kill forge processes
pkill -f "forge test" 2>/dev/null && echo "   ✓ Killed forge" || echo "   - No forge process"

# Kill python run_hacker_bench processes
pkill -f "run_hacker_bench.py" 2>/dev/null && echo "   ✓ Killed run_hacker_bench" || echo "   - No run_hacker_bench process"

# Kill conda run wrapper if exists
pkill -f "conda run.*evmbench" 2>/dev/null && echo "   ✓ Killed conda wrapper" || echo "   - No conda wrapper"

# Wait a bit
sleep 1

# Check if any remain
remaining=$(ps aux | grep -E 'run_hacker_bench|anvil.*--port|forge test' | grep -v grep | wc -l)
if [ "$remaining" -gt 0 ]; then
    echo "⚠️  $remaining processes still running, force killing..."
    pkill -9 -f "anvil.*--port" 2>/dev/null
    pkill -9 -f "forge test" 2>/dev/null
    pkill -9 -f "run_hacker_bench.py" 2>/dev/null
    pkill -9 -f "conda run.*evmbench" 2>/dev/null
fi

echo "✅ Done"
