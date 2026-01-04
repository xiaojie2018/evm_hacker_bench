# EVM Hacker Bench

> A comprehensive benchmark for evaluating LLM ability to exploit vulnerable smart contracts on EVM-compatible blockchains.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

EVM Hacker Bench is a benchmark framework that evaluates Large Language Models' (LLMs) capabilities in understanding and exploiting smart contract vulnerabilities. It provides:

- **400+ Real-World Attack Cases** from DeFiHackLabs POC repository
- **Multi-Chain Support**: BSC, Ethereum Mainnet, Arbitrum, Base, Polygon
- **Multi-Model Support**: OpenRouter API compatible (Claude, GPT, Gemini, DeepSeek, Qwen, etc.)
- **Parallel Benchmark**: Run multiple models concurrently with isolated environments
- **Automated Evaluation**: Fork-based testing with Foundry (anvil + forge)
- **Detailed Metrics**: Success rate, profit/loss calculation, turn analysis
- **Contract Pre-caching**: Automatic contract source caching to avoid API rate limits

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EVM Hacker Bench                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  CaseLoader  │  │PromptBuilder │  │   LLM API    │          │
│  │              │  │              │  │ (OpenRouter) │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  HackerController                        │   │
│  │  - Multi-turn conversation management                    │   │
│  │  - Tool calling (bash, view_file, edit_file)            │   │
│  │  - Extended thinking support                             │   │
│  │  - Profit maximization strategy                          │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   EVMEnvironment                         │   │
│  │  - Anvil fork management                                 │   │
│  │  - Foundry project scaffolding                           │   │
│  │  - Forge test execution                                  │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 ExploitValidator                         │   │
│  │  - Profit/Loss calculation                               │   │
│  │  - Success criteria validation                           │   │
│  │  - Scoring system                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

1. **Foundry** (forge, cast, anvil)
   ```bash
   curl -L https://foundry.paradigm.xyz | bash
   foundryup
   ```

2. **Python 3.10+**
   ```bash
   pip install -r requirements.txt
   ```

### Dataset

**The benchmark includes 400+ bundled attack cases with reference POCs** - no external setup required!

All data is self-contained in the repository:
- `data/cases.json` - Attack case metadata (target addresses, fork blocks, etc.)
- `data/exploit_workspace/*/reference_poc/` - Reference POC files for each case

Data sources (already integrated):
- [DeFiHackLabs](https://github.com/SunWeb3Sec/DeFiHackLabs) - Real-world exploit POCs
- [SCONE-bench](https://github.com/scone-bench/scone-bench) - Benchmark cases

#### Using Bundled Data (Default)

```bash
# Just run - uses bundled cases automatically
python run_hacker_bench.py --model anthropic/claude-haiku-4.5 --chain bsc
```

#### Custom Dataset

You can also provide your own dataset:

```bash
python run_hacker_bench.py --dataset custom --custom-cases /path/to/your/cases.json
```

JSON format:
```json
[
  {
    "case_id": "example_case",
    "case_name": "Example Exploit",
    "chain": "bsc",
    "target_address": "0x...",
    "fork_block": 12345678,
    "attack_date": "2025-03"
  }
]
```

### Configuration

Set environment variables:

```bash
# Required: LLM API Key
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"

# Required: RPC URLs for fork testing
export BSC_FORK_URL="https://your-bsc-rpc-url"
export ETH_FORK_URL="https://your-eth-rpc-url"  # Optional, for ETH cases

# Optional: Block explorer API keys (for contract source fetching)
export ETHERSCAN_API_KEY="your-etherscan-key"
export BSCSCAN_API_KEY="your-bscscan-key"
```

### Usage

#### Single Model Benchmark

```bash
# Run on a specific case
python run_hacker_bench.py \
    --model anthropic/claude-sonnet-4 \
    --case gamma \
    --fork-url $BSC_FORK_URL

# Run all BSC cases with turn limit
python run_hacker_bench.py \
    --model anthropic/claude-haiku-4.5 \
    --chain bsc \
    --max-turns 100

# Run with time limit only (30 minutes per case, unlimited turns)
python run_hacker_bench.py \
    --model anthropic/claude-haiku-4.5 \
    --max-turns 9999 \
    --session-timeout 1800 \
    --bsc-fork-url $BSC_FORK_URL

# Run recent cases (since March 2025)
python run_hacker_bench.py \
    --model anthropic/claude-sonnet-4.5 \
    -since 2025-03 \
    --session-timeout 1800
```

#### Parallel Multi-Model Benchmark

Run multiple models concurrently with isolated Anvil instances:

```bash
# Run 2 models in parallel on BSC cases
python run_parallel_bench.py \
    --models "anthropic/claude-sonnet-4" "openai/gpt-4o" \
    --parallel 2 \
    --chain bsc

# Run with extended thinking (append :thinking)
python run_parallel_bench.py \
    --models "anthropic/claude-sonnet-4:thinking" \
    --parallel 1

# Use custom short names for cleaner logs
python run_parallel_bench.py \
    --models "claude=anthropic/claude-sonnet-4" "gpt=openai/gpt-4o" \
    --parallel 2 \
    --session-timeout 1800

# Run 4 models on recent BSC cases
python run_parallel_bench.py \
    --models \
        "claude-sonnet=anthropic/claude-sonnet-4" \
        "claude-opus=anthropic/claude-opus-4.5" \
        "gpt4o=openai/gpt-4o" \
        "deepseek=deepseek/deepseek-v3" \
    --parallel 4 \
    --chain bsc \
    -since 2025-03
```

#### Multi-Model Benchmark (Shell Script)

For more complex multi-model runs with advanced configuration:

```bash
# 1. Copy the example script
cp test/run_multi_model_bench.sh.example test/run_multi_model_bench.sh

# 2. Edit the script and set your API key and RPC URLs
#    - API_KEY: Your OpenRouter API key
#    - BSC_FORK_URL: Your BSC RPC URL
#    - ETH_FORK_URL: Your Ethereum RPC URL (optional)

# 3. Make it executable
chmod +x test/run_multi_model_bench.sh

# 4. Run the benchmark
./test/run_multi_model_bench.sh                         # Run all chains (default)
./test/run_multi_model_bench.sh -bsc                    # Run BSC cases only
./test/run_multi_model_bench.sh -eth                    # Run ETH cases only
./test/run_multi_model_bench.sh -b                      # Run in background
./test/run_multi_model_bench.sh claude-haiku-4.5        # Run specific model
./test/run_multi_model_bench.sh --list                  # List available models

# Typical usage: run recent cases (>=2025-03), all chains, 30min timeout, background
./test/run_multi_model_bench.sh -since 202503 -all -t 30 -b
```

### Output Files

After running a benchmark, the following files are generated in the output directory (e.g., `logs/20260104_120000/`):

| File | Description |
|------|-------------|
| `<model>_<timestamp>.log` | Full console output with all LLM interactions |
| `benchmark_results_<timestamp>.json` | Structured JSON results |
| `partial_results_<timestamp>.json` | Incremental results (updated after each case) |
| `summary_score.log` | Quick overview: success rate, profit/loss per model |
| `summary_detail.log` | Detailed per-case results with successful cases |
| `raw_data/` | Raw LLM conversation data for analysis |

#### Summary Score Format

```
========================================================================
EVM Hacker Bench - Model Success Rate Summary (Including Losses)
========================================================================

Model                           | Success  | Loss    | Total   | Rate     | Net Profit       | Gross Profit     | Total Loss      
--------------------------------|----------|---------|---------|----------|------------------|------------------|------------------
claude-sonnet-4.5               | 8        | 2       | 34      | 23.5%    |     125000.5000  |     135000.5000  |     -10000.0000
gpt-4o                          | 5        | 1       | 34      | 14.7%    |      85000.2000  |      90000.2000  |      -5000.0000
```

### Command Line Options

#### run_hacker_bench.py (Single Model)

```
python run_hacker_bench.py [OPTIONS]

Dataset Options:
  --dataset {bundled,custom}                 Dataset source (default: bundled)
  --custom-cases PATH                        Path to custom JSON cases file
  --chain CHAIN                              Filter by chain (bsc, mainnet, etc.)
  --case CASE                                Run single case by ID
  --case-ids FILE                            File with case IDs (one per line)
  --max-cases N                              Limit number of cases
  --start-index N                            Resume from index
  -since YYYY-MM                             Filter cases >= date

Model Options:
  --model MODEL                              LLM model name (OpenRouter format)
  --api-key KEY                              API key (or use OPENROUTER_API_KEY)
  --base-url URL                             Custom API base URL
  --temperature T                            LLM temperature (default: 0.7)
  --thinking                                 Enable extended thinking
  --thinking-budget N                        Max thinking tokens (default: 10000)

Execution Options:
  --max-turns N                              Max conversation turns (default: 50)
  --timeout N                                Timeout per turn in seconds (default: 300)
  --session-timeout N                        Max session duration (default: 3600 = 60 min)
  --enable-compression                       Enable message compression (default: True)
  --no-compression                           Disable message compression
  --progress-mode {time,turns}               Progress display mode

Output Options:
  --output-dir DIR                           Output directory (default: logs)
  --no-timestamp-dir                         Use output-dir directly without timestamp
  --no-console-log                           Disable console log file
  --verbose                                  Print detailed debug output

Pre-caching Options:
  --skip-precache                            Skip contract source pre-caching
  --precache-delay N                         Delay between API requests (default: 0.5s)

RPC Options:
  --fork-url URL                             Generic fork URL (all chains)
  --bsc-fork-url URL                         BSC-specific RPC
  --eth-fork-url URL                         Ethereum mainnet RPC
  --arbitrum-fork-url URL                    Arbitrum RPC
  --base-fork-url URL                        Base RPC
  --polygon-fork-url URL                     Polygon RPC

Explorer API Keys:
  --etherscan-api-key KEY                    Etherscan API key
  --bscscan-api-key KEY                      BSCScan API key
  --arbiscan-api-key KEY                     Arbiscan API key
  --basescan-api-key KEY                     BaseScan API key
  --polygonscan-api-key KEY                  PolygonScan API key
```

#### run_parallel_bench.py (Multi-Model Parallel)

```
python run_parallel_bench.py [OPTIONS]

Model Options:
  --models MODEL [MODEL ...]                 Model names to run
                                             Format: "model_name" or "short=model_name:thinking"
  --parallel N                               Max parallel models (default: 2)
  --base-port PORT                           Base port for Anvil (default: 8545)

Dataset Options:
  --chain CHAIN                              Filter by chain
  --case CASE                                Run specific case
  --case-ids FILE                            File with case IDs
  -since YYYY-MM                             Filter cases >= date
  --max-cases N                              Max cases per model

Execution Options:
  --max-turns N                              Max turns per case (default: 120)
  --timeout N                                Timeout per turn (default: 120s)
  --session-timeout N                        Session timeout (default: 1800 = 30min)
  --no-compression                           Disable message compression

Output Options:
  --output-dir DIR                           Output directory (default: logs/parallel)
  --no-timestamp-dir                         Use output-dir directly

Pre-caching Options:
  --skip-precache                            Skip contract pre-caching (not recommended)
  --precache-delay N                         Delay between API requests (default: 0.5s)
```

## Project Structure

```
evm_hacker_bench/
├── run_hacker_bench.py              # Main entry point (single model)
├── run_parallel_bench.py            # Parallel multi-model benchmark
├── generate_summary.py              # Generate summary from existing results
├── calculate_profit_with_loss.py    # Recalculate profit/loss summaries
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Container setup
├── docker-compose.yml               # Docker compose config
│
├── data/
│   ├── cases.json                   # Bundled attack cases (400+)
│   └── exploit_workspace/           # Reference POC files and contract cache
│       └── scone_<case_name>/
│           ├── reference_poc/       # Reference POC files
│           │   └── original_poc.sol
│           └── etherscan-contracts/ # Cached contract source code
│               └── <address>/
│                   └── <ContractName>/
│                       ├── <ContractName>.sol
│                       ├── abi.json
│                       └── metadata.json
│
├── config/
│   ├── system_config.json           # System prompt templates
│   └── env_config_example.txt       # Environment variable examples
│
├── test/
│   └── run_multi_model_bench.sh     # Multi-model benchmark script
│
└── evm_hacker_bench/                # Core library
    ├── __init__.py
    ├── case_loader.py         # Attack case loading & parsing
    ├── prompt_builder.py      # LLM prompt construction
    ├── hacker_controller.py   # LLM conversation management
    ├── parallel_runner.py     # Parallel model execution
    ├── evm_env.py             # Foundry/Anvil environment
    ├── exploit_validator.py   # Success validation & scoring
    ├── contract_fetcher.py    # Block explorer integration & caching
    ├── tool_executor.py       # Tool execution engine
    │
    └── tools/                 # Available tools for LLM
        ├── bash_tool.py       # Shell command execution
        ├── file_editor.py     # File view/edit operations
        ├── slither_tool.py    # Static analysis (Slither)
        ├── solc_manager.py    # Solidity compiler management
        └── uniswap_path.py    # DEX path finding
```

## LLM Tools

The LLM has access to the following tools during exploitation:

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands (forge, cast, etc.) |
| `view_file` | View file contents with optional line range (`start_line`, `end_line`) |
| `edit_file` | Edit files using search/replace |
| `create_file` | Create new files |
| `delete_file` | Delete files |

### view_file Example

```json
{
  "tool": "view_file",
  "path": "src/Exploit.sol",
  "start_line": 10,
  "end_line": 50
}
```

## Supported Attack Categories

| Category | Description |
|----------|-------------|
| `reentrancy` | Reentrancy attacks |
| `flash_loan` | Flash loan exploits |
| `price_manipulation` | Price oracle manipulation |
| `access_control` | Access control bypass |
| `integer_overflow` | Integer overflow/underflow |
| `logic_flaw` | Business logic vulnerabilities |
| `precision_loss` | Precision loss in calculations |
| `front_running` | Front-running attacks |
| `signature` | Signature-related exploits |
| `oracle` | Oracle manipulation |

## Evaluation Metrics

- **Success Rate**: Percentage of cases where LLM successfully exploits the vulnerability
- **Gross Profit**: Sum of all positive profits from successful exploits
- **Total Loss**: Sum of all losses (negative profits) from failed attempts
- **Net Profit**: Gross Profit + Total Loss (actual profit/loss)
- **Turns**: Number of conversation turns to achieve success
- **Token Usage**: LLM token consumption (prompt + completion)

## Contract Source Caching

The benchmark automatically caches contract source code from block explorers to:
- Avoid API rate limits during parallel runs
- Speed up repeated benchmark runs
- Ensure consistent contract code across runs

Cache location: `data/exploit_workspace/<case_id>/etherscan-contracts/<address>/`

To skip pre-caching (not recommended for parallel runs):
```bash
python run_parallel_bench.py --skip-precache ...
```

## Docker Usage

```bash
# Build container
docker build -t evm-hacker-bench .

# Run with docker-compose
docker-compose up

# Or run directly
docker run -it \
    -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
    -e BSC_FORK_URL=$BSC_FORK_URL \
    evm-hacker-bench \
    python run_hacker_bench.py --model anthropic/claude-haiku-4.5 --chain bsc
```

## Supported Models

Any OpenRouter-compatible model can be used. Tested models include:

| Provider | Model | Notes |
|----------|-------|-------|
| Anthropic | `anthropic/claude-opus-4.5` | Best performance |
| Anthropic | `anthropic/claude-sonnet-4.5` | Excellent balance |
| Anthropic | `anthropic/claude-sonnet-4` | Great performance |
| Anthropic | `anthropic/claude-haiku-4.5` | Fast & cost-effective |
| OpenAI | `openai/gpt-4o` | Good baseline |
| OpenAI | `openai/o1` | Extended thinking |
| Google | `google/gemini-2.5-flash` | Fast inference |
| Google | `google/gemini-2.5-pro` | High quality |
| DeepSeek | `deepseek/deepseek-v3` | Cost-effective |
| Qwen | `qwen/qwen3-235b` | Large context |

## RPC Providers

You need archive RPC access for fork testing. Recommended providers:

| Provider | Free Tier | Notes |
|----------|-----------|-------|
| [QuickNode](https://www.quicknode.com/) | Limited | Fast, reliable |
| [Alchemy](https://www.alchemy.com/) | Yes | Good for ETH |
| [Ankr](https://www.ankr.com/) | Yes | Multi-chain |
| [Public RPCs](https://chainlist.org/) | Yes | Rate limited |

## Graceful Termination

The benchmark supports graceful termination via signals:
- **Ctrl+C (SIGINT)**: Stops current case, saves partial results
- **SIGTERM**: Same as SIGINT, for process management

All active subprocesses (Anvil, forge) are properly cleaned up on termination.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

⚠️ **For Research Purposes Only**

This tool is designed for security research and educational purposes. The attack cases are based on historical exploits from DeFiHackLabs. Do not use this tool for malicious purposes.

## Acknowledgments

- [DeFiHackLabs](https://github.com/SunWeb3Sec/DeFiHackLabs) - POC repository
- [SCONE-bench](https://github.com/scone-bench/scone-bench) - Benchmark framework reference
- [Foundry](https://github.com/foundry-rs/foundry) - Ethereum development toolkit

## License

MIT License - see [LICENSE](LICENSE) for details.
