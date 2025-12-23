# EVM Hacker Bench

> A comprehensive benchmark for evaluating LLM ability to exploit vulnerable smart contracts on EVM-compatible blockchains.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

EVM Hacker Bench is a benchmark framework that evaluates Large Language Models' (LLMs) capabilities in understanding and exploiting smart contract vulnerabilities. It provides:

- **400+ Real-World Attack Cases** from DeFiHackLabs POC repository
- **Multi-Chain Support**: BSC, Ethereum Mainnet, Arbitrum, Base, Polygon
- **Multi-Model Support**: OpenRouter API compatible (Claude, GPT, Gemini, etc.)
- **Automated Evaluation**: Fork-based testing with Foundry (anvil + forge)
- **Detailed Metrics**: Success rate, profit calculation, turn analysis

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
│  │  - Profit calculation                                    │   │
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

3. **DeFiHackLabs POC Repository**
   ```bash
   git clone https://github.com/SunWeb3Sec/DeFiHackLabs.git ../DeFiHackLabs
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

#### Single Case Test

```bash
# Run on a specific case
python run_hacker_bench.py \
    --model anthropic/claude-sonnet-4 \
    --case gamma \
    --fork-url $BSC_FORK_URL
```

#### Batch Benchmark

```bash
# Run all BSC cases with a specific model
python run_hacker_bench.py \
    --model anthropic/claude-haiku-4.5 \
    --chain bsc \
    --max-turns 30 \
    --bsc-fork-url $BSC_FORK_URL
```

#### Multi-Model Benchmark (via shell script)

```bash
# Run benchmark on all models (BSC only)
./run_multi_model_bench.sh -bsc

# Run on all chains (~400+ cases)
./run_multi_model_bench.sh -all

# Run in background
./run_multi_model_bench.sh -b -all

# Run specific model
./run_multi_model_bench.sh claude-haiku-4.5
```

### Command Line Options

```
python run_hacker_bench.py [OPTIONS]

Dataset Options:
  --dataset {scone,defihacklabs,combined}  Dataset source
  --chain CHAIN                            Filter by chain (bsc, mainnet, etc.)
  --case CASE                              Run single case by ID
  --max-cases N                            Limit number of cases
  --start-index N                          Resume from index

Model Options:
  --model MODEL                            LLM model name (OpenRouter format)
  --api-key KEY                            API key (or use OPENROUTER_API_KEY)
  --api-base URL                           Custom API base URL
  --thinking                               Enable extended thinking
  --thinking-budget N                      Max thinking tokens (default: 10000)

Execution Options:
  --max-turns N                            Max conversation turns (default: 30)
  --output-dir DIR                         Output directory for results

RPC Options:
  --fork-url URL                           Generic fork URL (all chains)
  --bsc-fork-url URL                       BSC-specific RPC
  --eth-fork-url URL                       Ethereum mainnet RPC
  --arbitrum-fork-url URL                  Arbitrum RPC
  --base-fork-url URL                      Base RPC
  --polygon-fork-url URL                   Polygon RPC
```

## Project Structure

```
evm_hacker_bench/
├── run_hacker_bench.py        # Main entry point
├── run_multi_model_bench.sh   # Multi-model batch runner
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container setup
├── docker-compose.yml         # Docker compose config
│
├── config/
│   ├── system_config.json     # System prompt templates
│   └── env_config_example.txt # Environment variable examples
│
└── evm_hacker_bench/          # Core library
    ├── __init__.py
    ├── case_loader.py         # Attack case loading & parsing
    ├── prompt_builder.py      # LLM prompt construction
    ├── hacker_controller.py   # LLM conversation management
    ├── evm_env.py             # Foundry/Anvil environment
    ├── exploit_validator.py   # Success validation & scoring
    ├── contract_fetcher.py    # Block explorer integration
    ├── tool_executor.py       # Tool execution engine
    │
    └── tools/                 # Available tools for LLM
        ├── bash_tool.py       # Shell command execution
        ├── file_editor.py     # File view/edit operations
        ├── slither_tool.py    # Static analysis (Slither)
        ├── solc_manager.py    # Solidity compiler management
        └── uniswap_path.py    # DEX path finding
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
- **Profit**: Native token profit from successful exploits
- **Turns**: Number of conversation turns to achieve success
- **Token Usage**: LLM token consumption (prompt + completion)

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
| Anthropic | `anthropic/claude-sonnet-4` | Best performance |
| Anthropic | `anthropic/claude-haiku-4.5` | Fast & cost-effective |
| OpenAI | `openai/gpt-4o` | Good baseline |
| Google | `google/gemini-2.5-flash` | Fast inference |
| DeepSeek | `deepseek/deepseek-v3` | Cost-effective |

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
