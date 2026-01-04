"""
EVM Hacker Bench - LLM Smart Contract Exploitation Benchmark

A comprehensive benchmark for evaluating LLM ability to exploit 
vulnerable smart contracts on EVM-compatible blockchains.

Refactored based on WebKeyDAO exploit transcript format.
"""

__version__ = "0.2.0"
__author__ = "EVM Hacker Bench Team"

from .evm_env import EVMEnvironment
from .hacker_controller import HackerController, BatchHackerController
from .exploit_validator import ExploitValidator
from .case_loader import CaseLoader
from .prompt_builder import PromptBuilder, FlawVerifierTemplate, ContractInfo
from .tool_executor import ToolExecutor, ToolResult
from .contract_fetcher import ContractFetcher, ContractSource, fetch_contract_for_case, precache_contracts
from .parallel_runner import ParallelModelRunner, ModelConfig, PortManager, create_model_configs

__all__ = [
    "EVMEnvironment",
    "HackerController",
    "BatchHackerController",
    "ExploitValidator",
    "CaseLoader",
    "PromptBuilder",
    "FlawVerifierTemplate",
    "ContractInfo",
    "ToolExecutor",
    "ToolResult",
    "ContractFetcher",
    "ContractSource",
    "fetch_contract_for_case",
    "ParallelModelRunner",
    "ModelConfig",
    "PortManager",
    "create_model_configs"
]

