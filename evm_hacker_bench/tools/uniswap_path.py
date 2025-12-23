"""
Uniswap Smart Path Tool - DEX Path Finding

Based on SCONE-bench requirements:
- Find optimal swap paths (including multi-hop routes)
- Support PancakeSwap V2/V3, Uniswap V2/V3
- Calculate expected output amounts
"""

import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class SwapPath:
    """Represents a swap path"""
    path: List[str]
    version: str  # V2 or V3
    expected_output: int
    fee_tier: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            "path": self.path,
            "version": self.version,
            "expected_output": self.expected_output,
            "expected_output_formatted": self.expected_output / 10**18,
            "fee_tier": self.fee_tier
        }


class UniswapSmartPath:
    """
    Uniswap Smart Path Finder
    
    Features:
    - Find optimal swap paths across V2/V3
    - Support for multiple DEXes
    - Multi-hop route discovery
    - Slippage calculation
    """
    
    # Common DEX configurations
    DEX_CONFIGS = {
        "pancakeswap_bsc": {
            "v2_factory": "0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73",
            "v2_router": "0x10ED43C718714eb63d5aA57B78B54704E256024E",
            "v3_factory": "0x0BFbCF9fa4f9C56B0F40a671Ad40E0805A091865",
            "v3_quoter": "0xB048Bbc1Ee6b733FFfCFb9e9CeF7375518e25997",
            "weth": "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c"  # WBNB
        },
        "uniswap_mainnet": {
            "v2_factory": "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
            "v2_router": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
            "v3_factory": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
            "v3_quoter": "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
            "weth": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        },
        "uniswap_arbitrum": {
            "v3_factory": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
            "v3_quoter": "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
            "weth": "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1"
        }
    }
    
    # Common pivot tokens
    PIVOT_TOKENS = {
        "bsc": [
            "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",  # WBNB
            "0x55d398326f99059fF775485246999027B3197955",  # USDT
            "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",  # USDC
            "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56"   # BUSD
        ],
        "mainnet": [
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
            "0x6B175474E89094C44Da98b954EescdeCB5BE3830"   # DAI
        ]
    }
    
    def __init__(
        self,
        rpc_url: str = "http://127.0.0.1:8545",
        dex: str = "pancakeswap_bsc",
        tool_path: Optional[str] = None
    ):
        """
        Initialize path finder
        
        Args:
            rpc_url: RPC endpoint URL
            dex: DEX configuration name
            tool_path: Path to uniswap-smart-path binary
        """
        self.rpc_url = rpc_url
        self.dex = dex
        self.config = self.DEX_CONFIGS.get(dex, self.DEX_CONFIGS["pancakeswap_bsc"])
        self.tool_path = tool_path or "/usr/local/bin/uniswap-smart-path"
    
    def find_path(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
        pivot_tokens: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Find optimal swap path
        
        Args:
            token_in: Input token address
            token_out: Output token address
            amount_in: Input amount (in wei)
            pivot_tokens: Custom pivot tokens for multi-hop
            
        Returns:
            Best path and expected output
        """
        # Try using external tool if available
        result = self._use_external_tool(token_in, token_out, amount_in, pivot_tokens)
        if result.get("success"):
            return result
        
        # Fallback: basic path calculation
        return self._calculate_basic_path(token_in, token_out, amount_in)
    
    def _use_external_tool(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
        pivot_tokens: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Try to use external uniswap-smart-path tool"""
        if not Path(self.tool_path).exists():
            return {"success": False, "error": "Tool not found"}
        
        # Determine chain pivot tokens
        chain = "bsc" if "pancake" in self.dex else "mainnet"
        pivots = pivot_tokens or self.PIVOT_TOKENS.get(chain, [])
        
        cmd = [
            self.tool_path,
            "--rpc-endpoint", self.rpc_url,
            "--token-in", token_in,
            "--token-out", token_out,
            "--exact-amount-in", str(amount_in),
            "--pivot-tokens", ",".join(pivots)
        ]
        
        # Add DEX-specific configs
        if "v2_factory" in self.config:
            cmd.extend(["--v2-factory", self.config["v2_factory"]])
            cmd.extend(["--v2-router", self.config["v2_router"]])
        
        if "v3_factory" in self.config:
            cmd.extend(["--v3-factory", self.config["v3_factory"]])
            cmd.extend(["--v3-quoter", self.config["v3_quoter"]])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            return self._parse_tool_output(result.stdout)
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _parse_tool_output(self, output: str) -> Dict[str, Any]:
        """Parse uniswap-smart-path output"""
        try:
            # Try to extract path and output from stdout
            lines = output.split('\n')
            
            path = None
            expected_output = 0
            version = "V2"
            
            for line in lines:
                if "best swap path" in line.lower():
                    continue
                if "path:" in line.lower():
                    # Extract path tuple/list
                    path_str = line.split(":", 1)[1].strip()
                    # Parse path from string like ('0x...', '0x...', '0x...')
                    import ast
                    try:
                        path = list(ast.literal_eval(path_str))
                    except:
                        pass
                elif "estimated output" in line.lower():
                    try:
                        expected_output = int(line.split(":")[-1].strip().split()[0])
                    except:
                        pass
                elif "v3" in line.lower():
                    version = "V3"
            
            if path:
                return {
                    "success": True,
                    "path": path,
                    "version": version,
                    "expected_output": expected_output,
                    "expected_output_formatted": expected_output / 10**18,
                    "raw_output": output
                }
            
            return {
                "success": True,
                "raw_output": output,
                "path": None
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "raw_output": output}
    
    def _calculate_basic_path(
        self,
        token_in: str,
        token_out: str,
        amount_in: int
    ) -> Dict[str, Any]:
        """
        Basic path calculation (without external tool)
        
        Returns direct path suggestion
        """
        # Get WETH/WBNB for the chain
        weth = self.config.get("weth")
        
        # Suggest common paths
        paths = [
            # Direct path
            [token_in, token_out],
        ]
        
        # Add WETH hop if not already in path
        if weth and token_in != weth and token_out != weth:
            paths.append([token_in, weth, token_out])
        
        # Add stablecoin hops
        chain = "bsc" if "pancake" in self.dex else "mainnet"
        for stable in self.PIVOT_TOKENS.get(chain, [])[:2]:
            if stable != token_in and stable != token_out:
                paths.append([token_in, stable, token_out])
        
        return {
            "success": True,
            "suggested_paths": paths,
            "note": "External tool not available, using suggested paths",
            "v2_router": self.config.get("v2_router"),
            "v3_router": self.config.get("v3_quoter")
        }
    
    def get_dex_addresses(self) -> Dict[str, str]:
        """Get DEX contract addresses"""
        return self.config
    
    def format_for_solidity(self, path: List[str]) -> str:
        """
        Format path for Solidity code
        
        Args:
            path: List of token addresses
            
        Returns:
            Solidity array initialization string
        """
        addresses = ", ".join(path)
        return f"new address[]({len(path)})"
    
    def generate_swap_code(
        self,
        path: List[str],
        amount_in: str,
        version: str = "V2"
    ) -> str:
        """
        Generate Solidity swap code
        
        Args:
            path: Swap path
            amount_in: Amount variable name
            version: V2 or V3
            
        Returns:
            Solidity code snippet
        """
        router = self.config.get("v2_router") if version == "V2" else self.config.get("v3_router")
        
        if version == "V2":
            path_code = f"address[] memory path = new address[]({len(path)});\n"
            for i, addr in enumerate(path):
                path_code += f"        path[{i}] = {addr};\n"
            
            return f"""
        // Swap via {self.dex} V2
        {path_code}
        IUniswapV2Router({router}).swapExactTokensForTokensSupportingFeeOnTransferTokens(
            {amount_in},
            0,
            path,
            address(this),
            block.timestamp
        );
"""
        else:
            return f"""
        // Swap via {self.dex} V3
        // TODO: Implement V3 swap
"""

