"""
Uniswap/PancakeSwap Path Finding Tool

Pure implementation using cast commands for V2 path finding.
No external library dependency - fully compatible with PancakeSwap V2 and Uniswap V2.

Supports: BSC (PancakeSwap), Ethereum mainnet (Uniswap), Arbitrum (Uniswap)
"""

import subprocess
import os
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# DEX configurations for different chains
DEX_CONFIGS = {
    "bsc": {
        "name": "PancakeSwap",
        "native_symbol": "BNB",
        "wrapped_native": "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",  # WBNB
        "pivot_tokens": [
            ("0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c", "WBNB", 18),
            ("0x55d398326f99059fF775485246999027B3197955", "USDT", 18),
            ("0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d", "USDC", 18),
            ("0x1AF3F329e8BE154074D8769D1FFa4eE058B1DBc3", "DAI", 18),
            ("0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56", "BUSD", 18),
        ],
        "v2_router": "0x10ED43C718714eb63d5aA57B78B54704E256024E",
        "v2_factory": "0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73",
    },
    "mainnet": {
        "name": "Uniswap",
        "native_symbol": "ETH",
        "wrapped_native": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
        "pivot_tokens": [
            ("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "WETH", 18),
            ("0xdAC17F958D2ee523a2206206994597C13D831ec7", "USDT", 6),
            ("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "USDC", 6),
            ("0x6B175474E89094C44Da98b954EecdeCB5bAd78d9", "DAI", 18),
        ],
        "v2_router": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
        "v2_factory": "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
    },
    "arbitrum": {
        "name": "Uniswap",
        "native_symbol": "ETH",
        "wrapped_native": "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",  # WETH
        "pivot_tokens": [
            ("0x82aF49447D8a07e3bd95BD0d56f35241523fBab1", "WETH", 18),
            ("0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9", "USDT", 6),
            ("0xaf88d065e77c8cC2239327C5EDb3A432268e5831", "USDC", 6),
            ("0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1", "DAI", 18),
        ],
        "v2_router": "0x4752ba5dbc23f44d87826276bf6fd6b1c372ad24",
        "v2_factory": "0xf1D7CC64Fb4452F05c498126312eBE29f30Fbcf9",
    },
}


@dataclass
class PathResult:
    """Result of a path query"""
    path: List[str]
    path_symbols: List[str]
    amount_out: int
    amount_out_formatted: float
    is_valid: bool
    error: Optional[str] = None


class DEXPathFinder:
    """
    DEX V2 Path Finder using cast commands
    
    Features:
    - Find optimal V2 swap paths
    - Support for PancakeSwap, Uniswap V2
    - Multi-hop route discovery via pivot tokens
    - Expected output calculation using getAmountsOut
    """
    
    def __init__(
        self,
        rpc_url: str = "http://127.0.0.1:8545",
        chain: str = "bsc",
        cast_path: Optional[str] = None,
    ):
        """
        Initialize path finder
        
        Args:
            rpc_url: RPC endpoint URL
            chain: Chain name (bsc, mainnet, arbitrum)
            cast_path: Path to cast binary (auto-detected if not provided)
        """
        self.rpc_url = rpc_url
        self.chain = chain
        self.config = DEX_CONFIGS.get(chain, DEX_CONFIGS["bsc"])
        self.cast_path = cast_path or self._find_cast()
    
    def _find_cast(self) -> str:
        """Find cast binary path"""
        paths = [
            os.path.expanduser("~/.foundry/bin/cast"),
            "/usr/local/bin/cast",
            "cast"
        ]
        for path in paths:
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return path
            except:
                continue
        return "cast"
    
    def _run_cast(self, args: List[str], timeout: int = 30) -> Tuple[bool, str]:
        """
        Run cast command
        
        Returns:
            (success, output/error)
        """
        # Clean environment (remove proxy)
        env = os.environ.copy()
        for var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
            env.pop(var, None)
        env['no_proxy'] = '*'
        
        try:
            result = subprocess.run(
                [self.cast_path] + args,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )
            
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                return False, result.stderr.strip()
                
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)
    
    def get_pair(self, token0: str, token1: str) -> Optional[str]:
        """
        Get pair address from V2 factory
        
        Returns:
            Pair address or None if not found
        """
        success, output = self._run_cast([
            "call",
            self.config["v2_factory"],
            "getPair(address,address)(address)",
            token0, token1,
            "--rpc-url", self.rpc_url
        ])
        
        if not success:
            return None
        
        pair = output.strip()
        if pair == "0x0000000000000000000000000000000000000000":
            return None
        
        return pair
    
    def get_amounts_out(self, amount_in: int, path: List[str]) -> Optional[List[int]]:
        """
        Get expected output amounts for a path using V2 router
        
        Args:
            amount_in: Input amount in wei
            path: List of token addresses
            
        Returns:
            List of amounts [amountIn, ..., amountOut] or None if failed
        """
        # Format path as array for cast
        path_str = "[" + ",".join(path) + "]"
        
        success, output = self._run_cast([
            "call",
            self.config["v2_router"],
            "getAmountsOut(uint256,address[])(uint256[])",
            str(amount_in),
            path_str,
            "--rpc-url", self.rpc_url
        ])
        
        if not success:
            logger.debug(f"getAmountsOut failed: {output}")
            return None
        
        try:
            # Parse output from cast
            # cast returns format like: [1000000000000000000 [1e18], 898605310168070069996 [8.986e20]]
            # We need to extract the raw numbers before the [scientific notation]
            output = output.strip()
            
            if not output.startswith("["):
                logger.debug(f"Unexpected output format: {output}")
                return None
            
            # Remove outer brackets
            inner = output[1:-1]
            
            # Split by ", " and extract first number from each part
            # e.g., "1000000000000000000 [1e18]" -> "1000000000000000000"
            amounts = []
            parts = inner.split(", ")
            
            for part in parts:
                # Extract the number before any space or bracket
                # Handle: "1000000000000000000 [1e18]" or just "1000000000000000000"
                num_str = part.split()[0].split("[")[0].strip()
                
                # Remove any trailing brackets or spaces
                num_str = num_str.rstrip("]").strip()
                
                if num_str:
                    amounts.append(int(num_str))
            
            if len(amounts) >= 2:
                return amounts
            
            logger.debug(f"Failed to parse enough amounts from: {output}")
            return None
            
        except Exception as e:
            logger.debug(f"Failed to parse amounts: {e}, output: {output}")
        
        return None
    
    def _check_path_valid(self, path: List[str]) -> bool:
        """Check if all pairs in path exist"""
        for i in range(len(path) - 1):
            pair = self.get_pair(path[i], path[i + 1])
            if pair is None:
                return False
        return True
    
    def _get_path_output(self, amount_in: int, path: List[str]) -> PathResult:
        """Get output amount for a specific path"""
        # Get token info map: address -> (symbol, decimals)
        pivot_map = {addr.lower(): (symbol, decimals) for addr, symbol, decimals in self.config["pivot_tokens"]}
        
        # Build path symbols
        path_symbols = []
        for addr in path:
            info = pivot_map.get(addr.lower())
            if info:
                path_symbols.append(info[0])  # symbol
            else:
                path_symbols.append(addr[:8] + "...")
        
        amounts = self.get_amounts_out(amount_in, path)
        
        if amounts is None or len(amounts) < 2:
            return PathResult(
                path=path,
                path_symbols=path_symbols,
                amount_out=0,
                amount_out_formatted=0,
                is_valid=False,
                error="Failed to get amounts"
            )
        
        amount_out = amounts[-1]
        
        # Get decimals for output token (last in path)
        output_addr = path[-1].lower()
        output_info = pivot_map.get(output_addr)
        output_decimals = output_info[1] if output_info else 18
        
        return PathResult(
            path=path,
            path_symbols=path_symbols,
            amount_out=amount_out,
            amount_out_formatted=amount_out / (10 ** output_decimals),
            is_valid=True
        )
    
    def find_best_path(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
    ) -> Dict[str, Any]:
        """
        Find the best swap path between two tokens
        
        Args:
            token_in: Input token address
            token_out: Output token address
            amount_in: Input amount in wei
            
        Returns:
            Dict with best path and all tried paths
        """
        results = []
        
        # 1. Try direct path
        direct_path = [token_in, token_out]
        direct_result = self._get_path_output(amount_in, direct_path)
        if direct_result.is_valid:
            results.append(("direct", direct_result))
        
        # 2. Try paths via pivot tokens
        for pivot_addr, pivot_symbol, _ in self.config["pivot_tokens"]:
            # Skip if pivot is same as input or output
            if pivot_addr.lower() == token_in.lower() or pivot_addr.lower() == token_out.lower():
                continue
            
            # Path: token_in -> pivot -> token_out
            pivot_path = [token_in, pivot_addr, token_out]
            pivot_result = self._get_path_output(amount_in, pivot_path)
            
            if pivot_result.is_valid:
                results.append((f"via_{pivot_symbol}", pivot_result))
        
        # 3. Try 3-hop paths via wrapped native + stablecoin
        wrapped = self.config["wrapped_native"]
        stables = [addr for addr, sym, _ in self.config["pivot_tokens"] if sym in ["USDT", "USDC", "BUSD"]]
        
        for stable in stables[:2]:  # Try USDT and USDC
            if (wrapped.lower() != token_in.lower() and 
                wrapped.lower() != token_out.lower() and
                stable.lower() != token_in.lower() and 
                stable.lower() != token_out.lower()):
                
                # Path: token_in -> wrapped -> stable -> token_out
                three_hop = [token_in, wrapped, stable, token_out]
                three_hop_result = self._get_path_output(amount_in, three_hop)
                
                if three_hop_result.is_valid:
                    stable_sym = next((s for a, s, _ in self.config["pivot_tokens"] if a.lower() == stable.lower()), "STABLE")
                    native_sym = "WBNB" if self.chain == "bsc" else "WETH"
                    results.append((f"via_{native_sym}_{stable_sym}", three_hop_result))
        
        # Sort by output amount (descending)
        results.sort(key=lambda x: x[1].amount_out, reverse=True)
        
        if not results:
            return {
                "success": False,
                "error": "No valid path found",
                "chain": self.chain,
                "dex": self.config["name"],
                "token_in": token_in,
                "token_out": token_out,
                "amount_in": amount_in,
                "suggested_paths": self._get_suggested_paths(token_in, token_out),
            }
        
        best_type, best_result = results[0]
        
        return {
            "success": True,
            "chain": self.chain,
            "dex": self.config["name"],
            "token_in": token_in,
            "token_out": token_out,
            "amount_in": amount_in,
            "amount_in_formatted": amount_in / 10**18,
            "best_path": {
                "type": best_type,
                "path": best_result.path,
                "path_display": " → ".join(best_result.path_symbols),
                "amount_out": best_result.amount_out,
                "amount_out_formatted": best_result.amount_out_formatted,
            },
            "all_paths": [
                {
                    "type": t,
                    "path": r.path,
                    "path_display": " → ".join(r.path_symbols),
                    "amount_out": r.amount_out,
                    "amount_out_formatted": r.amount_out_formatted,
                }
                for t, r in results
            ],
            "v2_router": self.config["v2_router"],
        }
    
    def _get_suggested_paths(self, token_in: str, token_out: str) -> List[Dict]:
        """Generate suggested paths when no valid path found"""
        paths = []
        
        # Direct
        paths.append({
            "path": [token_in, token_out],
            "type": "direct",
            "note": "Direct swap"
        })
        
        # Via wrapped native
        wrapped = self.config["wrapped_native"]
        native_sym = "WBNB" if self.chain == "bsc" else "WETH"
        if wrapped.lower() != token_in.lower() and wrapped.lower() != token_out.lower():
            paths.append({
                "path": [token_in, wrapped, token_out],
                "type": f"via_{native_sym}",
                "note": f"Via {native_sym}"
            })
        
        # Via stablecoins
        for addr, symbol, _ in self.config["pivot_tokens"][1:3]:
            if addr.lower() != token_in.lower() and addr.lower() != token_out.lower():
                paths.append({
                    "path": [token_in, addr, token_out],
                    "type": f"via_{symbol}",
                    "note": f"Via {symbol}"
                })
        
        return paths
    
    def get_dex_info(self) -> Dict[str, Any]:
        """Get DEX configuration info"""
        return {
            "chain": self.chain,
            "name": self.config["name"],
            "native_symbol": self.config["native_symbol"],
            "v2_router": self.config["v2_router"],
            "v2_factory": self.config["v2_factory"],
            "pivot_tokens": [
                {"address": addr, "symbol": sym, "decimals": dec}
                for addr, sym, dec in self.config["pivot_tokens"]
            ],
        }
    
    def generate_swap_code(
        self,
        path: List[str],
        amount_var: str = "amountIn",
        min_out: str = "0",
    ) -> str:
        """
        Generate Solidity V2 swap code
        
        Args:
            path: Swap path addresses
            amount_var: Variable name for input amount
            min_out: Minimum output amount
            
        Returns:
            Solidity code snippet
        """
        router = self.config["v2_router"]
        
        path_code = f"address[] memory path = new address[]({len(path)});"
        for i, addr in enumerate(path):
            path_code += f"\n        path[{i}] = {addr};"
        
        return f"""// Swap via {self.config['name']} V2
        {path_code}

        // Approve router
        IERC20(path[0]).approve({router}, {amount_var});

        // Execute swap
        IRouter({router}).swapExactTokensForTokensSupportingFeeOnTransferTokens(
            {amount_var},
            {min_out},
            path,
            address(this),
            block.timestamp
        );"""


# Backward compatibility aliases
UniswapSmartPath = DEXPathFinder


def find_swap_path(
    token_in: str,
    token_out: str,
    amount_in: int,
    chain: str = "bsc",
    rpc_url: str = "http://127.0.0.1:8545",
) -> Dict[str, Any]:
    """
    Find optimal swap path - convenience function
    
    Args:
        token_in: Input token address
        token_out: Output token address  
        amount_in: Input amount in wei (e.g., 1000000000000000000 for 1 token with 18 decimals)
        chain: Chain name (bsc, mainnet, arbitrum)
        rpc_url: RPC endpoint URL
        
    Returns:
        Dict with best path and expected output
    """
    finder = DEXPathFinder(rpc_url=rpc_url, chain=chain)
    return finder.find_best_path(token_in, token_out, amount_in)
