"""
Contract Fetcher - Fetch contract source code from block explorers

Supports:
- Etherscan (mainnet)
- BSCScan (BSC)
- Arbiscan (Arbitrum)
- BaseScan (Base)
- PolygonScan (Polygon)
"""

import os
import json
import time
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class ContractSource:
    """Contract source code and metadata"""
    address: str
    name: str
    source_code: str
    abi: List[Dict]
    compiler_version: str
    is_proxy: bool = False
    implementation_address: Optional[str] = None
    implementation_source: Optional["ContractSource"] = None


# Block explorer API configurations
# Updated to Etherscan API V2 (unified multichain API as of August 2025)
# All chains now use the same base URL with chainid parameter
# See: https://docs.etherscan.io/v2-migration

EXPLORER_CONFIGS = {
    "mainnet": {
        "api_url": "https://api.etherscan.io/v2/api",
        "chain_id": 1,
        "api_key_env": "ETHERSCAN_API_KEY",
        "name": "Etherscan"
    },
    "bsc": {
        "api_url": "https://api.etherscan.io/v2/api",
        "chain_id": 56,
        "api_key_env": "ETHERSCAN_API_KEY",  # V2 uses unified Etherscan API key
        "name": "BSCScan"
    },
    "arbitrum": {
        "api_url": "https://api.etherscan.io/v2/api",
        "chain_id": 42161,
        "api_key_env": "ETHERSCAN_API_KEY",
        "name": "Arbiscan"
    },
    "base": {
        "api_url": "https://api.etherscan.io/v2/api",
        "chain_id": 8453,
        "api_key_env": "ETHERSCAN_API_KEY",
        "name": "BaseScan"
    },
    "polygon": {
        "api_url": "https://api.etherscan.io/v2/api",
        "chain_id": 137,
        "api_key_env": "ETHERSCAN_API_KEY",
        "name": "PolygonScan"
    },
    "optimism": {
        "api_url": "https://api.etherscan.io/v2/api",
        "chain_id": 10,
        "api_key_env": "ETHERSCAN_API_KEY",
        "name": "Optimism Etherscan"
    }
}


class ContractFetcher:
    """
    Fetch and process contract source code from block explorers
    """
    
    def __init__(self, chain: str, api_key: Optional[str] = None):
        """
        Initialize fetcher
        
        Args:
            chain: Chain name (mainnet, bsc, arbitrum, etc.)
            api_key: API key (optional, uses env var if not provided)
        """
        if chain not in EXPLORER_CONFIGS:
            raise ValueError(f"Unsupported chain: {chain}. Supported: {list(EXPLORER_CONFIGS.keys())}")
        
        self.chain = chain
        self.config = EXPLORER_CONFIGS[chain]
        self.api_key = api_key or os.getenv(self.config["api_key_env"], "")
        
        if not self.api_key:
            print(f"⚠️  Warning: No API key for {self.config['name']}. Rate limits may apply.")
    
    def fetch_source(self, address: str) -> Optional[ContractSource]:
        """
        Fetch contract source code
        
        Args:
            address: Contract address
            
        Returns:
            ContractSource or None if not verified
        """
        print(f"📥 Fetching source code for {address} from {self.config['name']}...")
        
        # Etherscan API V2 params (unified multichain API)
        params = {
            "chainid": self.config.get("chain_id", 1),
            "module": "contract",
            "action": "getsourcecode",
            "address": address
        }
        
        if self.api_key:
            params["apikey"] = self.api_key
        
        try:
            response = requests.get(
                self.config["api_url"],
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "1" or not data.get("result"):
                print(f"   ❌ Contract not verified or not found")
                return None
            
            result = data["result"][0]
            
            # Check if contract is verified
            if not result.get("SourceCode"):
                print(f"   ❌ Contract source code not available")
                return None
            
            # Parse source code
            source_code = result["SourceCode"]
            
            # Handle JSON format (multi-file contracts)
            if source_code.startswith("{{"):
                # Remove extra braces
                source_code = source_code[1:-1]
                try:
                    sources = json.loads(source_code)
                    if "sources" in sources:
                        # Flatten all source files
                        all_sources = []
                        for filename, content in sources["sources"].items():
                            all_sources.append(f"// File: {filename}\n{content.get('content', '')}")
                        source_code = "\n\n".join(all_sources)
                    else:
                        # Direct sources dict
                        all_sources = []
                        for filename, content in sources.items():
                            if isinstance(content, dict):
                                all_sources.append(f"// File: {filename}\n{content.get('content', '')}")
                            else:
                                all_sources.append(f"// File: {filename}\n{content}")
                        source_code = "\n\n".join(all_sources)
                except json.JSONDecodeError:
                    pass
            
            # Parse ABI
            abi = []
            if result.get("ABI") and result["ABI"] != "Contract source code not verified":
                try:
                    abi = json.loads(result["ABI"])
                except:
                    pass
            
            # Check if proxy
            is_proxy = result.get("Proxy") == "1"
            implementation_address = result.get("Implementation") if is_proxy else None
            
            # Use lowercase address for directory naming (case-insensitive)
            # This prevents issues with LLM using slightly different case
            normalized_address = address.lower()
            
            contract_source = ContractSource(
                address=normalized_address,
                name=result.get("ContractName", "Unknown"),
                source_code=source_code,
                abi=abi,
                compiler_version=result.get("CompilerVersion", ""),
                is_proxy=is_proxy,
                implementation_address=implementation_address
            )
            
            print(f"   ✓ Found: {contract_source.name}")
            
            # Fetch implementation if proxy
            if is_proxy and implementation_address:
                print(f"   📎 Proxy detected, fetching implementation...")
                time.sleep(0.3)  # Rate limiting
                impl_source = self.fetch_source(implementation_address)
                if impl_source:
                    contract_source.implementation_source = impl_source
            
            return contract_source
            
        except requests.RequestException as e:
            print(f"   ❌ Request failed: {e}")
            return None
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    def save_to_directory(
        self,
        source: ContractSource,
        base_dir: Path
    ) -> Path:
        """
        Save contract source to directory structure
        
        Args:
            source: Contract source
            base_dir: Base directory for contracts
            
        Returns:
            Path to contract directory
        
        Note:
            Uses lowercase addresses for directory names to prevent
            case-sensitivity issues on Linux filesystems.
        """
        # Use lowercase address for directory to prevent case-sensitivity issues
        normalized_address = source.address.lower()
        contract_dir = base_dir / normalized_address / source.name
        contract_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main source
        source_file = contract_dir / f"{source.name}.sol"
        source_file.write_text(source.source_code, encoding='utf-8')
        
        # Save ABI
        if source.abi:
            abi_file = contract_dir / "abi.json"
            abi_file.write_text(json.dumps(source.abi, indent=2), encoding='utf-8')
        
        # Save metadata
        metadata = {
            "address": source.address,
            "name": source.name,
            "compiler_version": source.compiler_version,
            "is_proxy": source.is_proxy,
            "implementation_address": source.implementation_address
        }
        metadata_file = contract_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        
        # Save implementation if exists
        if source.implementation_source:
            impl_normalized_address = source.implementation_address.lower()
            impl_dir = base_dir / impl_normalized_address / source.implementation_source.name
            impl_dir.mkdir(parents=True, exist_ok=True)
            
            impl_source_file = impl_dir / f"{source.implementation_source.name}.sol"
            impl_source_file.write_text(source.implementation_source.source_code, encoding='utf-8')
            
            if source.implementation_source.abi:
                impl_abi_file = impl_dir / "abi.json"
                impl_abi_file.write_text(
                    json.dumps(source.implementation_source.abi, indent=2),
                    encoding='utf-8'
                )
        
        return contract_dir
    
    def get_contract_state(
        self,
        address: str,
        abi: List[Dict],
        rpc_url: str = "http://127.0.0.1:8545",
        query_mappings: bool = True
    ) -> Dict[str, Any]:
        """
        Query contract state variables using web3
        
        Args:
            address: Contract address
            abi: Contract ABI
            rpc_url: RPC endpoint
            query_mappings: Whether to query mapping-type state variables
            
        Returns:
            Dict of state variable names and values
        """
        try:
            from web3 import Web3
            
            w3 = Web3(Web3.HTTPProvider(rpc_url))
            contract = w3.eth.contract(address=Web3.to_checksum_address(address), abi=abi)
            
            state = {}
            mapping_state = {}
            
            # Common addresses to query for mapping values
            # These include the contract itself, common DEX addresses, and zero address
            probe_addresses = [
                address,  # The contract itself
                "0x0000000000000000000000000000000000000000",  # Zero address
                "0x000000000000000000000000000000000000dEaD",  # Dead address
            ]
            
            # Find view/pure functions
            for item in abi:
                if item.get("type") != "function":
                    continue
                if item.get("stateMutability") not in ["view", "pure"]:
                    continue
                
                func_name = item.get("name", "")
                inputs = item.get("inputs", [])
                outputs = item.get("outputs", [])
                
                if not outputs or len(outputs) != 1:
                    continue
                
                output_type = outputs[0].get("type", "unknown")
                
                # Case 1: No inputs - simple state variable
                if not inputs:
                    try:
                        func = getattr(contract.functions, func_name)
                        value = func().call()
                        
                        # Format value based on type
                        if output_type == "address":
                            state[func_name] = {"value": value, "type": "address"}
                            # Add discovered address to probe list for mapping queries
                            if value and value != "0x0000000000000000000000000000000000000000":
                                probe_addresses.append(value)
                        elif output_type.startswith("uint") or output_type.startswith("int"):
                            state[func_name] = {"value": str(value), "type": output_type}
                        elif output_type == "bool":
                            state[func_name] = {"value": value, "type": "bool"}
                        elif output_type == "string":
                            state[func_name] = {"value": value, "type": "string"}
                        else:
                            state[func_name] = {"value": str(value), "type": output_type}
                            
                    except Exception:
                        continue
                
                # Case 2: Single address input - likely a mapping (e.g., balanceOf, allowance)
                elif query_mappings and len(inputs) == 1 and inputs[0].get("type") == "address":
                    # Common mapping getter patterns
                    mapping_patterns = ["balanceOf", "allowance", "getApproved", "isApprovedForAll", 
                                       "nonces", "delegates", "checkpoints", "numCheckpoints"]
                    
                    # Only query for common patterns to avoid too many RPC calls
                    if any(pattern.lower() in func_name.lower() for pattern in mapping_patterns):
                        mapping_results = {}
                        func = getattr(contract.functions, func_name)
                        
                        for probe_addr in probe_addresses[:5]:  # Limit to 5 probes
                            try:
                                checksum_addr = Web3.to_checksum_address(probe_addr)
                                value = func(checksum_addr).call()
                                
                                # Only record non-zero values
                                if value and value != 0:
                                    if output_type.startswith("uint") or output_type.startswith("int"):
                                        mapping_results[probe_addr] = str(value)
                                    else:
                                        mapping_results[probe_addr] = value
                            except Exception:
                                continue
                        
                        # Always record mapping, even if no non-zero samples found
                        # This helps LLM understand the contract structure
                        mapping_state[func_name] = {
                            "type": f"mapping(address => {output_type})",
                            "samples": mapping_results  # May be empty dict
                        }
                
                # Case 3: Two address inputs - likely allowance-style mapping
                elif query_mappings and len(inputs) == 2 and all(i.get("type") == "address" for i in inputs):
                    if "allowance" in func_name.lower():
                        # Query allowance for contract -> common DEX routers
                        dex_routers = [
                            "0x10ED43C718714eb63d5aA57B78B54704E256024E",  # PancakeSwap Router
                            "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",  # Uniswap V2 Router
                        ]
                        
                        allowance_results = {}
                        func = getattr(contract.functions, func_name)
                        
                        for owner in probe_addresses[:3]:
                            for spender in dex_routers:
                                try:
                                    owner_checksum = Web3.to_checksum_address(owner)
                                    spender_checksum = Web3.to_checksum_address(spender)
                                    value = func(owner_checksum, spender_checksum).call()
                                    
                                    if value and value != 0:
                                        key = f"{owner[:10]}...→{spender[:10]}..."
                                        allowance_results[key] = str(value)
                                except Exception:
                                    continue
                        
                        if allowance_results:
                            mapping_state[func_name] = {
                                "type": "mapping(address => mapping(address => uint256))",
                                "samples": allowance_results
                            }
            
            # Merge mapping state into main state with special prefix
            for name, info in mapping_state.items():
                state[f"[mapping]{name}"] = info
            
            return state
            
        except ImportError:
            print("   ⚠️  web3 not installed, skipping state query")
            return {}
        except Exception as e:
            print(f"   ⚠️  State query failed: {e}")
            return {}


def precache_contracts(
    cases: List,
    base_dir: Path,
    delay_seconds: float = 0.5
) -> Dict[str, bool]:
    """
    Pre-cache contract source code for all cases before benchmark starts.
    
    This function fetches contract source code serially to avoid API rate limits,
    and caches them locally for faster access during the actual benchmark run.
    
    Args:
        cases: List of AttackCase objects
        base_dir: Base directory for exploit workspace (e.g., data/exploit_workspace)
        delay_seconds: Delay between API requests to avoid rate limiting
        
    Returns:
        Dict mapping case_id to whether caching was successful
    """
    results = {}
    
    # Deduplicate by (chain, address) to avoid fetching same contract multiple times
    seen = set()
    contracts_to_fetch = []
    
    for case in cases:
        key = (case.chain, case.target_address.lower())
        if key not in seen:
            seen.add(key)
            contracts_to_fetch.append(case)
    
    print(f"\n{'='*70}")
    print(f"📦 PRE-CACHING CONTRACT SOURCE CODE")
    print(f"{'='*70}")
    print(f"Total unique contracts: {len(contracts_to_fetch)}")
    print(f"API delay: {delay_seconds}s between requests")
    print(f"Cache directory: {base_dir}")
    print(f"{'='*70}\n")
    
    cached_count = 0
    fetched_count = 0
    failed_count = 0
    
    for i, case in enumerate(contracts_to_fetch, 1):
        address = case.target_address
        chain = case.chain
        case_dir = base_dir / case.case_id
        contracts_dir = case_dir / "etherscan-contracts"
        address_lower = address.lower()
        address_dir = contracts_dir / address_lower
        
        print(f"[{i}/{len(contracts_to_fetch)}] {case.case_id} ({chain})")
        print(f"   Address: {address}")
        
        # Check if already cached
        cache_exists = False
        if address_dir.exists():
            for subdir in address_dir.iterdir():
                if subdir.is_dir():
                    sol_files = list(subdir.glob("*.sol"))
                    if sol_files:
                        cache_exists = True
                        print(f"   ✅ Already cached: {subdir.name}")
                        cached_count += 1
                        results[case.case_id] = True
                        break
        
        if cache_exists:
            continue
        
        # Fetch from API
        try:
            fetcher = ContractFetcher(chain)
            source = fetcher.fetch_source(address)
            
            if source:
                # Save to cache directory
                contracts_dir.mkdir(parents=True, exist_ok=True)
                fetcher.save_to_directory(source, contracts_dir)
                print(f"   ✅ Fetched and cached: {source.name}")
                fetched_count += 1
                results[case.case_id] = True
            else:
                print(f"   ❌ Failed to fetch (not verified?)")
                failed_count += 1
                results[case.case_id] = False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed_count += 1
            results[case.case_id] = False
        
        # Rate limiting delay (only if we made an API call)
        if i < len(contracts_to_fetch):
            time.sleep(delay_seconds)
    
    print(f"\n{'='*70}")
    print(f"📦 PRE-CACHE COMPLETE")
    print(f"{'='*70}")
    print(f"Already cached: {cached_count}")
    print(f"Newly fetched:  {fetched_count}")
    print(f"Failed:         {failed_count}")
    print(f"{'='*70}\n")
    
    return results


def fetch_contract_for_case(
    address: str,
    chain: str,
    work_dir: Path,
    rpc_url: str = "http://127.0.0.1:8545"
) -> Tuple[Optional[ContractSource], Dict[str, Any]]:
    """
    Convenience function to fetch contract for an attack case
    
    Args:
        address: Contract address
        chain: Chain name
        work_dir: Working directory
        rpc_url: RPC endpoint for state query
        
    Returns:
        (ContractSource, state_variables)
    """
    contracts_dir = work_dir / "etherscan-contracts"
    address_lower = address.lower()
    address_dir = contracts_dir / address_lower
    
    # Check if local cache exists
    source = None
    if address_dir.exists():
        # Find the contract directory (first subdirectory with .sol files)
        for subdir in address_dir.iterdir():
            if subdir.is_dir():
                sol_files = list(subdir.glob("*.sol"))
                abi_file = subdir / "abi.json"
                metadata_file = subdir / "metadata.json"
                
                if sol_files:
                    print(f"📥 Using cached source code for {address}")
                    print(f"   ✓ Found cached: {subdir.name}")
                    
                    # Load from cache
                    source_code = sol_files[0].read_text(encoding='utf-8')
                    
                    abi = []
                    if abi_file.exists():
                        try:
                            abi = json.loads(abi_file.read_text(encoding='utf-8'))
                        except:
                            pass
                    
                    # Check metadata for proxy info
                    is_proxy = False
                    implementation_address = None
                    if metadata_file.exists():
                        try:
                            metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
                            is_proxy = metadata.get('is_proxy', False)
                            implementation_address = metadata.get('implementation_address')
                        except:
                            pass
                    
                    source = ContractSource(
                        address=address,
                        name=subdir.name,
                        source_code=source_code,
                        abi=abi,
                        compiler_version="",
                        is_proxy=is_proxy,
                        implementation_address=implementation_address
                    )
                    break
    
    # Fetch from API if not cached
    if not source:
        fetcher = ContractFetcher(chain)
        source = fetcher.fetch_source(address)
        if not source:
            return None, {}
        
        # Save to directory for future use
        fetcher.save_to_directory(source, contracts_dir)
    
    # Get state if ABI available
    state = {}
    fetcher = ContractFetcher(chain)
    if source.abi:
        state = fetcher.get_contract_state(address, source.abi, rpc_url)
    elif source.implementation_source and source.implementation_source.abi:
        state = fetcher.get_contract_state(
            address,
            source.implementation_source.abi,
            rpc_url
        )
    
    return source, state

