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
EXPLORER_CONFIGS = {
    "mainnet": {
        "api_url": "https://api.etherscan.io/api",
        "api_key_env": "ETHERSCAN_API_KEY",
        "name": "Etherscan"
    },
    "bsc": {
        "api_url": "https://api.bscscan.com/api",
        "api_key_env": "BSCSCAN_API_KEY",
        "name": "BSCScan"
    },
    "arbitrum": {
        "api_url": "https://api.arbiscan.io/api",
        "api_key_env": "ARBISCAN_API_KEY",
        "name": "Arbiscan"
    },
    "base": {
        "api_url": "https://api.basescan.org/api",
        "api_key_env": "BASESCAN_API_KEY",
        "name": "BaseScan"
    },
    "polygon": {
        "api_url": "https://api.polygonscan.com/api",
        "api_key_env": "POLYGONSCAN_API_KEY",
        "name": "PolygonScan"
    },
    "optimism": {
        "api_url": "https://api-optimistic.etherscan.io/api",
        "api_key_env": "OPTIMISM_API_KEY",
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
        
        params = {
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
            
            contract_source = ContractSource(
                address=address,
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
        """
        contract_dir = base_dir / source.address / source.name
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
            impl_dir = base_dir / source.implementation_address / source.implementation_source.name
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
        rpc_url: str = "http://127.0.0.1:8545"
    ) -> Dict[str, Any]:
        """
        Query contract state variables using web3
        
        Args:
            address: Contract address
            abi: Contract ABI
            rpc_url: RPC endpoint
            
        Returns:
            Dict of state variable names and values
        """
        try:
            from web3 import Web3
            
            w3 = Web3(Web3.HTTPProvider(rpc_url))
            contract = w3.eth.contract(address=Web3.to_checksum_address(address), abi=abi)
            
            state = {}
            
            # Find view/pure functions that look like state getters
            for item in abi:
                if item.get("type") != "function":
                    continue
                if item.get("stateMutability") not in ["view", "pure"]:
                    continue
                if item.get("inputs"):  # Skip functions with inputs
                    continue
                
                func_name = item.get("name", "")
                outputs = item.get("outputs", [])
                
                if not outputs or len(outputs) != 1:
                    continue
                
                try:
                    func = getattr(contract.functions, func_name)
                    value = func().call()
                    
                    output_type = outputs[0].get("type", "unknown")
                    
                    # Format value based on type
                    if output_type == "address":
                        state[func_name] = {"value": value, "type": "address"}
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
            
            return state
            
        except ImportError:
            print("   ⚠️  web3 not installed, skipping state query")
            return {}
        except Exception as e:
            print(f"   ⚠️  State query failed: {e}")
            return {}


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
    fetcher = ContractFetcher(chain)
    
    source = fetcher.fetch_source(address)
    if not source:
        return None, {}
    
    # Save to directory
    contracts_dir = work_dir / "etherscan-contracts"
    fetcher.save_to_directory(source, contracts_dir)
    
    # Get state if ABI available
    state = {}
    if source.abi:
        state = fetcher.get_contract_state(address, source.abi, rpc_url)
    elif source.implementation_source and source.implementation_source.abi:
        state = fetcher.get_contract_state(
            address,
            source.implementation_source.abi,
            rpc_url
        )
    
    return source, state

