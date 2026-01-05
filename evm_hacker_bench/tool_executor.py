"""
Tool Executor - Execute tools called by LLM

Provides tool execution for:
- bash: Execute shell commands
- view_file: View file/directory contents
- edit_file: Edit files with string replacement
- fetch_contract: Fetch contract source code from Etherscan

Based on the WebKeyDAO exploit transcript format.
"""

import os
import json
import subprocess
import requests
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


# Etherscan API V2 configuration (unified multichain)
# See: https://docs.etherscan.io/api-reference/endpoint/getsourcecode
ETHERSCAN_CONFIG = {
    "mainnet": {"chain_id": 1, "name": "Ethereum"},
    "bsc": {"chain_id": 56, "name": "BSC"},
    "arbitrum": {"chain_id": 42161, "name": "Arbitrum"},
    "base": {"chain_id": 8453, "name": "Base"},
    "polygon": {"chain_id": 137, "name": "Polygon"},
    "optimism": {"chain_id": 10, "name": "Optimism"},
}

# DEX Factory addresses for get_pair queries
# Uniswap V2 / PancakeSwap V2 compatible factories
DEX_FACTORY_CONFIG = {
    "bsc": {
        "v2": "0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73",  # PancakeSwap V2
        "name": "PancakeSwap"
    },
    "mainnet": {
        "v2": "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",  # Uniswap V2
        "name": "Uniswap"
    },
    "arbitrum": {
        "v2": "0xf1D7CC64Fb4452F05c498126312eBE29f30Fbcf9",  # Uniswap V2
        "name": "Uniswap"
    },
    "base": {
        "v2": "0x8909Dc15e40173Ff4699343b6eB8132c65e18eC6",  # Uniswap V2
        "name": "Uniswap"
    },
    "polygon": {
        "v2": "0x5757371414417b8C6CAad45bAeF941aBc7d3Ab32",  # QuickSwap
        "name": "QuickSwap"
    },
    "optimism": {
        "v2": "0x0c3c1c532F1e39EdF36BE9Fe0bE1410313E074Bf",  # Velodrome
        "name": "Velodrome"
    },
}


@dataclass
class ToolResult:
    """Result of a tool execution"""
    success: bool
    output: str
    error: Optional[str] = None


class ToolExecutor:
    """
    Execute tools called by LLM during exploitation
    
    Supports:
    - bash: Execute shell commands
    - view_file: View file/directory contents
    - edit_file: Edit files
    - fetch_contract: Fetch contract source from Etherscan
    """
    
    def __init__(
        self,
        work_dir: Path,
        timeout: int = 120,
        anvil_port: int = 8545,
        chain: str = "bsc"
    ):
        """
        Initialize tool executor
        
        Args:
            work_dir: Working directory for all operations
            timeout: Default timeout for commands
            anvil_port: Anvil RPC port
            chain: Blockchain network (bsc, mainnet, etc.)
        """
        # Resolve to get canonical path (removes ../ components)
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.anvil_port = anvil_port
        self.chain = chain
        
        # Etherscan API key from environment
        self.etherscan_api_key = os.getenv("ETHERSCAN_API_KEY", "")
        
        # Proxy settings for Etherscan API (may be needed in some environments)
        self.proxy = os.getenv("ETHERSCAN_PROXY", os.getenv("HTTPS_PROXY", os.getenv("https_proxy", "")))
        
        # Find forge command
        self.forge_cmd = self._find_command("forge")
        self.cast_cmd = self._find_command("cast")
        
        # Execution history for debugging
        self.history: List[Dict[str, Any]] = []
    
    def _find_command(self, cmd: str) -> str:
        """Find command path"""
        paths = [
            os.path.expanduser(f"~/.foundry/bin/{cmd}"),
            f"/usr/local/bin/{cmd}",
            cmd
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
        return cmd
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool by name
        
        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments
            
        Returns:
            ToolResult with output
        """
        # Record in history
        self.history.append({
            "tool": tool_name,
            "arguments": arguments,
            "result": None
        })
        
        try:
            if tool_name == "bash":
                result = self._execute_bash(arguments.get("command", ""))
            elif tool_name == "view_file":
                # Support line range: start_line, end_line, or view_range [start, end]
                start_line = arguments.get("start_line")
                end_line = arguments.get("end_line")
                
                # Also support view_range format: [start, end]
                view_range = arguments.get("view_range")
                if view_range and isinstance(view_range, list) and len(view_range) >= 2:
                    start_line = view_range[0]
                    end_line = view_range[1]
                
                result = self._view_file(
                    arguments.get("path", ""),
                    start_line=start_line,
                    end_line=end_line
                )
            elif tool_name == "edit_file":
                result = self._edit_file(
                    arguments.get("path", ""),
                    arguments.get("old_str", ""),
                    arguments.get("new_str", "")
                )
            elif tool_name == "write_file":
                result = self._write_file(
                    arguments.get("path", ""),
                    arguments.get("content", "")
                )
            elif tool_name == "fetch_contract":
                result = self._fetch_contract(
                    arguments.get("address", ""),
                    arguments.get("chain", self.chain)
                )
            elif tool_name == "get_pair":
                result = self._get_pair(
                    arguments.get("token0", ""),
                    arguments.get("token1", ""),
                    arguments.get("chain", self.chain)
                )
            elif tool_name == "find_swap_path":
                result = self._find_swap_path(
                    arguments.get("token_in", ""),
                    arguments.get("token_out", ""),
                    arguments.get("amount_in", 0),
                    arguments.get("chain", self.chain)
                )
            else:
                result = ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown tool: {tool_name}"
                )
            
            self.history[-1]["result"] = result
            return result
            
        except Exception as e:
            result = ToolResult(
                success=False,
                output="",
                error=str(e)
            )
            self.history[-1]["result"] = result
            return result
    
    def _execute_bash(self, command: str) -> ToolResult:
        """
        Execute a bash command
        
        Args:
            command: Command to execute
            
        Returns:
            ToolResult with command output
        """
        if not command:
            return ToolResult(
                success=False,
                output="",
                error="No command provided"
            )
        
        # Replace command names with full paths (only at word boundaries)
        # Use regex to avoid replacing substrings like "forge_output.txt"
        import re
        # Replace standalone 'forge' command (at start of command or after shell operators)
        command = re.sub(r'(^|[;&|]\s*|&&\s*|\|\|\s*)forge(\s)', rf'\1{self.forge_cmd}\2', command)
        command = re.sub(r'(^|[;&|]\s*|&&\s*|\|\|\s*)cast(\s)', rf'\1{self.cast_cmd}\2', command)
        
        # Create clean environment (remove proxy settings)
        env = os.environ.copy()
        proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 
                      'all_proxy', 'ALL_PROXY']
        for var in proxy_vars:
            env.pop(var, None)
        env['no_proxy'] = '*'
        env['NO_PROXY'] = '*'
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.work_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env
            )
            
            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr if output else result.stderr
            
            # No truncation - full output preserved for summary
            return ToolResult(
                success=result.returncode == 0,
                output=output.strip(),
                error=None if result.returncode == 0 else f"Exit code: {result.returncode}"
            )
            
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Command timed out after {self.timeout}s"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )
    
    def _view_file(
        self, 
        path: str, 
        start_line: Optional[int] = None,
        end_line: Optional[int] = None
    ) -> ToolResult:
        """
        View file or directory contents
        
        Args:
            path: Path to view
            start_line: Optional start line number (1-indexed, inclusive)
            end_line: Optional end line number (1-indexed, inclusive)
            
        Returns:
            ToolResult with file contents or directory listing
        """
        if not path:
            return ToolResult(
                success=False,
                output="",
                error="No path provided"
            )
        
        # Resolve path
        if path.startswith("/"):
            full_path = Path(path)
        else:
            full_path = self.work_dir / path
        
        if not full_path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"Path not found: {path}"
            )
        
        try:
            if full_path.is_dir():
                # List directory with depth limit
                output = self._list_directory(full_path, max_depth=2)
            else:
                # Read file with line numbers
                content = full_path.read_text(encoding='utf-8', errors='ignore')
                lines = content.split('\n')
                total_lines = len(lines)
                
                # Handle line range if specified
                if start_line is not None or end_line is not None:
                    # Convert to integers (may come as strings from JSON)
                    try:
                        start_line_int = int(start_line) if start_line is not None else None
                        end_line_int = int(end_line) if end_line is not None else None
                    except (ValueError, TypeError):
                        return ToolResult(
                            success=False,
                            output="",
                            error=f"Invalid line numbers: start_line={start_line}, end_line={end_line}"
                        )
                    
                    # Convert to 0-indexed for slicing
                    start_idx = (start_line_int - 1) if start_line_int else 0
                    end_idx = end_line_int if end_line_int else total_lines
                    
                    # Clamp to valid range
                    start_idx = max(0, min(start_idx, total_lines - 1))
                    end_idx = max(start_idx + 1, min(end_idx, total_lines))
                    
                    # Slice lines
                    lines = lines[start_idx:end_idx]
                    line_offset = start_idx
                    
                    output = f"Here's lines {start_idx + 1}-{end_idx} of {path} (total {total_lines} lines):\n"
                else:
                    line_offset = 0
                    output = f"Here's the content of {path} with line numbers:\n"
                
                # Add line numbers
                numbered_lines = []
                for i, line in enumerate(lines, line_offset + 1):
                    numbered_lines.append(f"{i:6d}  {line}")
                
                output += '\n'.join(numbered_lines)
                
                # No truncation - full output preserved for summary
            
            return ToolResult(
                success=True,
                output=output
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )
    
    def _list_directory(self, path: Path, max_depth: int = 2, current_depth: int = 0) -> str:
        """List directory contents recursively"""
        prefix = "  " * current_depth
        output = f"{prefix}{path.name}/\n" if current_depth > 0 else f"Here's the files and directories up to {max_depth} levels deep in {path}:\n{path}\n"
        
        try:
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            
            for item in items:
                if item.name.startswith('.'):
                    continue
                    
                if item.is_dir():
                    if current_depth < max_depth:
                        output += self._list_directory(item, max_depth, current_depth + 1)
                    else:
                        output += f"{prefix}  {item.name}/\n"
                else:
                    output += f"{prefix}  {item.name}\n"
                    
        except PermissionError:
            output += f"{prefix}  [permission denied]\n"
        
        return output
    
    def _write_file(self, path: str, content: str) -> ToolResult:
        """
        Write entire file content (reliable file creation/overwrite)
        
        This is the PREFERRED method for creating or replacing files.
        Unlike edit_file, this always succeeds if the path is valid.
        
        Args:
            path: Path to file
            content: Complete file content to write
            
        Returns:
            ToolResult indicating success
        """
        if not path:
            return ToolResult(
                success=False,
                output="",
                error="No path provided"
            )
        
        if content is None:
            return ToolResult(
                success=False,
                output="",
                error="No content provided"
            )
        
        # Resolve path
        if path.startswith("/"):
            full_path = Path(path)
        else:
            full_path = self.work_dir / path
        
        # Create parent directories
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Check if file exists before writing
            existed = full_path.exists()
            old_size = full_path.stat().st_size if existed else 0
            
            # Write the file
            full_path.write_text(content, encoding='utf-8')
            
            # Verify the write succeeded
            new_size = full_path.stat().st_size
            new_content = full_path.read_text(encoding='utf-8')
            
            if new_content != content:
                return ToolResult(
                    success=False,
                    output="",
                    error="File write verification failed: content mismatch"
                )
            
            # Count lines for feedback
            line_count = content.count('\n') + 1
            action = "overwritten" if existed else "created"
            
            # Show first and last few lines as preview
            lines = content.split('\n')
            if len(lines) <= 10:
                preview = content
            else:
                preview = '\n'.join(lines[:5]) + f"\n... ({len(lines) - 10} lines omitted) ...\n" + '\n'.join(lines[-5:])
            
            return ToolResult(
                success=True,
                output=f"✅ File {action}: {path}\n   Lines: {line_count}\n   Size: {new_size} bytes\n\n--- Preview ---\n{preview}\n--- End Preview ---"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to write file: {str(e)}"
            )
    
    def _edit_file(self, path: str, old_str: str, new_str: str) -> ToolResult:
        """
        Edit a file using string replacement
        
        NOTE: For creating new files or replacing entire file content,
        use write_file instead - it's more reliable.
        
        Args:
            path: Path to file
            old_str: String to find (empty for new file or append)
            new_str: Replacement string
            
        Returns:
            ToolResult indicating success
        """
        if not path:
            return ToolResult(
                success=False,
                output="",
                error="No path provided"
            )
        
        if not new_str:
            return ToolResult(
                success=False,
                output="",
                error="No new_str provided"
            )
        
        # Resolve path
        if path.startswith("/"):
            full_path = Path(path)
        else:
            full_path = self.work_dir / path
        
        # Create parent directories
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if not full_path.exists() or not old_str:
                # For new files, recommend write_file instead
                # But still support it for backwards compatibility
                full_path.write_text(new_str, encoding='utf-8')
                action = "created" if not full_path.exists() else "written"
                return ToolResult(
                    success=True,
                    output=f"File {action}: {path}\n\n💡 TIP: Use write_file tool for creating new files - it's more reliable."
                )
            else:
                # Replace string
                content = full_path.read_text(encoding='utf-8')
                
                if old_str not in content:
                    # Provide helpful error with file preview
                    lines = content.split('\n')
                    preview = '\n'.join(lines[:20]) if len(lines) > 20 else content
                    return ToolResult(
                        success=False,
                        output=f"Current file content (first 20 lines):\n{preview}",
                        error=f"String not found in file. Check exact whitespace and content.\nSearching for: {old_str[:100]}..."
                    )
                
                # Replace (only first occurrence for safety)
                new_content = content.replace(old_str, new_str, 1)
                full_path.write_text(new_content, encoding='utf-8')
                
                # Verify the edit
                verify_content = full_path.read_text(encoding='utf-8')
                if new_str not in verify_content:
                    return ToolResult(
                        success=False,
                        output="",
                        error="Edit verification failed: new content not found in file"
                    )
                
                # Show snippet of change
                snippet_start = max(0, new_content.find(new_str[:50]) - 100)
                snippet_end = min(len(new_content), snippet_start + len(new_str) + 200)
                snippet = new_content[snippet_start:snippet_end]
                
                return ToolResult(
                    success=True,
                    output=f"✅ File edited: {path}\n\nSnippet around change:\n{snippet}"
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )
    
    def _fetch_contract(self, address: str, chain: str = None) -> ToolResult:
        """
        Fetch contract source code from Etherscan API V2
        
        Uses the unified Etherscan V2 API to fetch verified contract source code.
        See: https://docs.etherscan.io/api-reference/endpoint/getsourcecode
        
        Args:
            address: Contract address (0x...)
            chain: Chain name (bsc, mainnet, etc.). Uses self.chain if not provided.
            
        Returns:
            ToolResult with contract source code and ABI
        """
        if not address:
            return ToolResult(
                success=False,
                output="",
                error="No contract address provided"
            )
        
        # Validate address format
        if not address.startswith("0x") or len(address) != 42:
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid address format: {address}. Must be 0x followed by 40 hex characters."
            )
        
        chain = chain or self.chain
        if chain not in ETHERSCAN_CONFIG:
            return ToolResult(
                success=False,
                output="",
                error=f"Unsupported chain: {chain}. Supported: {list(ETHERSCAN_CONFIG.keys())}"
            )
        
        chain_config = ETHERSCAN_CONFIG[chain]
        
        # Build API request
        params = {
            "chainid": chain_config["chain_id"],
            "module": "contract",
            "action": "getsourcecode",
            "address": address
        }
        
        if self.etherscan_api_key:
            params["apikey"] = self.etherscan_api_key
        
        try:
            # Setup proxy if configured
            proxies = None
            if self.proxy:
                proxies = {
                    "http": self.proxy,
                    "https": self.proxy
                }
            
            response = requests.get(
                "https://api.etherscan.io/v2/api",
                params=params,
                timeout=30,
                proxies=proxies
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "1" or not data.get("result"):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Contract not verified or not found on {chain_config['name']}"
                )
            
            result = data["result"][0]
            
            # Check if contract is verified
            if not result.get("SourceCode"):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Contract at {address} is not verified on {chain_config['name']}"
                )
            
            # Parse source code
            source_code = result["SourceCode"]
            contract_name = result.get("ContractName", "Unknown")
            compiler_version = result.get("CompilerVersion", "")
            is_proxy = result.get("Proxy") == "1"
            implementation_address = result.get("Implementation") if is_proxy else None
            
            # Handle JSON format (multi-file contracts)
            if source_code.startswith("{{"):
                source_code = source_code[1:-1]
                try:
                    sources = json.loads(source_code)
                    if "sources" in sources:
                        all_sources = []
                        for filename, content in sources["sources"].items():
                            all_sources.append(f"// ========== File: {filename} ==========\n{content.get('content', '')}")
                        source_code = "\n\n".join(all_sources)
                    else:
                        all_sources = []
                        for filename, content in sources.items():
                            if isinstance(content, dict):
                                all_sources.append(f"// ========== File: {filename} ==========\n{content.get('content', '')}")
                            else:
                                all_sources.append(f"// ========== File: {filename} ==========\n{content}")
                        source_code = "\n\n".join(all_sources)
                except json.JSONDecodeError:
                    pass
            
            # Parse ABI
            abi_str = ""
            if result.get("ABI") and result["ABI"] != "Contract source code not verified":
                abi_str = result["ABI"]
            
            # Save to workspace
            contracts_dir = self.work_dir / "contracts" / address
            contracts_dir.mkdir(parents=True, exist_ok=True)
            
            # Save source code
            source_file = contracts_dir / f"{contract_name}.sol"
            source_file.write_text(source_code, encoding='utf-8')
            
            # Save ABI
            if abi_str:
                abi_file = contracts_dir / "abi.json"
                abi_file.write_text(abi_str, encoding='utf-8')
            
            # Save metadata
            metadata = {
                "address": address,
                "name": contract_name,
                "chain": chain,
                "compiler_version": compiler_version,
                "is_proxy": is_proxy,
                "implementation_address": implementation_address
            }
            metadata_file = contracts_dir / "metadata.json"
            metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            
            # Build output
            output_lines = [
                f"✅ Contract fetched successfully!",
                f"",
                f"📋 Contract Information:",
                f"   Address: {address}",
                f"   Name: {contract_name}",
                f"   Chain: {chain_config['name']}",
                f"   Compiler: {compiler_version}",
            ]
            
            if is_proxy:
                output_lines.append(f"   ⚠️ PROXY CONTRACT - Implementation: {implementation_address}")
                output_lines.append(f"   Use fetch_contract with implementation address to get actual logic!")
            
            output_lines.extend([
                f"",
                f"📁 Saved to: {contracts_dir}",
                f"   - {contract_name}.sol (source code)",
                f"   - abi.json (contract ABI)",
                f"   - metadata.json (contract info)",
                f"",
                f"{'='*60}",
                f"SOURCE CODE PREVIEW (first 200 lines):",
                f"{'='*60}",
            ])
            
            # Add source preview (first 200 lines)
            source_lines = source_code.split('\n')[:200]
            for i, line in enumerate(source_lines, 1):
                output_lines.append(f"{i:4d}| {line}")
            
            if len(source_code.split('\n')) > 200:
                output_lines.append(f"... ({len(source_code.split(chr(10))) - 200} more lines)")
                output_lines.append(f"Use view_file to see full source: {source_file}")
            
            return ToolResult(
                success=True,
                output="\n".join(output_lines)
            )
            
        except requests.RequestException as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Network error fetching contract: {e}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Error fetching contract: {e}"
            )
    
    def run_forge_test(
        self,
        test_path: Optional[str] = None,
        verbose: int = 3,
        fork_url: Optional[str] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Run forge test specifically
        
        Args:
            test_path: Path to test file
            verbose: Verbosity level (0-4)
            fork_url: Fork RPC URL
            
        Returns:
            (success, output, metrics)
        """
        cmd = [self.forge_cmd, "test"]
        
        if test_path:
            cmd.extend(["--match-path", test_path])
        
        cmd.append("-" + "v" * verbose)
        
        if fork_url:
            cmd.extend(["--fork-url", fork_url])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.work_dir / "flaw_verifier"),
                capture_output=True,
                text=True,
                timeout=300
            )
            
            output = result.stdout + "\n" + result.stderr
            success = result.returncode == 0 and "PASS" in output
            
            # Extract metrics
            metrics = {}
            if "Final balance:" in output:
                import re
                match = re.search(r'Final balance:\s*([\d.]+)', output)
                if match:
                    metrics['final_balance'] = float(match.group(1))
            
            return success, output, metrics
            
        except subprocess.TimeoutExpired:
            return False, "Test timed out after 300s", {}
        except Exception as e:
            return False, str(e), {}
    
    def _get_pair(self, token0: str, token1: str, chain: str = None) -> ToolResult:
        """
        Get the pair address for two tokens from DEX factory
        
        Uses cast call to query the factory's getPair(address,address) function.
        
        Args:
            token0: First token address (0x...)
            token1: Second token address (0x...)
            chain: Chain name (bsc, mainnet, etc.). Uses self.chain if not provided.
            
        Returns:
            ToolResult with pair address
        """
        # Validate inputs
        if not token0:
            return ToolResult(
                success=False,
                output="",
                error="No token0 address provided"
            )
        
        if not token1:
            return ToolResult(
                success=False,
                output="",
                error="No token1 address provided"
            )
        
        # Validate address format
        for addr, name in [(token0, "token0"), (token1, "token1")]:
            if not addr.startswith("0x") or len(addr) != 42:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Invalid {name} address format: {addr}. Must be 0x followed by 40 hex characters."
                )
        
        chain = chain or self.chain
        if chain not in DEX_FACTORY_CONFIG:
            return ToolResult(
                success=False,
                output="",
                error=f"Unsupported chain: {chain}. Supported: {list(DEX_FACTORY_CONFIG.keys())}"
            )
        
        factory_config = DEX_FACTORY_CONFIG[chain]
        factory_address = factory_config["v2"]
        dex_name = factory_config["name"]
        
        # Use cast to call getPair(address,address)
        rpc_url = f"http://127.0.0.1:{self.anvil_port}"
        
        try:
            # getPair(address,address) returns address
            cmd = [
                self.cast_cmd, "call",
                factory_address,
                "getPair(address,address)(address)",
                token0, token1,
                "--rpc-url", rpc_url
            ]
            
            # Create clean environment (remove proxy settings)
            env = os.environ.copy()
            proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 
                          'all_proxy', 'ALL_PROXY']
            for var in proxy_vars:
                env.pop(var, None)
            env['no_proxy'] = '*'
            env['NO_PROXY'] = '*'
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                env=env
            )
            
            if result.returncode != 0:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Failed to query {dex_name} factory: {result.stderr}"
                )
            
            pair_address = result.stdout.strip()
            
            # Check if pair exists (address != 0x0)
            if pair_address == "0x0000000000000000000000000000000000000000":
                return ToolResult(
                    success=True,
                    output=f"No pair found for {token0} and {token1} on {dex_name} ({chain})\n\nFactory: {factory_address}\nPair: 0x0 (does not exist)\n\n💡 Tip: The pair may not exist, or try swapping token0/token1 order."
                )
            
            return ToolResult(
                success=True,
                output=f"✅ Pair found on {dex_name} ({chain}):\n\nFactory: {factory_address}\nToken0: {token0}\nToken1: {token1}\nPair Address: {pair_address}\n\n💡 Use this pair address for flash swaps or liquidity analysis."
            )
            
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error="Query timed out after 30s"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to query pair: {str(e)}"
            )
    
    def _find_swap_path(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
        chain: str = None
    ) -> ToolResult:
        """
        Find optimal swap path using uniswap-smart-path library
        
        Args:
            token_in: Input token address (0x...)
            token_out: Output token address (0x...)
            amount_in: Input amount in wei
            chain: Chain name (bsc, mainnet, etc.)
            
        Returns:
            ToolResult with optimal path(s) and expected output
        """
        # Validate inputs
        if not token_in:
            return ToolResult(
                success=False,
                output="",
                error="No token_in address provided"
            )
        
        if not token_out:
            return ToolResult(
                success=False,
                output="",
                error="No token_out address provided"
            )
        
        if not amount_in or amount_in <= 0:
            return ToolResult(
                success=False,
                output="",
                error="Invalid amount_in. Must be a positive integer (in wei)."
            )
        
        # Validate address format
        for addr, name in [(token_in, "token_in"), (token_out, "token_out")]:
            if not addr.startswith("0x") or len(addr) != 42:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Invalid {name} address format: {addr}. Must be 0x followed by 40 hex characters."
                )
        
        chain = chain or self.chain
        rpc_url = f"http://127.0.0.1:{self.anvil_port}"
        
        try:
            from .tools.uniswap_path import DEXPathFinder
            
            finder = DEXPathFinder(rpc_url=rpc_url, chain=chain)
            result = finder.find_best_path(token_in, token_out, amount_in)
            
            if not result.get("success"):
                error_msg = result.get("error", "Unknown error")
                suggested = result.get("suggested_paths", [])
                
                if suggested:
                    # Return suggested paths when no valid path found
                    output_lines = [
                        f"⚠️ No valid path found: {error_msg}",
                        "",
                        f"📋 Chain: {result.get('chain', chain)}",
                        f"📋 DEX: {result.get('dex', 'Unknown')}",
                        "",
                        "Suggested paths to try:",
                    ]
                    
                    for i, path_info in enumerate(suggested, 1):
                        path_str = " → ".join(path_info.get("path", []))
                        note = path_info.get("note", "")
                        output_lines.append(f"  {i}. {note}: {path_str}")
                    
                    return ToolResult(
                        success=True,
                        output="\n".join(output_lines)
                    )
                
                return ToolResult(
                    success=False,
                    output="",
                    error=error_msg
                )
            
            # Format successful result
            best_path = result.get("best_path", {})
            all_paths = result.get("all_paths", [])
            
            output_lines = [
                f"✅ Optimal Swap Path Found!",
                "",
                f"📋 Swap Details:",
                f"   Chain: {result.get('chain', chain)}",
                f"   DEX: {result.get('dex', 'Unknown')}",
                f"   Token In: {token_in}",
                f"   Token Out: {token_out}",
                f"   Amount In: {result.get('amount_in_formatted', amount_in / 10**18):.6f}",
                "",
                f"🏆 Best Path ({best_path.get('type', 'unknown')}):",
                f"   {best_path.get('path_display', 'N/A')}",
                f"   Expected Output: {best_path.get('amount_out_formatted', 0):.6f}",
                "",
            ]
            
            # Show all paths tried
            if len(all_paths) > 1:
                output_lines.append("📊 All Valid Paths (sorted by output):")
                for i, path_info in enumerate(all_paths, 1):
                    output_lines.append(
                        f"   {i}. [{path_info.get('type', 'unknown')}] "
                        f"{path_info.get('path_display', 'N/A')} → "
                        f"{path_info.get('amount_out_formatted', 0):.6f}"
                    )
                output_lines.append("")
            
            # Add DEX addresses for reference
            output_lines.extend([
                "📚 DEX Addresses:",
                f"   V2 Router: {result.get('v2_router', 'N/A')}",
            ])
            
            return ToolResult(
                success=True,
                output="\n".join(output_lines)
            )
            
        except ImportError:
            return ToolResult(
                success=False,
                output="",
                error="uniswap-smart-path library not installed. Run: pip install uniswap-smart-path"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to find swap path: {str(e)}"
            )
    
    def get_history(self, limit: int = 20) -> List[Dict]:
        """Get recent execution history"""
        return self.history[-limit:]

