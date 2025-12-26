"""
EVM Environment - Anvil Fork Management

Responsibilities:
1. Start/stop Anvil fork at specific block numbers
2. Manage test accounts and balances
3. Provide Web3 connection for state queries
4. Support multiple chains (ETH, BSC, Arbitrum, Base)
"""

import os
import json
import socket
import subprocess
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from web3 import Web3
from eth_account import Account


@dataclass
class ChainConfig:
    """Configuration for a specific blockchain"""
    name: str
    chain_id: int
    rpc_url: str
    native_symbol: str
    block_explorer: str


# Supported chains configuration
CHAIN_CONFIGS = {
    "mainnet": ChainConfig(
        name="Ethereum Mainnet",
        chain_id=1,
        rpc_url="https://eth.llamarpc.com",
        native_symbol="ETH",
        block_explorer="https://etherscan.io"
    ),
    "bsc": ChainConfig(
        name="BNB Smart Chain",
        chain_id=56,
        rpc_url="https://bsc-dataseed.binance.org",
        native_symbol="BNB",
        block_explorer="https://bscscan.com"
    ),
    "arbitrum": ChainConfig(
        name="Arbitrum One",
        chain_id=42161,
        rpc_url="https://arb1.arbitrum.io/rpc",
        native_symbol="ETH",
        block_explorer="https://arbiscan.io"
    ),
    "base": ChainConfig(
        name="Base",
        chain_id=8453,
        rpc_url="https://mainnet.base.org",
        native_symbol="ETH",
        block_explorer="https://basescan.org"
    ),
    "polygon": ChainConfig(
        name="Polygon",
        chain_id=137,
        rpc_url="https://polygon-rpc.com",
        native_symbol="MATIC",
        block_explorer="https://polygonscan.com"
    ),
    "optimism": ChainConfig(
        name="Optimism",
        chain_id=10,
        rpc_url="https://mainnet.optimism.io",
        native_symbol="ETH",
        block_explorer="https://optimistic.etherscan.io"
    )
}


class EVMEnvironment:
    """EVM Environment Management for Exploit Testing"""
    
    def __init__(
        self,
        chain: str = "mainnet",
        fork_block: Optional[int] = None,
        rpc_url: Optional[str] = None,
        anvil_port: int = 8545,
        initial_balance: int = 1000000,  # Initial ETH/BNB balance (1M tokens)
        evm_version: Optional[str] = None
    ):
        """
        Initialize EVM environment
        
        Args:
            chain: Chain identifier (mainnet, bsc, arbitrum, base)
            fork_block: Block number to fork from (None = latest)
            rpc_url: Custom RPC URL (overrides chain default)
            anvil_port: Local Anvil port
            initial_balance: Initial native token balance for test account
            evm_version: EVM version (shanghai, cancun, etc.)
        """
        if chain not in CHAIN_CONFIGS:
            raise ValueError(f"Unsupported chain: {chain}. Supported: {list(CHAIN_CONFIGS.keys())}")
        
        self.chain_config = CHAIN_CONFIGS[chain]
        self.chain = chain
        self.fork_block = fork_block
        self.rpc_url = rpc_url or os.getenv(
            f"{chain.upper()}_RPC", 
            self.chain_config.rpc_url
        )
        self.anvil_port = anvil_port
        self.initial_balance = initial_balance
        self.evm_version = evm_version
        
        # Process state
        self.anvil_process: Optional[subprocess.Popen] = None
        self.w3: Optional[Web3] = None
        
        # Test account
        self.test_account: Optional[Account] = None
        self.test_address: Optional[str] = None
        self.test_private_key: Optional[str] = None
        
        # Attacker contract address (if deployed)
        self.attacker_address: Optional[str] = None
        
        # Snapshot management
        self.initial_snapshot_id: Optional[str] = None
        
    def start(self) -> Dict[str, Any]:
        """
        Start the EVM environment
        
        Returns:
            Environment info dictionary
        """
        print(f"🚀 Starting EVM Environment")
        print(f"   Chain: {self.chain_config.name}")
        print(f"   RPC: {self.rpc_url}")
        if self.fork_block:
            print(f"   Fork Block: {self.fork_block}")
        
        # 1. Start Anvil fork
        self._start_anvil()
        
        # 2. Connect Web3
        self._connect_web3()
        
        # 3. Create and fund test account
        self._setup_test_account()
        
        # 4. Create initial snapshot
        self._create_initial_snapshot()
        
        return self.get_env_info()
    
    def _find_anvil_command(self) -> str:
        """Find anvil command by searching multiple paths"""
        # Search paths in order of preference
        anvil_paths = [
            os.path.expanduser('~/.foundry/bin/anvil'),
            '/usr/local/bin/anvil',
            'anvil',
        ]
        
        for path in anvil_paths:
            try:
                result = subprocess.run(
                    [path, '--version'],
                    capture_output=True,
                    check=True,
                    text=True,
                    timeout=5
                )
                print(f"   Found Anvil: {path}")
                return path
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        raise RuntimeError(
            "Anvil not found. Please install Foundry:\n"
            "  curl -L https://foundry.paradigm.xyz | bash\n"
            "  foundryup\n"
            "Then ensure ~/.foundry/bin is in your PATH"
        )
    
    def _start_anvil(self, max_retries: int = 3):
        """Start Anvil fork process with retry mechanism"""
        # Kill any existing process on the port first
        self._kill_port_process(self.anvil_port)
        
        # Find anvil command
        anvil_cmd = self._find_anvil_command()
        
        # Build Anvil command - keep it simple for reliability
        cmd = [
            anvil_cmd,
            "-f", self.rpc_url,  # Short form is more reliable
            "--port", str(self.anvil_port),
            "--chain-id", str(self.chain_config.chain_id),
            "--retries", "5",  # Retry RPC calls within Anvil
            "--timeout", "120000",  # 120 second timeout for RPC calls
        ]
        
        # IMPORTANT: fork-block-number must be specified for historical state
        if self.fork_block:
            cmd.extend(["--fork-block-number", str(self.fork_block)])
            print(f"   Fork Block: {self.fork_block}")
        
        if self.evm_version:
            cmd.extend(["--hardfork", self.evm_version])
        
        print(f"   Command: {' '.join(cmd)}")
        
        # Create environment without proxy settings (important for WSL)
        anvil_env = os.environ.copy()
        proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 
                      'all_proxy', 'ALL_PROXY', 'ftp_proxy', 'FTP_PROXY']
        for var in proxy_vars:
            anvil_env.pop(var, None)
        anvil_env['no_proxy'] = '*'
        anvil_env['NO_PROXY'] = '*'
        
        last_error = None
        for retry in range(max_retries):
            if retry > 0:
                wait_time = 5 * retry  # Exponential backoff: 5s, 10s
                print(f"   ⚠️ Retry {retry}/{max_retries-1}, waiting {wait_time}s...")
                time.sleep(wait_time)
            
            try:
                self.anvil_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=anvil_env,
                    preexec_fn=os.setsid if os.name != 'nt' else None
                )
                
                # Wait for Anvil to start
                if self._wait_for_anvil(timeout=60):
                    print(f"✓ Anvil started on port {self.anvil_port}")
                    return
                
                # Check if process crashed
                if self.anvil_process.poll() is not None:
                    stderr = self.anvil_process.stderr.read().decode() if self.anvil_process.stderr else ""
                    if "client error" in stderr.lower() or "rate" in stderr.lower():
                        last_error = f"RPC rate limiting: {stderr[:100]}"
                        continue  # Retry on rate limit
                    last_error = f"Anvil crashed: {stderr[:200]}"
                else:
                    last_error = "Anvil timeout"
                    self._stop_anvil_process()
                    
            except FileNotFoundError:
                raise RuntimeError(
                    "Anvil not found. Please install Foundry: "
                    "curl -L https://foundry.paradigm.xyz | bash && foundryup"
                )
            except Exception as e:
                last_error = str(e)
        
        raise RuntimeError(f"Anvil failed to start after {max_retries} attempts: {last_error}")
    
    def _stop_anvil_process(self):
        """Stop the Anvil process if running"""
        if self.anvil_process:
            try:
                self.anvil_process.terminate()
                self.anvil_process.wait(timeout=5)
            except:
                try:
                    self.anvil_process.kill()
                except:
                    pass
            self.anvil_process = None
    
    def _kill_port_process(self, port: int):
        """Kill only Anvil processes using the specified port (safe version)"""
        try:
            # Only kill anvil processes, not other processes
            # Use pkill to target anvil specifically
            subprocess.run(
                ["pkill", "-f", f"anvil.*--port.*{port}"],
                capture_output=True,
                timeout=5
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        try:
            # Also try killing by process name 'anvil' that's listening on the port
            result = subprocess.run(
                ["lsof", "-t", f"-i:{port}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        pid_int = int(pid)
                        # Check if this is an anvil process before killing
                        proc_check = subprocess.run(
                            ["ps", "-p", str(pid_int), "-o", "comm="],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        if "anvil" in proc_check.stdout.lower():
                            os.kill(pid_int, 15)  # SIGTERM first, not SIGKILL
                            time.sleep(0.3)
                            # If still running, force kill
                            try:
                                os.kill(pid_int, 0)  # Check if still alive
                                os.kill(pid_int, 9)  # SIGKILL
                            except ProcessLookupError:
                                pass  # Already dead
                    except (ValueError, ProcessLookupError, subprocess.TimeoutExpired):
                        pass
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Give OS time to release the port
        time.sleep(0.3)
    
    def _wait_for_anvil(self, timeout: int = 120) -> bool:
        """Wait for Anvil to be fully ready (not just port open)"""
        import requests
        
        start_time = time.time()
        anvil_rpc = f"http://127.0.0.1:{self.anvil_port}"
        
        # First wait for port to open
        port_ready = False
        while time.time() - start_time < timeout and not port_ready:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', self.anvil_port))
                sock.close()
                if result == 0:
                    port_ready = True
            except Exception:
                pass
            if not port_ready:
                time.sleep(0.5)
        
        if not port_ready:
            return False
        
        # Now wait for Anvil to respond to RPC calls with correct fork block
        session = requests.Session()
        session.proxies = {'http': None, 'https': None}
        session.trust_env = False
        
        while time.time() - start_time < timeout:
            try:
                # Check if Anvil responds to eth_blockNumber
                response = session.post(
                    anvil_rpc,
                    json={"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1},
                    timeout=5
                )
                if response.status_code == 200:
                    result = response.json()
                    if 'result' in result:
                        block_num = int(result['result'], 16)
                        # If we specified a fork block, verify it matches
                        if self.fork_block:
                            # Allow some tolerance (Anvil might be 1-2 blocks ahead)
                            if abs(block_num - self.fork_block) <= 2:
                                return True
                            # If block number is way off, wait more (Anvil still initializing)
                            elif block_num > self.fork_block + 1000000:
                                time.sleep(1)
                                continue
                        else:
                            return True
            except Exception:
                pass
            time.sleep(1)
        
        return False
    
    def _connect_web3(self):
        """Connect to Anvil via Web3"""
        import requests
        
        anvil_rpc = f"http://127.0.0.1:{self.anvil_port}"
        
        # Create session without proxy
        session = requests.Session()
        session.proxies = {'http': None, 'https': None}
        session.trust_env = False
        
        from web3.providers.rpc import HTTPProvider
        provider = HTTPProvider(
            anvil_rpc,
            session=session,
            request_kwargs={'timeout': 60}
        )
        self.w3 = Web3(provider)
        
        # Inject POA middleware if needed
        try:
            from web3.middleware import ExtraDataToPOAMiddleware
            self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        except ImportError:
            try:
                from web3.middleware import geth_poa_middleware
                self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            except ImportError:
                pass
        
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to Anvil at {anvil_rpc}")
        
        actual_block = self.w3.eth.block_number
        print(f"✓ Web3 connected")
        print(f"   Block Number: {actual_block}")
        print(f"   Chain ID: {self.w3.eth.chain_id}")
        
        # Verify fork block
        if self.fork_block:
            # Simple verification: check if current block matches fork block (within tolerance)
            if abs(actual_block - self.fork_block) <= 2:
                print(f"   ✓ Fork Block Verified: {self.fork_block}")
            else:
                # Try to get more info from anvil_nodeInfo
                try:
                    node_info = self.w3.provider.make_request("anvil_nodeInfo", [])
                    if node_info and 'result' in node_info:
                        fork_config = node_info['result'].get('forkConfig', {})
                        actual_fork_block = fork_config.get('forkBlockNumber')
                        if actual_fork_block and abs(actual_fork_block - self.fork_block) <= 2:
                            print(f"   ✓ Fork Block Verified: {self.fork_block}")
                        else:
                            print(f"   ⚠️  WARNING: Fork block may not match!")
                            print(f"      Requested: {self.fork_block}")
                            print(f"      Current block: {actual_block}")
                            if actual_fork_block:
                                print(f"      Fork config block: {actual_fork_block}")
                    else:
                        print(f"   Fork Block: {self.fork_block} (configured, verification skipped)")
                except Exception:
                    print(f"   Fork Block: {self.fork_block} (configured, verification failed)")
    
    def _setup_test_account(self):
        """Create and fund test account"""
        self.test_account = Account.create()
        self.test_address = self.test_account.address
        self.test_private_key = self.test_account.key.hex()
        
        # Fund the account
        balance_wei = self.initial_balance * 10**18
        self._set_balance(self.test_address, balance_wei)
        
        actual_balance = self.w3.eth.get_balance(self.test_address) / 10**18
        print(f"✓ Test account created")
        print(f"   Address: {self.test_address}")
        print(f"   Balance: {actual_balance} {self.chain_config.native_symbol}")
    
    def _set_balance(self, address: str, balance_wei: int):
        """Set account balance using Anvil RPC"""
        self.w3.provider.make_request(
            'anvil_setBalance',
            [address, hex(balance_wei)]
        )
    
    def _create_initial_snapshot(self):
        """Create snapshot of initial state"""
        try:
            result = self.w3.provider.make_request("evm_snapshot", [])
            self.initial_snapshot_id = result.get('result')
            print(f"✓ Initial snapshot created: {self.initial_snapshot_id}")
        except Exception as e:
            print(f"⚠️  Snapshot creation failed: {e}")
    
    def get_env_info(self) -> Dict[str, Any]:
        """Get environment information"""
        return {
            'chain': self.chain,
            'chain_name': self.chain_config.name,
            'chain_id': self.chain_config.chain_id,
            'rpc_url': f"http://127.0.0.1:{self.anvil_port}",
            'fork_rpc': self.rpc_url,
            'fork_block': self.fork_block or self.w3.eth.block_number,
            'test_address': self.test_address,
            'test_private_key': self.test_private_key,
            'native_symbol': self.chain_config.native_symbol,
            'evm_version': self.evm_version
        }
    
    def get_balance(self, address: Optional[str] = None) -> float:
        """Get native token balance in ETH/BNB"""
        addr = address or self.test_address
        balance_wei = self.w3.eth.get_balance(addr)
        return balance_wei / 10**18
    
    def reset(self) -> bool:
        """Reset environment to initial state using snapshot"""
        if not self.initial_snapshot_id:
            print("⚠️  No snapshot available for reset")
            return False
        
        try:
            # Revert to snapshot
            result = self.w3.provider.make_request(
                "evm_revert", 
                [self.initial_snapshot_id]
            )
            
            if not result.get('result', False):
                return False
            
            # Recreate snapshot (some Anvil versions consume it)
            new_snapshot = self.w3.provider.make_request("evm_snapshot", [])
            self.initial_snapshot_id = new_snapshot.get('result')
            
            return True
        except Exception as e:
            print(f"⚠️  Reset failed: {e}")
            return False
    
    def mine_block(self, timestamp: Optional[int] = None):
        """Mine a new block (for tests that need block progression)"""
        if timestamp:
            self.w3.provider.make_request("evm_setNextBlockTimestamp", [timestamp])
        self.w3.provider.make_request("evm_mine", [])
    
    def impersonate_account(self, address: str) -> str:
        """Impersonate an account (for testing)"""
        self.w3.provider.make_request("anvil_impersonateAccount", [address])
        return address
    
    def stop_impersonating(self, address: str):
        """Stop impersonating an account"""
        self.w3.provider.make_request("anvil_stopImpersonatingAccount", [address])
    
    def get_contract_code(self, address: str) -> str:
        """Get contract bytecode at address"""
        return self.w3.eth.get_code(address).hex()
    
    def call_contract(
        self, 
        to: str, 
        data: str, 
        value: int = 0,
        from_addr: Optional[str] = None
    ) -> str:
        """Make a contract call"""
        call_params = {
            'to': to,
            'data': data,
            'value': value
        }
        if from_addr:
            call_params['from'] = from_addr
        
        return self.w3.eth.call(call_params).hex()
    
    def stop(self):
        """Stop the environment"""
        if self.anvil_process:
            try:
                import signal
                if os.name != 'nt':
                    os.killpg(os.getpgid(self.anvil_process.pid), signal.SIGTERM)
                else:
                    self.anvil_process.terminate()
                self.anvil_process.wait(timeout=5)
            except Exception as e:
                print(f"⚠️  Error stopping Anvil: {e}")
                try:
                    self.anvil_process.kill()
                except Exception:
                    pass
            finally:
                self.anvil_process = None
        
        self.w3 = None
        print("✓ Environment stopped")
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
        return False


class FoundryRunner:
    """Execute Foundry tests for exploit validation"""
    
    def __init__(self, env: EVMEnvironment, work_dir: Optional[Path] = None):
        """
        Initialize Foundry runner
        
        Args:
            env: EVM environment instance
            work_dir: Working directory for Foundry project
        """
        self.env = env
        # Default work_dir: data/exploit_workspace
        if work_dir:
            self.work_dir = work_dir
        else:
            project_root = Path(__file__).parent.parent
            self.work_dir = project_root / "data" / "exploit_workspace"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Find forge command
        self.forge_cmd = self._find_forge_command()
        
        # Initialize Foundry project if needed
        self._init_foundry_project()
    
    def _find_forge_command(self) -> str:
        """Find forge command by searching multiple paths"""
        forge_paths = [
            os.path.expanduser('~/.foundry/bin/forge'),
            '/usr/local/bin/forge',
            'forge',
        ]
        
        for path in forge_paths:
            try:
                result = subprocess.run(
                    [path, '--version'],
                    capture_output=True,
                    check=True,
                    text=True,
                    timeout=5
                )
                return path
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        raise RuntimeError(
            "Forge not found. Please install Foundry:\n"
            "  curl -L https://foundry.paradigm.xyz | bash\n"
            "  foundryup"
        )
    
    def _init_foundry_project(self):
        """Initialize Foundry project structure"""
        # Create foundry.toml
        foundry_toml = self.work_dir / "foundry.toml"
        if not foundry_toml.exists():
            config = f"""[profile.default]
src = "src"
out = "out"
libs = ["lib"]
evm_version = "{self.env.evm_version or 'shanghai'}"
gas_limit = 30000000
ffi = true

[rpc_endpoints]
local = "http://127.0.0.1:{self.env.anvil_port}"
"""
            foundry_toml.write_text(config)
        
        # Create directories
        (self.work_dir / "src").mkdir(exist_ok=True)
        (self.work_dir / "test").mkdir(exist_ok=True)
        
        # Install forge-std if not present
        lib_dir = self.work_dir / "lib" / "forge-std"
        if not lib_dir.exists():
            subprocess.run(
                [self.forge_cmd, "install", "foundry-rs/forge-std", "--no-commit"],
                cwd=self.work_dir,
                capture_output=True
            )
    
    def write_exploit(self, code: str, filename: str = "Exploit.t.sol") -> Path:
        """Write exploit code to file"""
        test_file = self.work_dir / "test" / filename
        test_file.write_text(code)
        return test_file
    
    def run_exploit(
        self, 
        test_file: Optional[Path] = None,
        test_function: str = "testExploit",
        timeout: int = 300,
        verbose: int = 2
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Run exploit test
        
        Args:
            test_file: Path to test file
            test_function: Test function name
            timeout: Timeout in seconds
            verbose: Verbosity level (0-4)
            
        Returns:
            (success, output, metrics)
        """
        test_path = test_file or (self.work_dir / "test" / "Exploit.t.sol")
        
        cmd = [
            self.forge_cmd, "test",
            "--match-path", str(test_path),
            "--match-test", test_function,
            "--fork-url", f"http://127.0.0.1:{self.env.anvil_port}",
            "-" + "v" * verbose,
            "--json"
        ]
        
        if self.env.fork_block:
            cmd.extend(["--fork-block-number", str(self.env.fork_block)])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            success = result.returncode == 0
            output = result.stdout + result.stderr
            
            # Parse metrics from output
            metrics = self._parse_test_output(output)
            
            return success, output, metrics
            
        except subprocess.TimeoutExpired:
            return False, "Test timed out", {"error": "timeout"}
        except Exception as e:
            return False, str(e), {"error": str(e)}
    
    def _parse_test_output(self, output: str) -> Dict[str, Any]:
        """Parse Foundry test output for metrics"""
        metrics = {
            "gas_used": None,
            "profit": None,
            "duration_ms": None
        }
        
        # Try to parse JSON output
        try:
            for line in output.split('\n'):
                if line.strip().startswith('{'):
                    data = json.loads(line)
                    if 'gas' in data:
                        metrics['gas_used'] = data['gas']
                    break
        except json.JSONDecodeError:
            pass
        
        return metrics
    
    def cleanup(self):
        """Clean up workspace"""
        import shutil
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)

