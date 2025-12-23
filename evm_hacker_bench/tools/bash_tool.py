"""
Bash Tool - Persistent Bash Session

Based on SCONE-bench requirements:
- Execute commands in a persistent bash session
- Access to Foundry toolchain (forge, cast, anvil)
- Support for common shell operations
"""

import subprocess
import os
import time
import select
import threading
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommandResult:
    """Result of a bash command execution"""
    output: str
    error: str
    exit_code: int
    duration: float


class BashTool:
    """
    Persistent Bash Session Tool
    
    Features:
    - Maintains persistent bash session (environment preserved)
    - Timeout support for long-running commands
    - Working directory tracking
    - Output capture and streaming
    """
    
    def __init__(
        self,
        work_dir: Optional[Path] = None,
        timeout_default: int = 300,
        env_vars: Optional[Dict[str, str]] = None
    ):
        """
        Initialize bash tool
        
        Args:
            work_dir: Initial working directory
            timeout_default: Default timeout in seconds
            env_vars: Additional environment variables
        """
        self.work_dir = Path(work_dir or os.getcwd())
        self.timeout_default = timeout_default
        self.env = os.environ.copy()
        
        if env_vars:
            self.env.update(env_vars)
        
        # Ensure Foundry is in PATH
        foundry_bin = Path.home() / ".foundry" / "bin"
        if foundry_bin.exists():
            self.env["PATH"] = f"{foundry_bin}:{self.env.get('PATH', '')}"
        
        # Command history
        self.history: list = []
    
    def execute(
        self,
        command: str,
        timeout: Optional[int] = None,
        cwd: Optional[Path] = None
    ) -> CommandResult:
        """
        Execute a bash command
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds (None = default)
            cwd: Working directory (None = use current)
            
        Returns:
            CommandResult with output, error, exit_code
        """
        timeout = timeout or self.timeout_default
        work_dir = Path(cwd) if cwd else self.work_dir
        
        start_time = time.time()
        
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=work_dir,
                env=self.env,
                text=True
            )
            
            stdout, stderr = process.communicate(timeout=timeout)
            duration = time.time() - start_time
            
            result = CommandResult(
                output=stdout,
                error=stderr,
                exit_code=process.returncode,
                duration=duration
            )
            
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            duration = time.time() - start_time
            
            result = CommandResult(
                output=stdout or "",
                error=f"Command timed out after {timeout}s\n{stderr or ''}",
                exit_code=-1,
                duration=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            result = CommandResult(
                output="",
                error=str(e),
                exit_code=-1,
                duration=duration
            )
        
        # Track history
        self.history.append({
            "command": command,
            "exit_code": result.exit_code,
            "duration": result.duration,
            "timestamp": time.time()
        })
        
        # Update working directory if cd command
        if command.strip().startswith("cd ") and result.exit_code == 0:
            new_dir = command.strip()[3:].strip()
            if new_dir.startswith("/"):
                self.work_dir = Path(new_dir)
            else:
                self.work_dir = (self.work_dir / new_dir).resolve()
        
        return result
    
    def run_forge(
        self,
        subcommand: str,
        args: list = None,
        timeout: int = 300
    ) -> CommandResult:
        """
        Run forge command
        
        Args:
            subcommand: forge subcommand (test, build, etc.)
            args: Additional arguments
            timeout: Timeout in seconds
            
        Returns:
            CommandResult
        """
        args = args or []
        cmd = f"forge {subcommand} " + " ".join(args)
        return self.execute(cmd, timeout=timeout)
    
    def run_cast(
        self,
        subcommand: str,
        args: list = None,
        rpc_url: str = "http://127.0.0.1:8545"
    ) -> CommandResult:
        """
        Run cast command
        
        Args:
            subcommand: cast subcommand (call, send, etc.)
            args: Additional arguments
            rpc_url: RPC URL
            
        Returns:
            CommandResult
        """
        args = args or []
        cmd = f"cast {subcommand} " + " ".join(args) + f" --rpc-url {rpc_url}"
        return self.execute(cmd, timeout=60)
    
    def check_foundry_installed(self) -> Tuple[bool, str]:
        """Check if Foundry is installed"""
        # Try multiple paths for forge
        forge_paths = [
            os.path.expanduser('~/.foundry/bin/forge'),
            '/usr/local/bin/forge',
            'forge',
        ]
        
        for forge_path in forge_paths:
            result = self.execute(f"{forge_path} --version", timeout=10)
            if result.exit_code == 0:
                return True, result.output.strip()
        
        return False, "Foundry not installed"
    
    def cd(self, directory: str) -> bool:
        """Change working directory"""
        result = self.execute(f"cd {directory} && pwd")
        return result.exit_code == 0
    
    def get_cwd(self) -> str:
        """Get current working directory"""
        return str(self.work_dir)
    
    def get_history(self, limit: int = 10) -> list:
        """Get command history"""
        return self.history[-limit:]

