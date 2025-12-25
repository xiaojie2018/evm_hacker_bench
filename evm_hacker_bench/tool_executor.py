"""
Tool Executor - Execute tools called by LLM

Provides tool execution for:
- bash: Execute shell commands
- view_file: View file/directory contents
- edit_file: Edit files with string replacement

Based on the WebKeyDAO exploit transcript format.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


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
    """
    
    def __init__(
        self,
        work_dir: Path,
        timeout: int = 120,
        anvil_port: int = 8545
    ):
        """
        Initialize tool executor
        
        Args:
            work_dir: Working directory for all operations
            timeout: Default timeout for commands
            anvil_port: Anvil RPC port
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.anvil_port = anvil_port
        
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
                result = self._view_file(arguments.get("path", ""))
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
        
        # Replace placeholders
        command = command.replace("forge", self.forge_cmd)
        command = command.replace("cast", self.cast_cmd)
        
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
            
            # Truncate very long output
            if len(output) > 10000:
                output = output[:5000] + "\n...\n[truncated]\n...\n" + output[-2000:]
            
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
    
    def _view_file(self, path: str) -> ToolResult:
        """
        View file or directory contents
        
        Args:
            path: Path to view
            
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
                
                # Add line numbers
                numbered_lines = []
                for i, line in enumerate(lines, 1):
                    numbered_lines.append(f"{i:6d}  {line}")
                
                output = f"Here's the content of {path} with line numbers:\n"
                output += '\n'.join(numbered_lines)
                
                # Truncate if too long
                if len(output) > 15000:
                    output = output[:7500] + "\n...\n[truncated]\n...\n" + output[-3000:]
            
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
    
    def get_history(self, limit: int = 20) -> List[Dict]:
        """Get recent execution history"""
        return self.history[-limit:]

