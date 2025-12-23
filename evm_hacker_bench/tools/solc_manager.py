"""
Solc Manager - Solidity Compiler Version Management

Based on SCONE-bench requirements:
- Install and switch Solidity compiler versions
- Support for contracts with different pragma requirements
- Integration with solc-select
"""

import subprocess
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple


class SolcManager:
    """
    Solidity Compiler Version Manager
    
    Features:
    - Install solc versions via solc-select
    - Switch between versions
    - Auto-detect version from pragma
    - Compile contracts with specific versions
    """
    
    def __init__(self):
        """Initialize solc manager"""
        self._solc_select_available = self._check_solc_select()
    
    def _check_solc_select(self) -> bool:
        """Check if solc-select is installed"""
        try:
            # Use 'solc-select versions' instead of '--version' as solc-select doesn't support --version flag
            result = subprocess.run(
                ["solc-select", "versions"],
                capture_output=True,
                text=True,
                timeout=10
            )
            # solc-select versions returns 0 even when no versions are installed
            return result.returncode == 0
        except:
            return False
    
    def get_installed_versions(self) -> List[str]:
        """Get list of installed solc versions"""
        if not self._solc_select_available:
            return []
        
        try:
            result = subprocess.run(
                ["solc-select", "versions"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            versions = []
            for line in result.stdout.split('\n'):
                line = line.strip()
                if line and re.match(r'^\d+\.\d+\.\d+', line):
                    # Remove any markers like (current)
                    version = line.split()[0]
                    versions.append(version)
            
            return versions
            
        except:
            return []
    
    def get_current_version(self) -> Optional[str]:
        """Get currently selected solc version"""
        try:
            result = subprocess.run(
                ["solc", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            match = re.search(r'Version:\s*(\d+\.\d+\.\d+)', result.stdout)
            if match:
                return match.group(1)
            
            return None
        except:
            return None
    
    def install_version(self, version: str) -> Dict[str, Any]:
        """
        Install a solc version
        
        Args:
            version: Version to install (e.g., "0.8.19")
            
        Returns:
            Installation result
        """
        if not self._solc_select_available:
            return {
                "success": False,
                "error": "solc-select not installed. Run: pip install solc-select"
            }
        
        try:
            result = subprocess.run(
                ["solc-select", "install", version],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "version": version,
                    "message": f"Installed solc {version}"
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr or "Installation failed"
                }
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Installation timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def use_version(self, version: str) -> Dict[str, Any]:
        """
        Switch to a solc version
        
        Args:
            version: Version to use
            
        Returns:
            Switch result
        """
        if not self._solc_select_available:
            return {
                "success": False,
                "error": "solc-select not installed"
            }
        
        # Install if not available
        installed = self.get_installed_versions()
        if version not in installed:
            install_result = self.install_version(version)
            if not install_result["success"]:
                return install_result
        
        try:
            result = subprocess.run(
                ["solc-select", "use", version],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "version": version,
                    "message": f"Switched to solc {version}"
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr or "Switch failed"
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def detect_version_from_pragma(self, code: str) -> Optional[str]:
        """
        Detect required solc version from pragma
        
        Args:
            code: Solidity source code
            
        Returns:
            Detected version or None
        """
        # Match pragma solidity patterns
        patterns = [
            r'pragma\s+solidity\s*\^?(\d+\.\d+\.\d+)',
            r'pragma\s+solidity\s*>=?\s*(\d+\.\d+\.\d+)',
            r'pragma\s+solidity\s*(\d+\.\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, code)
            if match:
                version = match.group(1)
                # Ensure it has patch version
                if version.count('.') == 1:
                    version += '.0'
                return version
        
        return None
    
    def detect_version_from_file(self, file_path: str) -> Optional[str]:
        """
        Detect solc version from Solidity file
        
        Args:
            file_path: Path to Solidity file
            
        Returns:
            Detected version
        """
        try:
            content = Path(file_path).read_text()
            return self.detect_version_from_pragma(content)
        except:
            return None
    
    def ensure_version(self, version: str) -> Dict[str, Any]:
        """
        Ensure a version is installed and selected
        
        Args:
            version: Required version
            
        Returns:
            Result
        """
        current = self.get_current_version()
        
        if current == version:
            return {
                "success": True,
                "version": version,
                "message": f"Already using solc {version}"
            }
        
        return self.use_version(version)
    
    def compile_file(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        optimize: bool = True,
        optimize_runs: int = 200
    ) -> Dict[str, Any]:
        """
        Compile a Solidity file
        
        Args:
            file_path: Path to Solidity file
            output_dir: Output directory
            optimize: Enable optimization
            optimize_runs: Optimization runs
            
        Returns:
            Compilation result
        """
        # Auto-detect and set version
        version = self.detect_version_from_file(file_path)
        if version:
            version_result = self.ensure_version(version)
            if not version_result["success"]:
                return version_result
        
        # Build command
        cmd = ["solc"]
        
        if optimize:
            cmd.extend(["--optimize", "--optimize-runs", str(optimize_runs)])
        
        if output_dir:
            cmd.extend(["--output-dir", output_dir])
            cmd.append("--overwrite")
        
        cmd.extend(["--abi", "--bin", file_path])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "output": result.stdout,
                    "version": self.get_current_version()
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_recommended_version(self, code: str) -> str:
        """
        Get recommended solc version for code
        
        Args:
            code: Solidity source code
            
        Returns:
            Recommended version
        """
        detected = self.detect_version_from_pragma(code)
        
        if detected:
            major_minor = '.'.join(detected.split('.')[:2])
            
            # Map to latest stable patch version
            version_map = {
                "0.8": "0.8.19",
                "0.7": "0.7.6",
                "0.6": "0.6.12",
                "0.5": "0.5.17",
                "0.4": "0.4.26"
            }
            
            return version_map.get(major_minor, detected)
        
        # Default to 0.8.19
        return "0.8.19"

