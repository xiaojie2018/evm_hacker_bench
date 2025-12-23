"""
File Editor Tool - File CRUD Operations

Based on SCONE-bench requirements:
- Create, read, update, delete files
- Support for Solidity contract files
- Safe file operations with backup
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List


class FileEditor:
    """
    File Editor Tool for CRUD Operations
    
    Features:
    - Read/write files with encoding support
    - Safe writes with backup
    - Directory operations
    - File search and listing
    """
    
    def __init__(
        self,
        work_dir: Optional[Path] = None,
        allowed_extensions: Optional[List[str]] = None,
        create_backups: bool = True
    ):
        """
        Initialize file editor
        
        Args:
            work_dir: Working directory
            allowed_extensions: List of allowed file extensions
            create_backups: Whether to create backups on write
        """
        self.work_dir = Path(work_dir or os.getcwd())
        self.allowed_extensions = allowed_extensions or [
            '.sol', '.t.sol', '.s.sol',  # Solidity
            '.json', '.toml', '.txt',     # Config
            '.py', '.sh'                   # Scripts
        ]
        self.create_backups = create_backups
        self.backup_dir = self.work_dir / ".file_backups"
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to work_dir"""
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.work_dir / p).resolve()
    
    def _validate_extension(self, path: Path) -> bool:
        """Check if file extension is allowed"""
        if not self.allowed_extensions:
            return True
        return path.suffix in self.allowed_extensions or \
               ''.join(path.suffixes[-2:]) in self.allowed_extensions
    
    def read(self, path: str, encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        Read file contents
        
        Args:
            path: File path
            encoding: File encoding
            
        Returns:
            Dict with success, content, error
        """
        file_path = self._resolve_path(path)
        
        try:
            if not file_path.exists():
                return {
                    "success": False,
                    "content": None,
                    "error": f"File not found: {file_path}"
                }
            
            content = file_path.read_text(encoding=encoding)
            
            return {
                "success": True,
                "content": content,
                "path": str(file_path),
                "size": len(content),
                "lines": content.count('\n') + 1
            }
            
        except Exception as e:
            return {
                "success": False,
                "content": None,
                "error": str(e)
            }
    
    def write(
        self,
        path: str,
        content: str,
        encoding: str = 'utf-8',
        create_dirs: bool = True
    ) -> Dict[str, Any]:
        """
        Write content to file
        
        Args:
            path: File path
            content: Content to write
            encoding: File encoding
            create_dirs: Create parent directories if needed
            
        Returns:
            Dict with success, path, error
        """
        file_path = self._resolve_path(path)
        
        # Validate extension
        if not self._validate_extension(file_path):
            return {
                "success": False,
                "error": f"Extension not allowed: {file_path.suffix}"
            }
        
        try:
            # Create backup if file exists
            if self.create_backups and file_path.exists():
                self._create_backup(file_path)
            
            # Create directories
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content
            file_path.write_text(content, encoding=encoding)
            
            return {
                "success": True,
                "path": str(file_path),
                "size": len(content)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_backup(self, file_path: Path):
        """Create backup of existing file"""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
    
    def delete(self, path: str) -> Dict[str, Any]:
        """
        Delete file
        
        Args:
            path: File path
            
        Returns:
            Dict with success, error
        """
        file_path = self._resolve_path(path)
        
        try:
            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {file_path}"
                }
            
            # Create backup before delete
            if self.create_backups:
                self._create_backup(file_path)
            
            file_path.unlink()
            
            return {
                "success": True,
                "path": str(file_path)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_dir(
        self,
        path: str = ".",
        pattern: str = "*",
        recursive: bool = False
    ) -> Dict[str, Any]:
        """
        List directory contents
        
        Args:
            path: Directory path
            pattern: Glob pattern
            recursive: Search recursively
            
        Returns:
            Dict with files list
        """
        dir_path = self._resolve_path(path)
        
        try:
            if not dir_path.exists():
                return {
                    "success": False,
                    "error": f"Directory not found: {dir_path}"
                }
            
            if recursive:
                files = list(dir_path.rglob(pattern))
            else:
                files = list(dir_path.glob(pattern))
            
            file_info = []
            for f in files[:100]:  # Limit to 100 files
                info = {
                    "path": str(f.relative_to(dir_path)),
                    "is_dir": f.is_dir(),
                    "size": f.stat().st_size if f.is_file() else 0
                }
                file_info.append(info)
            
            return {
                "success": True,
                "files": file_info,
                "count": len(files),
                "truncated": len(files) > 100
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_solidity_file(
        self,
        name: str,
        content: str,
        file_type: str = "test"
    ) -> Dict[str, Any]:
        """
        Create a Solidity file with proper structure
        
        Args:
            name: Contract name
            content: Contract code
            file_type: 'test', 'script', or 'src'
            
        Returns:
            Dict with success, path
        """
        # Determine file extension and directory
        if file_type == "test":
            ext = ".t.sol"
            subdir = "test"
        elif file_type == "script":
            ext = ".s.sol"
            subdir = "script"
        else:
            ext = ".sol"
            subdir = "src"
        
        filename = f"{name}{ext}"
        path = f"{subdir}/{filename}"
        
        return self.write(path, content)
    
    def search_files(
        self,
        pattern: str,
        search_content: Optional[str] = None,
        path: str = "."
    ) -> Dict[str, Any]:
        """
        Search for files, optionally by content
        
        Args:
            pattern: File name pattern
            search_content: Content to search within files
            path: Directory to search in
            
        Returns:
            Dict with matching files
        """
        dir_path = self._resolve_path(path)
        
        try:
            files = list(dir_path.rglob(pattern))
            
            if search_content:
                matching_files = []
                for f in files:
                    if f.is_file():
                        try:
                            content = f.read_text()
                            if search_content in content:
                                matching_files.append(str(f.relative_to(dir_path)))
                        except Exception:
                            pass
                return {
                    "success": True,
                    "files": matching_files[:50],
                    "count": len(matching_files)
                }
            else:
                return {
                    "success": True,
                    "files": [str(f.relative_to(dir_path)) for f in files[:50]],
                    "count": len(files)
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

