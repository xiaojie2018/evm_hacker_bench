"""
EVM Hacker Bench - Tool Suite

Tools based on SCONE-bench requirements:
1. bash - Persistent bash session
2. file_editor - File CRUD operations
3. slither - Static analysis
4. uniswap_smart_path - DEX path finding
5. solc_manager - Solidity version management
"""

from .bash_tool import BashTool
from .file_editor import FileEditor
from .slither_tool import SlitherTool
from .uniswap_path import UniswapSmartPath
from .solc_manager import SolcManager

__all__ = [
    'BashTool',
    'FileEditor', 
    'SlitherTool',
    'UniswapSmartPath',
    'SolcManager'
]

