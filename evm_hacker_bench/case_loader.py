"""
Attack Case Loader - Load and parse attack cases from multiple sources

Supports:
1. SCONE-bench CSV format
2. DeFiHackLabs POC files
3. Custom JSON format
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class AttackCategory(Enum):
    """Categories of smart contract attacks"""
    REENTRANCY = "reentrancy"
    FLASH_LOAN = "flash_loan"
    PRICE_MANIPULATION = "price_manipulation"
    ACCESS_CONTROL = "access_control"
    INTEGER_OVERFLOW = "integer_overflow"
    LOGIC_FLAW = "logic_flaw"
    PRECISION_LOSS = "precision_loss"
    FRONT_RUNNING = "front_running"
    SIGNATURE = "signature"
    ORACLE = "oracle"
    UNKNOWN = "unknown"


@dataclass
class AttackCase:
    """Represents a single attack case"""
    case_id: str
    case_name: str
    chain: str
    target_address: str
    fork_block: int
    evm_version: Optional[str] = None
    
    # Attack details
    category: AttackCategory = AttackCategory.UNKNOWN
    description: str = ""
    lost_amount: str = ""
    attack_date: Optional[str] = None  # Attack date in YYYY-MM format (extracted from POC path or CSV)
    
    # Source information
    source: str = ""  # defihacklabs, scone-bench, etc.
    poc_file: Optional[str] = None
    reference_links: List[str] = field(default_factory=list)
    
    # Contract source (if available)
    contract_source: Optional[str] = None
    contract_abi: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "case_id": self.case_id,
            "case_name": self.case_name,
            "chain": self.chain,
            "target_address": self.target_address,
            "fork_block": self.fork_block,
            "evm_version": self.evm_version,
            "category": self.category.value,
            "description": self.description,
            "lost_amount": self.lost_amount,
            "attack_date": self.attack_date,
            "source": self.source,
            "poc_file": self.poc_file,
            "reference_links": self.reference_links
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttackCase":
        """Create from dictionary"""
        category = AttackCategory.UNKNOWN
        if "category" in data:
            try:
                category = AttackCategory(data["category"])
            except ValueError:
                category = AttackCategory.UNKNOWN
        
        return cls(
            case_id=data.get("case_id", data.get("case_name", "")),
            case_name=data.get("case_name", ""),
            chain=data.get("chain", "mainnet"),
            target_address=data.get("target_address", data.get("target_contract_address", "")),
            fork_block=int(data.get("fork_block", data.get("fork_block_number", 0))),
            evm_version=data.get("evm_version") or None,
            category=category,
            description=data.get("description", ""),
            lost_amount=data.get("lost_amount", ""),
            attack_date=data.get("attack_date"),
            source=data.get("source", ""),
            poc_file=data.get("poc_file"),
            reference_links=data.get("reference_links", [])
        )


class CaseLoader:
    """Load attack cases from various sources"""
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize loader
        
        Args:
            base_path: Base path for relative file references
        """
        self.base_path = base_path or Path.cwd()
        self.cases: List[AttackCase] = []
    
    def load_scone_csv(self, csv_path: Path) -> List[AttackCase]:
        """
        Load cases from SCONE-bench CSV format
        
        Expected columns:
        - case_name
        - task_source
        - chain
        - fork_block_number
        - target_contract_address
        - evm_version
        """
        cases = []
        csv_path = Path(csv_path)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                case = AttackCase(
                    case_id=f"scone_{row['case_name']}",
                    case_name=row['case_name'],
                    chain=row.get('chain', 'mainnet'),
                    target_address=row['target_contract_address'],
                    fork_block=int(row['fork_block_number']),
                    evm_version=row.get('evm_version') or None,
                    source="scone-bench"
                )
                cases.append(case)
        
        self.cases.extend(cases)
        print(f"✓ Loaded {len(cases)} cases from SCONE-bench")
        return cases
    
    def load_defihacklabs(self, poc_dir: Path) -> List[AttackCase]:
        """
        Load cases from DeFiHackLabs POC directory
        
        Parses Solidity files for:
        - Fork block number
        - Target contracts
        - Attack category from filename/comments
        """
        cases = []
        poc_dir = Path(poc_dir)
        
        if not poc_dir.exists():
            raise FileNotFoundError(f"POC directory not found: {poc_dir}")
        
        # Find all .sol files (excluding Zone.Identifier files)
        sol_files = list(poc_dir.rglob("*.sol"))
        sol_files = [f for f in sol_files if "Zone.Identifier" not in f.name]
        
        for sol_file in sol_files:
            try:
                case = self._parse_defihacklabs_poc(sol_file)
                if case:
                    cases.append(case)
            except Exception as e:
                print(f"⚠️  Failed to parse {sol_file.name}: {e}")
        
        self.cases.extend(cases)
        print(f"✓ Loaded {len(cases)} cases from DeFiHackLabs")
        return cases
    
    def _parse_defihacklabs_poc(self, sol_file: Path) -> Optional[AttackCase]:
        """Parse a DeFiHackLabs POC file"""
        content = sol_file.read_text(encoding='utf-8', errors='ignore')
        
        # Extract case name from filename
        case_name = sol_file.stem.replace('_exp', '')
        
        # Extract date from parent directory (e.g., 2024-01)
        parent_name = sol_file.parent.name
        date_match = re.match(r'(\d{4})-(\d{2})', parent_name)
        
        # Extract fork block from vm.createSelectFork
        fork_match = re.search(
            r'vm\.createSelectFork\s*\(\s*["\'](\w+)["\']\s*,\s*(\d+(?:_\d+)*)\s*\)',
            content
        )
        
        if not fork_match:
            # Try alternative pattern
            fork_match = re.search(
                r'vm\.createSelectFork\s*\(\s*["\'](\w+)["\']\s*,\s*(\d+)\s*\)',
                content
            )
        
        if not fork_match:
            return None
        
        chain = fork_match.group(1).lower()
        fork_block = int(fork_match.group(2).replace('_', ''))
        
        # Map chain names
        chain_map = {
            'mainnet': 'mainnet',
            'eth': 'mainnet',
            'ethereum': 'mainnet',
            'bsc': 'bsc',
            'binance': 'bsc',
            'arbitrum': 'arbitrum',
            'arb': 'arbitrum',
            'base': 'base',
            'polygon': 'polygon',
            'optimism': 'optimism',
            'op': 'optimism'
        }
        chain = chain_map.get(chain, chain)
        
        # Extract target contract (look for constant addresses)
        target_address = None
        addr_matches = re.findall(
            r'(?:constant|address)\s+\w+\s*=\s*(0x[a-fA-F0-9]{40})',
            content
        )
        if addr_matches:
            target_address = addr_matches[0]
        
        # Extract lost amount from comments
        lost_match = re.search(r'@KeyInfo.*Lost\s*:?\s*~?\s*\$?([\d,\.]+\s*\w+)', content)
        lost_amount = lost_match.group(1) if lost_match else ""
        
        # Detect attack category from filename or content
        category = self._detect_category(case_name, content)
        
        # Extract reference links
        ref_links = re.findall(r'https?://[^\s\'"<>]+', content)
        
        # Extract attack date from parent directory name (e.g., 2024-01)
        attack_date = None
        if date_match:
            attack_date = f"{date_match.group(1)}-{date_match.group(2)}"
        
        return AttackCase(
            case_id=f"dfl_{parent_name}_{case_name}",
            case_name=case_name,
            chain=chain,
            target_address=target_address or "",
            fork_block=fork_block,
            category=category,
            lost_amount=lost_amount,
            attack_date=attack_date,
            source="defihacklabs",
            poc_file=str(sol_file),
            reference_links=ref_links[:5]  # Limit to 5 links
        )
    
    def _detect_category(self, name: str, content: str) -> AttackCategory:
        """Detect attack category from name and content"""
        name_lower = name.lower()
        content_lower = content.lower()
        
        # Category keywords
        keywords = {
            AttackCategory.REENTRANCY: ['reentrancy', 'reentrant', 're-entry'],
            AttackCategory.FLASH_LOAN: ['flashloan', 'flash_loan', 'flash loan', 'aave.flashloan'],
            AttackCategory.PRICE_MANIPULATION: ['price manipulation', 'price oracle', 'manipulation', 'oracle'],
            AttackCategory.ACCESS_CONTROL: ['access control', 'unauthorized', 'permission', 'onlyowner'],
            AttackCategory.INTEGER_OVERFLOW: ['overflow', 'underflow', 'integer'],
            AttackCategory.LOGIC_FLAW: ['logic flaw', 'business logic', 'logic error'],
            AttackCategory.PRECISION_LOSS: ['precision', 'rounding', 'truncation'],
            AttackCategory.FRONT_RUNNING: ['front-run', 'frontrun', 'sandwich', 'mev'],
            AttackCategory.SIGNATURE: ['signature', 'ecdsa', 'replay', 'verification']
        }
        
        for category, kws in keywords.items():
            for kw in kws:
                if kw in name_lower or kw in content_lower:
                    return category
        
        return AttackCategory.UNKNOWN
    
    def load_json(self, json_path: Path) -> List[AttackCase]:
        """Load cases from JSON file"""
        json_path = Path(json_path)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cases = []
        items = data if isinstance(data, list) else data.get('cases', [])
        
        for item in items:
            case = AttackCase.from_dict(item)
            cases.append(case)
        
        self.cases.extend(cases)
        print(f"✓ Loaded {len(cases)} cases from JSON")
        return cases
    
    def save_json(self, output_path: Path):
        """Save all loaded cases to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "total": len(self.cases),
            "cases": [case.to_dict() for case in self.cases]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(self.cases)} cases to {output_path}")
    
    def enrich_with_poc(self, defihacklabs_dir: Path) -> int:
        """
        Try to match and enrich cases with DeFiHackLabs POC files
        
        Args:
            defihacklabs_dir: Path to DeFiHackLabs test directory
            
        Returns:
            Number of cases enriched
        """
        defihacklabs_dir = Path(defihacklabs_dir)
        if not defihacklabs_dir.exists():
            return 0
        
        # Build a map of case names to POC files
        poc_map = {}
        sol_files = list(defihacklabs_dir.rglob("*.sol"))
        sol_files = [f for f in sol_files if "Zone.Identifier" not in f.name]
        
        for sol_file in sol_files:
            # Extract case name from filename
            name = sol_file.stem.replace('_exp', '').lower()
            poc_map[name] = sol_file
        
        enriched_count = 0
        for case in self.cases:
            if case.poc_file:
                continue  # Already has POC
            
            # Try to find matching POC
            case_name_lower = case.case_name.lower()
            
            # Direct match
            if case_name_lower in poc_map:
                case.poc_file = str(poc_map[case_name_lower])
                enriched_count += 1
                continue
            
            # Fuzzy match: check if case name contains POC name or vice versa
            for poc_name, poc_file in poc_map.items():
                if poc_name in case_name_lower or case_name_lower in poc_name:
                    case.poc_file = str(poc_file)
                    enriched_count += 1
                    break
        
        if enriched_count > 0:
            print(f"✓ Enriched {enriched_count} cases with POC files")
        return enriched_count
    
    def filter_by_chain(self, chain: str) -> List[AttackCase]:
        """Filter cases by chain"""
        return [c for c in self.cases if c.chain == chain]
    
    def filter_by_category(self, category: AttackCategory) -> List[AttackCase]:
        """Filter cases by attack category"""
        return [c for c in self.cases if c.category == category]
    
    def get_case(self, case_id: str) -> Optional[AttackCase]:
        """Get case by ID"""
        for case in self.cases:
            if case.case_id == case_id or case.case_name == case_id:
                return case
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded cases"""
        chain_counts = {}
        category_counts = {}
        source_counts = {}
        
        for case in self.cases:
            chain_counts[case.chain] = chain_counts.get(case.chain, 0) + 1
            category_counts[case.category.value] = category_counts.get(case.category.value, 0) + 1
            source_counts[case.source] = source_counts.get(case.source, 0) + 1
        
        return {
            "total": len(self.cases),
            "by_chain": chain_counts,
            "by_category": category_counts,
            "by_source": source_counts
        }


def create_combined_dataset(
    scone_csv: Optional[Path] = None,
    defihacklabs_dir: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> "CaseLoader":
    """
    Create a combined dataset from multiple sources
    
    Args:
        scone_csv: Path to SCONE-bench CSV
        defihacklabs_dir: Path to DeFiHackLabs POC directory
        output_path: Path to save combined JSON
        
    Returns:
        Loaded CaseLoader instance
    """
    loader = CaseLoader()
    
    if scone_csv and Path(scone_csv).exists():
        loader.load_scone_csv(scone_csv)
    
    if defihacklabs_dir and Path(defihacklabs_dir).exists():
        loader.load_defihacklabs(defihacklabs_dir)
    
    if output_path:
        loader.save_json(output_path)
    
    # Print statistics
    stats = loader.get_statistics()
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total Cases: {stats['total']}")
    print(f"   By Chain: {stats['by_chain']}")
    print(f"   By Source: {stats['by_source']}")
    
    return loader

