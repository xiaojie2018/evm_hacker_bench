"""
Slither Tool - Static Analysis for Smart Contracts

Based on SCONE-bench requirements:
- Run Slither static analysis on contracts
- Parse and categorize vulnerabilities
- Integration with Foundry projects
"""

import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class SeverityLevel(Enum):
    """Vulnerability severity levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"
    OPTIMIZATION = "optimization"


@dataclass
class SlitherFinding:
    """A single Slither finding"""
    check: str
    severity: str
    confidence: str
    description: str
    elements: List[Dict]
    
    def to_dict(self) -> Dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "confidence": self.confidence,
            "description": self.description,
            "elements_count": len(self.elements)
        }


class SlitherTool:
    """
    Slither Static Analysis Tool
    
    Features:
    - Run Slither on Solidity files/projects
    - Parse JSON output
    - Categorize findings by severity
    - Integration with exploit development workflow
    """
    
    def __init__(
        self,
        slither_path: str = "slither",
        solc_version: Optional[str] = None,
        timeout: int = 300
    ):
        """
        Initialize Slither tool
        
        Args:
            slither_path: Path to slither binary
            solc_version: Solidity compiler version
            timeout: Analysis timeout in seconds
        """
        self.slither_path = slither_path
        self.solc_version = solc_version
        self.timeout = timeout
    
    def check_installed(self) -> tuple[bool, str]:
        """Check if Slither is installed"""
        try:
            result = subprocess.run(
                [self.slither_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return True, result.stdout.strip()
            return False, "Slither not installed"
        except Exception as e:
            return False, str(e)
    
    def analyze(
        self,
        target: str,
        detectors: Optional[List[str]] = None,
        exclude_detectors: Optional[List[str]] = None,
        filter_paths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run Slither analysis on target
        
        Args:
            target: Path to Solidity file or directory
            detectors: Specific detectors to run
            exclude_detectors: Detectors to exclude
            filter_paths: Paths to exclude from analysis
            
        Returns:
            Analysis results
        """
        target_path = Path(target)
        
        if not target_path.exists():
            return {
                "success": False,
                "error": f"Target not found: {target}"
            }
        
        # Build command
        cmd = [self.slither_path, str(target_path), "--json", "-"]
        
        if self.solc_version:
            cmd.extend(["--solc-solcs-select", self.solc_version])
        
        if detectors:
            cmd.extend(["--detect", ",".join(detectors)])
        
        if exclude_detectors:
            cmd.extend(["--exclude", ",".join(exclude_detectors)])
        
        if filter_paths:
            for fp in filter_paths:
                cmd.extend(["--filter-paths", fp])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # Parse JSON output
            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    return self._parse_results(data)
                except json.JSONDecodeError:
                    pass
            
            # If JSON parsing fails, try to extract useful info
            return {
                "success": result.returncode == 0,
                "raw_output": result.stdout[:5000],
                "error": result.stderr[:1000] if result.stderr else None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Analysis timed out after {self.timeout}s"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _parse_results(self, data: Dict) -> Dict[str, Any]:
        """Parse Slither JSON output"""
        findings = []
        
        if "results" in data and "detectors" in data["results"]:
            for detector in data["results"]["detectors"]:
                finding = SlitherFinding(
                    check=detector.get("check", "unknown"),
                    severity=detector.get("impact", "unknown"),
                    confidence=detector.get("confidence", "unknown"),
                    description=detector.get("description", ""),
                    elements=detector.get("elements", [])
                )
                findings.append(finding)
        
        # Categorize by severity
        by_severity = {
            "high": [],
            "medium": [],
            "low": [],
            "informational": [],
            "optimization": []
        }
        
        for f in findings:
            severity = f.severity.lower()
            if severity in by_severity:
                by_severity[severity].append(f.to_dict())
            else:
                by_severity["informational"].append(f.to_dict())
        
        return {
            "success": True,
            "total_findings": len(findings),
            "by_severity": by_severity,
            "high_count": len(by_severity["high"]),
            "medium_count": len(by_severity["medium"]),
            "findings": [f.to_dict() for f in findings[:50]]  # Limit output
        }
    
    def find_reentrancy(self, target: str) -> Dict[str, Any]:
        """Run reentrancy-specific detectors"""
        detectors = [
            "reentrancy-eth",
            "reentrancy-no-eth",
            "reentrancy-benign",
            "reentrancy-events",
            "reentrancy-unlimited-gas"
        ]
        return self.analyze(target, detectors=detectors)
    
    def find_access_control(self, target: str) -> Dict[str, Any]:
        """Run access control detectors"""
        detectors = [
            "arbitrary-send-eth",
            "arbitrary-send-erc20",
            "controlled-delegatecall",
            "protected-vars",
            "suicidal",
            "unprotected-upgrade"
        ]
        return self.analyze(target, detectors=detectors)
    
    def find_oracle_issues(self, target: str) -> Dict[str, Any]:
        """Run oracle/price manipulation detectors"""
        detectors = [
            "incorrect-modifier",
            "unchecked-transfer",
            "unchecked-lowlevel",
            "unchecked-send",
            "weak-prng"
        ]
        return self.analyze(target, detectors=detectors)
    
    def quick_scan(self, target: str) -> Dict[str, Any]:
        """
        Quick security scan focusing on high-impact issues
        
        Args:
            target: Path to analyze
            
        Returns:
            High-priority findings only
        """
        high_impact_detectors = [
            "reentrancy-eth",
            "arbitrary-send-eth",
            "controlled-delegatecall",
            "suicidal",
            "unprotected-upgrade",
            "unchecked-transfer"
        ]
        
        return self.analyze(target, detectors=high_impact_detectors)
    
    def format_findings_for_llm(self, results: Dict) -> str:
        """
        Format findings for LLM consumption
        
        Args:
            results: Slither analysis results
            
        Returns:
            Formatted string for LLM prompt
        """
        if not results.get("success"):
            return f"Slither analysis failed: {results.get('error', 'Unknown error')}"
        
        lines = [
            "## Slither Static Analysis Results",
            f"\nTotal findings: {results.get('total_findings', 0)}",
            f"High severity: {results.get('high_count', 0)}",
            f"Medium severity: {results.get('medium_count', 0)}",
            "\n### High Severity Findings:"
        ]
        
        for finding in results.get("by_severity", {}).get("high", []):
            lines.append(f"\n**{finding['check']}** (confidence: {finding['confidence']})")
            lines.append(finding['description'][:500])
        
        lines.append("\n### Medium Severity Findings:")
        for finding in results.get("by_severity", {}).get("medium", [])[:5]:
            lines.append(f"\n**{finding['check']}**")
            lines.append(finding['description'][:300])
        
        return "\n".join(lines)

