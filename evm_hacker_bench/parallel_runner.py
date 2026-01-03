"""
Parallel Runner - Run multiple models in parallel with separate Anvil instances

This module provides:
1. PortManager - Manages Anvil port allocation for parallel execution
2. ParallelModelRunner - Runs multiple models in parallel
3. ModelTask - Represents a single model benchmark task

Usage:
    from evm_hacker_bench.parallel_runner import ParallelModelRunner
    
    runner = ParallelModelRunner(
        models=["anthropic/claude-sonnet-4", "openai/gpt-4o"],
        api_key="your-api-key",
        max_parallel=2
    )
    results = runner.run(cases, output_dir="logs/parallel")
"""

import os
import sys
import json
import time
import socket
import signal
import threading
import subprocess
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

from .case_loader import AttackCase
from .evm_env import EVMEnvironment
from .hacker_controller import HackerController


class PortManager:
    """
    Thread-safe port allocation manager for parallel Anvil instances.
    
    Allocates ports starting from base_port (default 8545) and tracks
    which ports are in use.
    """
    
    def __init__(self, base_port: int = 8545, max_ports: int = 100):
        """
        Initialize port manager.
        
        Args:
            base_port: Starting port number
            max_ports: Maximum number of ports to manage
        """
        self.base_port = base_port
        self.max_ports = max_ports
        self._lock = threading.Lock()
        self._used_ports: set = set()
    
    def allocate(self) -> int:
        """
        Allocate an available port.
        
        Returns:
            Available port number
            
        Raises:
            RuntimeError: If no ports available
        """
        with self._lock:
            for offset in range(self.max_ports):
                port = self.base_port + offset
                if port not in self._used_ports and self._is_port_free(port):
                    self._used_ports.add(port)
                    return port
            raise RuntimeError(f"No available ports in range {self.base_port}-{self.base_port + self.max_ports}")
    
    def release(self, port: int):
        """
        Release a port back to the pool.
        
        Args:
            port: Port to release
        """
        with self._lock:
            self._used_ports.discard(port)
    
    def _is_port_free(self, port: int) -> bool:
        """Check if a port is free"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            return result != 0  # Port is free if connection fails
        except Exception:
            return True


@dataclass
class ModelConfig:
    """Configuration for a model to run"""
    name: str  # Full model name (e.g., "anthropic/claude-sonnet-4")
    short_name: Optional[str] = None  # Short name for logs (e.g., "claude-sonnet")
    thinking: bool = False  # Enable extended thinking
    thinking_budget: int = 10000
    temperature: float = 0.0
    
    def __post_init__(self):
        if self.short_name is None:
            # Generate short name from model name
            self.short_name = self.name.split("/")[-1].replace("-", "_")


@dataclass
class TaskResult:
    """Result of a model benchmark task"""
    model_name: str
    short_name: str
    success: bool
    cases_total: int = 0
    cases_passed: int = 0
    total_profit: float = 0.0
    duration_seconds: float = 0.0
    error: Optional[str] = None
    log_file: Optional[str] = None
    results: List[Dict[str, Any]] = field(default_factory=list)


def _run_model_task(
    model_config: ModelConfig,
    cases: List[Dict[str, Any]],  # Serialized cases
    api_key: str,
    output_dir: str,
    port: int,
    max_turns: int,
    timeout: int,
    session_timeout: int,
    enable_compression: bool
) -> Dict[str, Any]:
    """
    Run a single model benchmark task (executed in subprocess).
    
    This function is designed to be called via ProcessPoolExecutor.
    """
    import sys
    from pathlib import Path
    
    # Reconstruct cases from dicts
    from evm_hacker_bench.case_loader import AttackCase
    from evm_hacker_bench.evm_env import EVMEnvironment
    from evm_hacker_bench.hacker_controller import HackerController
    
    cases_list = [AttackCase.from_dict(c) for c in cases]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_path / f"{model_config.short_name}_{timestamp}.log"
    
    results = []
    start_time = time.time()
    total_profit = 0.0
    
    try:
        # Open log file with line buffering for immediate writes
        with open(log_file, 'w', encoding='utf-8', buffering=1) as log:
            log.write(f"# EVM Hacker Bench - Parallel Run\n")
            log.write(f"# Model: {model_config.name}\n")
            log.write(f"# Port: {port}\n")
            log.write(f"# Cases: {len(cases_list)}\n")
            log.write(f"# Started: {datetime.now().isoformat()}\n")
            log.write(f"# {'='*60}\n\n")
            log.flush()
            
            # Redirect stdout/stderr directly to log file
            # Each subprocess has its own file descriptor after fork
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = log
            sys.stderr = log
            
            try:
                for i, case in enumerate(cases_list):
                    case_start = time.time()
                    print(f"\n[{i+1}/{len(cases_list)}] Case: {case.case_id}")
                    
                    try:
                        # Create environment with specific port
                        with EVMEnvironment(
                            chain=case.chain,
                            fork_block=case.fork_block,
                            anvil_port=port,
                            evm_version=case.evm_version
                        ) as env:
                            # Create controller
                            controller = HackerController(
                                model_name=model_config.name,
                                api_key=api_key,
                                temperature=model_config.temperature,
                                max_turns=max_turns,
                                timeout_per_turn=timeout,
                                session_timeout=session_timeout,
                                enable_compression=enable_compression,
                                enable_thinking=model_config.thinking,
                                thinking_budget=model_config.thinking_budget,
                                log_dir=output_path
                            )
                            
                            # Run attack
                            result = controller.run_attack(case, env)
                            results.append(result)
                            
                            # Track profit
                            if result.get('final_success'):
                                profit = result.get('profit', 0)
                                total_profit += profit
                                print(f"   ✅ SUCCESS! Profit: {profit:.4f}")
                            else:
                                print(f"   ❌ FAILED")
                            
                    except Exception as e:
                        error_msg = f"{type(e).__name__}: {str(e)}"
                        print(f"   ❌ ERROR: {error_msg}")
                        results.append({
                            'case_id': case.case_id,
                            'final_success': False,
                            'error': error_msg
                        })
                    
                    case_duration = time.time() - case_start
                    print(f"   Duration: {case_duration:.1f}s")
                
                # Write summary
                duration = time.time() - start_time
                passed = sum(1 for r in results if r.get('final_success'))
                print(f"\n{'='*60}")
                print(f"# SUMMARY")
                print(f"# Total: {len(results)}, Passed: {passed}")
                print(f"# Total Profit: {total_profit:.4f}")
                print(f"# Duration: {duration:.1f}s")
                print(f"# {'='*60}")
            
            finally:
                # Restore stdout/stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr
        
        return {
            'model_name': model_config.name,
            'short_name': model_config.short_name,
            'success': True,
            'cases_total': len(results),
            'cases_passed': sum(1 for r in results if r.get('final_success')),
            'total_profit': total_profit,
            'duration_seconds': time.time() - start_time,
            'log_file': str(log_file),
            'results': results
        }
        
    except Exception as e:
        return {
            'model_name': model_config.name,
            'short_name': model_config.short_name,
            'success': False,
            'error': f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            'duration_seconds': time.time() - start_time,
            'log_file': str(log_file) if log_file.exists() else None,
            'results': results
        }


class ParallelModelRunner:
    """
    Run multiple models in parallel with separate Anvil instances.
    
    Each model runs in its own process with a dedicated Anvil port.
    """
    
    def __init__(
        self,
        models: List[ModelConfig],
        api_key: str,
        max_parallel: int = 4,
        base_port: int = 8545,
        max_turns: int = 120,
        timeout: int = 120,
        session_timeout: int = 3600,
        enable_compression: bool = True
    ):
        """
        Initialize parallel runner.
        
        Args:
            models: List of model configurations to run
            api_key: API key for LLM calls
            max_parallel: Maximum number of parallel executions
            base_port: Base port for Anvil instances
            max_turns: Maximum turns per case
            timeout: Timeout per turn (seconds)
            session_timeout: Total session timeout (seconds)
            enable_compression: Enable message compression
        """
        self.models = models
        self.api_key = api_key
        self.max_parallel = min(max_parallel, len(models))
        self.port_manager = PortManager(base_port=base_port)
        self.max_turns = max_turns
        self.timeout = timeout
        self.session_timeout = session_timeout
        self.enable_compression = enable_compression
    
    def run(
        self,
        cases: List[AttackCase],
        output_dir: Path,
        progress_callback: Optional[Callable[[str, str], None]] = None
    ) -> List[TaskResult]:
        """
        Run all models in parallel.
        
        Args:
            cases: List of attack cases to run
            output_dir: Output directory for logs and results
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of TaskResult for each model
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Serialize cases for subprocess
        cases_data = [case.to_dict() for case in cases]
        
        print(f"\n{'='*70}")
        print("🚀 PARALLEL MODEL BENCHMARK")
        print(f"{'='*70}")
        print(f"Models: {len(self.models)}")
        print(f"Cases per model: {len(cases)}")
        print(f"Max parallel: {self.max_parallel}")
        print(f"Base port: {self.port_manager.base_port}")
        print(f"Output: {output_dir}")
        print(f"{'='*70}\n")
        
        results: List[TaskResult] = []
        
        # Use ProcessPoolExecutor for true process isolation
        # This prevents stdout/stderr cross-contamination between models
        # Note: spawn context is used to avoid fork issues with file descriptors
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=self.max_parallel, mp_context=ctx) as executor:
            futures = {}
            
            for model in self.models:
                # Allocate port for this model
                port = self.port_manager.allocate()
                print(f"📌 {model.short_name}: Allocated port {port}")
                
                # Submit task to process pool
                future = executor.submit(
                    _run_model_task,  # Use global function directly for process pool
                    model,
                    cases_data,
                    self.api_key,
                    str(output_dir),
                    port,
                    self.max_turns,
                    self.timeout,
                    self.session_timeout,
                    self.enable_compression
                )
                futures[future] = (model, port)
            
            # Wait for all tasks to complete
            for future in as_completed(futures):
                model, port = futures[future]
                try:
                    result_dict = future.result()
                    result = TaskResult(**result_dict)
                    results.append(result)
                    
                    status = "✅" if result.success else "❌"
                    print(f"\n{status} {model.short_name}: {result.cases_passed}/{result.cases_total} passed, "
                          f"profit={result.total_profit:.4f}, time={result.duration_seconds:.1f}s")
                    
                    if progress_callback:
                        progress_callback(model.short_name, "completed")
                        
                except Exception as e:
                    print(f"\n❌ {model.short_name}: Task failed - {e}")
                    results.append(TaskResult(
                        model_name=model.name,
                        short_name=model.short_name,
                        success=False,
                        error=str(e)
                    ))
                finally:
                    # Release port
                    self.port_manager.release(port)
        
        # Print summary
        self._print_summary(results)
        
        # Save results
        self._save_results(results, output_dir)
        
        return results
    
    def _run_single_model(
        self,
        model: ModelConfig,
        cases_data: List[Dict],
        output_dir: Path,
        port: int
    ) -> Dict[str, Any]:
        """Run a single model benchmark (in current thread)"""
        return _run_model_task(
            model_config=model,
            cases=cases_data,
            api_key=self.api_key,
            output_dir=str(output_dir),
            port=port,
            max_turns=self.max_turns,
            timeout=self.timeout,
            session_timeout=self.session_timeout,
            enable_compression=self.enable_compression
        )
    
    def _print_summary(self, results: List[TaskResult]):
        """Print summary of all results"""
        print(f"\n{'='*70}")
        print("📊 PARALLEL BENCHMARK SUMMARY")
        print(f"{'='*70}")
        
        for result in sorted(results, key=lambda r: r.cases_passed, reverse=True):
            status = "✅" if result.success else "❌"
            if result.success:
                rate = (result.cases_passed / result.cases_total * 100) if result.cases_total > 0 else 0
                print(f"{status} {result.short_name:25} | "
                      f"Pass: {result.cases_passed:3}/{result.cases_total:3} ({rate:5.1f}%) | "
                      f"Profit: {result.total_profit:8.4f} | "
                      f"Time: {result.duration_seconds:7.1f}s")
            else:
                print(f"{status} {result.short_name:25} | ERROR: {result.error[:50]}...")
        
        print(f"{'='*70}")
    
    def _save_results(self, results: List[TaskResult], output_dir: Path):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"parallel_results_{timestamp}.json"
        
        # Convert to serializable format
        results_data = []
        for r in results:
            results_data.append({
                'model_name': r.model_name,
                'short_name': r.short_name,
                'success': r.success,
                'cases_total': r.cases_total,
                'cases_passed': r.cases_passed,
                'total_profit': r.total_profit,
                'duration_seconds': r.duration_seconds,
                'error': r.error,
                'log_file': r.log_file
            })
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'models': len(results),
                'results': results_data
            }, f, indent=2)
        
        print(f"\n📁 Results saved to: {results_file}")


def create_model_configs(model_specs: List[str]) -> List[ModelConfig]:
    """
    Create ModelConfig objects from model specification strings.
    
    Format: "model_name" or "model_name:thinking" or "short_name=model_name:thinking"
    
    Examples:
        - "anthropic/claude-sonnet-4"
        - "anthropic/claude-sonnet-4:thinking"
        - "claude-sonnet=anthropic/claude-sonnet-4:thinking"
    """
    configs = []
    for spec in model_specs:
        parts = spec.split('=', 1)
        if len(parts) == 2:
            short_name, rest = parts
        else:
            short_name = None
            rest = parts[0]
        
        thinking = ':thinking' in rest
        model_name = rest.replace(':thinking', '')
        
        configs.append(ModelConfig(
            name=model_name,
            short_name=short_name,
            thinking=thinking
        ))
    
    return configs

