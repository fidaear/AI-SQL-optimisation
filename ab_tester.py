"""
A/B Tester for SQL Query Optimization
Compares original vs optimized queries with measurable metrics.
"""

import logging
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from statistics import mean, stdev
import json

logger = logging.getLogger(__name__)


@dataclass
class QueryExecutionResult:
    """Result of a single query execution."""
    query_id: str
    execution_time_ms: float
    rows_returned: int
    rows_scanned: int
    estimated_cost: float
    execution_plan: Optional[Dict[str, Any]] = None


@dataclass
class TestResult:
    """Results from one test run (query type)."""
    query_name: str
    iterations: int
    execution_times: List[float] = field(default_factory=list)
    rows_returned_list: List[int] = field(default_factory=list)
    
    @property
    def average_time_ms(self) -> float:
        """Average execution time in milliseconds."""
        return mean(self.execution_times) if self.execution_times else 0.0
    
    @property
    def min_time_ms(self) -> float:
        """Minimum execution time."""
        return min(self.execution_times) if self.execution_times else 0.0
    
    @property
    def max_time_ms(self) -> float:
        """Maximum execution time."""
        return max(self.execution_times) if self.execution_times else 0.0
    
    @property
    def stddev_ms(self) -> float:
        """Standard deviation of execution times."""
        if len(self.execution_times) < 2:
            return 0.0
        return stdev(self.execution_times)
    
    @property
    def average_rows(self) -> float:
        """Average rows returned."""
        return mean(self.rows_returned_list) if self.rows_returned_list else 0.0


@dataclass
class ComparisonResult:
    """Results comparing original vs optimized queries."""
    original: TestResult
    optimized: TestResult
    
    @property
    def improvement_percentage(self) -> float:
        """
        Calculate improvement percentage.
        Positive = optimized is faster
        Negative = optimized is slower
        """
        if self.original.average_time_ms == 0:
            return 0.0
        
        improvement = ((self.original.average_time_ms - self.optimized.average_time_ms) 
                      / self.original.average_time_ms * 100)
        return improvement
    
    @property
    def speedup_factor(self) -> float:
        """Calculate speedup factor (e.g., 10x faster)."""
        if self.optimized.average_time_ms == 0:
            return 0.0
        return self.original.average_time_ms / self.optimized.average_time_ms
    
    @property
    def is_faster(self) -> bool:
        """Check if optimized query is faster."""
        return self.improvement_percentage > 0
    
    @property
    def is_significantly_faster(self, threshold_pct: float = 10.0) -> bool:
        """Check if improvement exceeds threshold (default 10%)."""
        return self.improvement_percentage >= threshold_pct


class ABTester:
    """
    A/B Testing framework for SQL query optimization.
    
    Compares original vs optimized queries by:
    1. Executing each query multiple times
    2. Collecting timing metrics
    3. Calculating average performance
    4. Computing improvement percentage
    """
    
    def __init__(self, executor, iterations: int = 5, warmup_iterations: int = 1):
        """
        Initialize A/B tester.
        
        Args:
            executor: Query executor instance (from executor.py)
            iterations: Number of times to run each query
            warmup_iterations: Number of warmup runs before actual testing
        """
        self.executor = executor
        self.iterations = iterations
        self.warmup_iterations = warmup_iterations
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        if warmup_iterations < 0:
            raise ValueError("warmup_iterations must be >= 0")
    
    def run_test(self, query: str, query_name: str, iterations: Optional[int] = None) -> TestResult:
        """
        Execute a query multiple times and collect metrics.
        
        Args:
            query: SQL query to execute
            query_name: Name for this test (e.g., "original" or "optimized")
            iterations: Override default iteration count
            
        Returns:
            TestResult with collected metrics
        """
        num_iterations = iterations or self.iterations
        test_result = TestResult(query_name=query_name, iterations=num_iterations)
        
        self.logger.info(f"Starting {query_name} test ({num_iterations} iterations)")
        
        # Warmup runs (to prime caches, etc.)
        if self.warmup_iterations > 0:
            self.logger.debug(f"Warmup: {self.warmup_iterations} iterations")
            for i in range(self.warmup_iterations):
                try:
                    self.executor.execute(query)
                except Exception as e:
                    self.logger.warning(f"Warmup iteration {i+1} failed: {str(e)}")
        
        # Actual test runs
        for iteration in range(num_iterations):
            try:
                start_time = time.time()
                result = self.executor.execute(query)
                execution_time_ms = (time.time() - start_time) * 1000
                
                # Collect metrics
                test_result.execution_times.append(execution_time_ms)
                test_result.rows_returned_list.append(
                    result.get("rows_returned", 0) if result else 0
                )
                
                self.logger.debug(f"{query_name} iteration {iteration+1}/{num_iterations}: {execution_time_ms:.2f}ms")
            
            except Exception as e:
                self.logger.error(f"{query_name} iteration {iteration+1} failed: {str(e)}")
                raise
        
        self.logger.info(
            f"{query_name} test complete: avg={test_result.average_time_ms:.2f}ms, "
            f"min={test_result.min_time_ms:.2f}ms, max={test_result.max_time_ms:.2f}ms"
        )
        
        return test_result
    
    def compare_queries(self, 
                       original_query: str, 
                       optimized_query: str,
                       original_name: str = "Original",
                       optimized_name: str = "Optimized") -> ComparisonResult:
        """
        Run A/B test comparing two queries.
        
        Args:
            original_query: The baseline query
            optimized_query: The optimized query to compare
            original_name: Label for original query
            optimized_name: Label for optimized query
            
        Returns:
            ComparisonResult with metrics and improvement %
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting A/B Test")
        self.logger.info("=" * 60)
        
        # Run both queries
        original_result = self.run_test(original_query, original_name)
        optimized_result = self.run_test(optimized_query, optimized_name)
        
        # Compare
        comparison = ComparisonResult(original=original_result, optimized=optimized_result)
        
        self.logger.info("=" * 60)
        self.logger.info("A/B TEST RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"\n{original_name}:")
        self.logger.info(f"  Average: {original_result.average_time_ms:.2f}ms")
        self.logger.info(f"  Min:     {original_result.min_time_ms:.2f}ms")
        self.logger.info(f"  Max:     {original_result.max_time_ms:.2f}ms")
        self.logger.info(f"  StdDev:  {original_result.stddev_ms:.2f}ms")
        self.logger.info(f"  Rows:    {int(original_result.average_rows)}")
        
        self.logger.info(f"\n{optimized_name}:")
        self.logger.info(f"  Average: {optimized_result.average_time_ms:.2f}ms")
        self.logger.info(f"  Min:     {optimized_result.min_time_ms:.2f}ms")
        self.logger.info(f"  Max:     {optimized_result.max_time_ms:.2f}ms")
        self.logger.info(f"  StdDev:  {optimized_result.stddev_ms:.2f}ms")
        self.logger.info(f"  Rows:    {int(optimized_result.average_rows)}")
        
        self.logger.info(f"\nIMPROVEMENT:")
        if comparison.is_faster:
            self.logger.info(f"  ✓ {optimized_name} is FASTER")
            self.logger.info(f"  Improvement: {comparison.improvement_percentage:.1f}%")
            self.logger.info(f"  Speedup: {comparison.speedup_factor:.2f}x")
            self.logger.info(f"  Time saved: {original_result.average_time_ms - optimized_result.average_time_ms:.2f}ms per query")
        else:
            self.logger.info(f"  ✗ {optimized_name} is SLOWER")
            self.logger.info(f"  Regression: {abs(comparison.improvement_percentage):.1f}%")
            self.logger.info(f"  Slowdown: {abs(comparison.speedup_factor - 1):.2f}x slower")
        
        self.logger.info("=" * 60)
        
        return comparison
    
    def format_results_table(self, comparison: ComparisonResult) -> str:
        """Format comparison results as ASCII table."""
        orig = comparison.original
        opt = comparison.optimized
        
        table = f"""
╔════════════════════════════════════════════════════════════════╗
║                    A/B TEST RESULTS                           ║
╠════════════════════════════════════════════════════════════════╣
║ Metric              │ {orig.query_name:15} │ {opt.query_name:15} │
╠═════════════════════╪════════════════════╪════════════════════╣
║ Average (ms)        │ {orig.average_time_ms:18.2f} │ {opt.average_time_ms:18.2f} │
║ Min (ms)            │ {orig.min_time_ms:18.2f} │ {opt.min_time_ms:18.2f} │
║ Max (ms)            │ {orig.max_time_ms:18.2f} │ {opt.max_time_ms:18.2f} │
║ StdDev (ms)         │ {orig.stddev_ms:18.2f} │ {opt.stddev_ms:18.2f} │
║ Rows Returned       │ {int(orig.average_rows):18} │ {int(opt.average_rows):18} │
╠═════════════════════╧════════════════════╧════════════════════╣
║ IMPROVEMENT: {comparison.improvement_percentage:+.1f}%  ({comparison.speedup_factor:.2f}x speedup)
║ STATUS: {"✓ FASTER" if comparison.is_faster else "✗ SLOWER"}
╚════════════════════════════════════════════════════════════════╝
"""
        return table
    
    def export_results_json(self, comparison: ComparisonResult, filepath: str):
        """Export results to JSON file."""
        data = {
            "test_config": {
                "iterations": comparison.original.iterations,
                "warmup_iterations": self.warmup_iterations
            },
            "original": {
                "name": comparison.original.query_name,
                "average_ms": comparison.original.average_time_ms,
                "min_ms": comparison.original.min_time_ms,
                "max_ms": comparison.original.max_time_ms,
                "stddev_ms": comparison.original.stddev_ms,
                "rows_average": int(comparison.original.average_rows),
            },
            "optimized": {
                "name": comparison.optimized.query_name,
                "average_ms": comparison.optimized.average_time_ms,
                "min_ms": comparison.optimized.min_time_ms,
                "max_ms": comparison.optimized.max_time_ms,
                "stddev_ms": comparison.optimized.stddev_ms,
                "rows_average": int(comparison.optimized.average_rows),
            },
            "comparison": {
                "improvement_percentage": comparison.improvement_percentage,
                "speedup_factor": comparison.speedup_factor,
                "is_faster": comparison.is_faster,
                "time_saved_ms_per_query": comparison.original.average_time_ms - comparison.optimized.average_time_ms
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Results exported to {filepath}")


class ABTestRunner:
    """High-level interface for running A/B tests."""
    
    def __init__(self, executor, config: Dict[str, Any] = None):
        """
        Initialize runner with config.
        
        Args:
            executor: Query executor instance
            config: Configuration dict with:
                - iterations: Number of test runs (default: 5)
                - warmup_iterations: Warmup runs (default: 1)
                - threshold_improvement_pct: Threshold for significant improvement (default: 10%)
        """
        self.config = config or {}
        self.iterations = self.config.get("iterations", 5)
        self.warmup_iterations = self.config.get("warmup_iterations", 1)
        self.threshold = self.config.get("threshold_improvement_pct", 10.0)
        
        self.tester = ABTester(
            executor=executor,
            iterations=self.iterations,
            warmup_iterations=self.warmup_iterations
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run_and_report(self, 
                      original_query: str, 
                      optimized_query: str,
                      export_json: Optional[str] = None) -> Dict[str, Any]:
        """
        Run A/B test and generate report.
        
        Args:
            original_query: Baseline query
            optimized_query: Optimized query
            export_json: Optional filepath to export results
            
        Returns:
            Dictionary with test results and verdict
        """
        comparison = self.tester.compare_queries(original_query, optimized_query)
        
        if export_json:
            self.tester.export_results_json(comparison, export_json)
        
        # Generate verdict
        verdict = {
            "original_avg_ms": comparison.original.average_time_ms,
            "optimized_avg_ms": comparison.optimized.average_time_ms,
            "improvement_pct": comparison.improvement_percentage,
            "speedup_factor": comparison.speedup_factor,
            "is_faster": comparison.is_faster,
            "is_significant": comparison.improvement_percentage >= self.threshold,
            "verdict": self._generate_verdict(comparison)
        }
        
        return verdict
    
    def _generate_verdict(self, comparison: ComparisonResult) -> str:
        """Generate human-readable verdict."""
        if not comparison.is_faster:
            return f" REGRESSION: {abs(comparison.improvement_percentage):.1f}% slower"
        
        if comparison.improvement_percentage < self.threshold:
            return f"  NEGLIGIBLE: {comparison.improvement_percentage:.1f}% faster (< {self.threshold}% threshold)"
        
        if comparison.improvement_percentage < 50:
            return f" GOOD: {comparison.improvement_percentage:.1f}% faster ({comparison.speedup_factor:.2f}x)"
        
        return f" EXCELLENT: {comparison.improvement_percentage:.1f}% faster ({comparison.speedup_factor:.2f}x)"
