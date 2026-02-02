"""
Main validation orchestrator for immersion cooling fluids.

This module provides the high-level interface for running complete
validation workflows combining thermal, electrical, lifetime, and
economic analyses.

Example usage:
    >>> from icv import ImmersionCoolingValidator, GroupIIIOil
    >>> fluid = GroupIIIOil()
    >>> validator = ImmersionCoolingValidator(fluid)
    >>> result = validator.run_full_validation()
    >>> print(result.summary())
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
from datetime import datetime

from .fluids import Fluid, GroupIIIOil, FluorinertFC77
from .thermal import GPUSpecification, HeatSinkGeometry
from .monte_carlo import MonteCarloEngine, MonteCarloConfig, MonteCarloResult
from .compliance import ComplianceChecker, ComplianceResult, OCPSpecification
from .economics import (
    EconomicAnalysis,
    FluidCost,
    DataCenterConfig,
    CostComparison,
    ForexSavingsProjection,
)


@dataclass
class ValidationResult:
    """
    Complete validation result combining all analyses.
    """
    
    # Metadata
    fluid_name: str
    timestamp: str
    
    # Monte Carlo results
    monte_carlo: MonteCarloResult
    
    # Compliance assessment
    compliance: ComplianceResult
    
    # Economic comparison (optional)
    economics: Optional[CostComparison] = None
    forex_projection: Optional[ForexSavingsProjection] = None
    
    # Benchmark comparison (optional)
    benchmark_comparison: Optional[Dict[str, Any]] = None
    
    def summary(self) -> str:
        """Generate comprehensive validation summary."""
        lines = [
            "=" * 70,
            "IMMERSION COOLING FLUID VALIDATION REPORT",
            "=" * 70,
            f"Fluid: {self.fluid_name}",
            f"Timestamp: {self.timestamp}",
            "",
            self.monte_carlo.summary(),
            "",
            self.compliance.summary(),
        ]
        
        if self.economics:
            lines.append("")
            lines.append(self.economics.summary())
        
        if self.forex_projection:
            lines.append("")
            lines.append(f"FOREX SAVINGS PROJECTION ({self.forex_projection.years[0]}-{self.forex_projection.years[-1]}):")
            lines.append(f"  Total: ${self.forex_projection.total_savings_usd:,.0f} USD")
            lines.append(f"         ₹{self.forex_projection.total_savings_inr_crores:.0f} Crores")
        
        if self.benchmark_comparison:
            lines.append("")
            lines.append("BENCHMARK COMPARISON:")
            lines.append("-" * 50)
            for metric, data in self.benchmark_comparison.items():
                lines.append(f"  {metric}:")
                lines.append(f"    {self.fluid_name}: {data['test']:.4g}")
                lines.append(f"    Fluorinert FC-77: {data['benchmark']:.4g}")
                lines.append(f"    Advantage: {data['advantage']}")
        
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"VERDICT: {'✅ PASS' if self.compliance.critical_pass else '❌ FAIL'}")
        lines.append(f"Joint Compliance Probability: {100*self.monte_carlo.joint_compliance_prob:.1f}%")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": {
                "fluid_name": self.fluid_name,
                "timestamp": self.timestamp,
            },
            "monte_carlo": {
                "n_samples": self.monte_carlo.n_samples,
                "junction_temperature": {
                    "mean": self.monte_carlo.t_junction_mean,
                    "std": self.monte_carlo.t_junction_std,
                    "percentiles": self.monte_carlo.t_junction_percentiles,
                },
                "breakdown_voltage": {
                    "mean": self.monte_carlo.bdv_mean,
                    "std": self.monte_carlo.bdv_std,
                },
                "compliance_probabilities": {
                    "thermal": self.monte_carlo.thermal_compliance_prob,
                    "bdv": self.monte_carlo.bdv_compliance_prob,
                    "resistivity": self.monte_carlo.resistivity_compliance_prob,
                    "lifetime": self.monte_carlo.lifetime_compliance_prob,
                    "joint": self.monte_carlo.joint_compliance_prob,
                },
            },
            "compliance": self.compliance.to_dict(),
            "economics": {
                "savings_usd": self.economics.absolute_savings_usd if self.economics else None,
                "savings_percent": self.economics.relative_savings_percent if self.economics else None,
            },
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())


class ImmersionCoolingValidator:
    """
    Main validation engine for immersion cooling fluids.
    
    Orchestrates Monte Carlo simulation, compliance checking,
    and economic analysis.
    
    Example:
        >>> validator = ImmersionCoolingValidator(GroupIIIOil())
        >>> result = validator.run_full_validation()
        >>> print(result.summary())
    """
    
    def __init__(
        self,
        fluid: Fluid,
        gpu: GPUSpecification = None,
        heat_sink: HeatSinkGeometry = None,
        benchmark_fluid: Fluid = None,
    ):
        """
        Initialize validator.
        
        Args:
            fluid: Cooling fluid to validate
            gpu: GPU specifications (default: NVIDIA H100)
            heat_sink: Heat sink geometry (default: optimized)
            benchmark_fluid: Reference fluid for comparison (default: FC-77)
        """
        self.fluid = fluid
        self.gpu = gpu or GPUSpecification()
        self.heat_sink = heat_sink or HeatSinkGeometry()
        self.benchmark_fluid = benchmark_fluid or FluorinertFC77()
        
        # Results cache
        self._mc_result: Optional[MonteCarloResult] = None
        self._benchmark_mc_result: Optional[MonteCarloResult] = None
    
    def run_monte_carlo(
        self,
        n_samples: int = 10_000,
        random_seed: int = 42,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.
        
        Args:
            n_samples: Number of samples (default: 10,000 from paper)
            random_seed: Random seed for reproducibility
            
        Returns:
            MonteCarloResult
        """
        config = MonteCarloConfig(
            n_samples=n_samples,
            random_seed=random_seed,
        )
        
        engine = MonteCarloEngine(
            fluid=self.fluid,
            gpu=self.gpu,
            heat_sink=self.heat_sink,
            config=config,
        )
        
        self._mc_result = engine.run()
        return self._mc_result
    
    def run_compliance_check(
        self,
        mc_result: MonteCarloResult = None
    ) -> ComplianceResult:
        """
        Run OCP compliance check.
        
        Args:
            mc_result: Monte Carlo result (runs simulation if None)
            
        Returns:
            ComplianceResult
        """
        if mc_result is None:
            mc_result = self._mc_result or self.run_monte_carlo()
        
        checker = ComplianceChecker()
        
        # Use mean values for compliance check
        values = {
            "junction_temperature_C": mc_result.t_junction_mean,
            "breakdown_voltage_kV": mc_result.bdv_mean,
            "volume_resistivity_ohm_cm": mc_result.resistivity_mean,
            "p5_life_years": mc_result.p5_life_mean,
        }
        
        # Compliance probabilities from MC
        probs = {
            "junction_temperature_C": mc_result.thermal_compliance_prob,
            "breakdown_voltage_kV": mc_result.bdv_compliance_prob,
            "volume_resistivity_ohm_cm": mc_result.resistivity_compliance_prob,
            "p5_life_years": mc_result.lifetime_compliance_prob,
        }
        
        return checker.check_all(values, probs)
    
    def run_economic_analysis(
        self,
        power_MW: float = 1.0,
    ) -> CostComparison:
        """
        Run economic comparison with benchmark fluid.
        
        Args:
            power_MW: Data center capacity in MW
            
        Returns:
            CostComparison
        """
        domestic = FluidCost(
            name=self.fluid.name,
            cost_per_liter_usd=self.fluid.properties.cost_per_liter_usd[0],
            is_domestic=True,
        )
        
        imported = FluidCost(
            name=self.benchmark_fluid.name,
            cost_per_liter_usd=self.benchmark_fluid.properties.cost_per_liter_usd[0],
            is_domestic=False,
            import_duty_percent=10.0,
        )
        
        analysis = EconomicAnalysis(domestic, imported)
        return analysis.compare_fluids(power_MW)
    
    def run_benchmark_comparison(
        self,
        n_samples: int = 10_000,
        random_seed: int = 42,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run comparison against benchmark fluid (Fluorinert FC-77).
        
        Args:
            n_samples: Monte Carlo samples
            random_seed: Random seed
            
        Returns:
            Dictionary of comparison metrics
        """
        # Run MC for test fluid if not already done
        if self._mc_result is None:
            self.run_monte_carlo(n_samples, random_seed)
        
        # Run MC for benchmark
        benchmark_config = MonteCarloConfig(n_samples=n_samples, random_seed=random_seed)
        benchmark_engine = MonteCarloEngine(
            fluid=self.benchmark_fluid,
            gpu=self.gpu,
            heat_sink=self.heat_sink,
            config=benchmark_config,
        )
        self._benchmark_mc_result = benchmark_engine.run()
        
        # Compare metrics
        test = self._mc_result
        bench = self._benchmark_mc_result
        
        comparisons = {
            "Junction Temperature (°C)": {
                "test": test.t_junction_mean,
                "benchmark": bench.t_junction_mean,
                "advantage": f"{bench.t_junction_mean - test.t_junction_mean:+.1f}°C (cooler)" 
                            if test.t_junction_mean < bench.t_junction_mean
                            else f"{test.t_junction_mean - bench.t_junction_mean:+.1f}°C (warmer)",
            },
            "Compliance Probability (%)": {
                "test": 100 * test.joint_compliance_prob,
                "benchmark": 100 * bench.joint_compliance_prob,
                "advantage": f"{100*(test.joint_compliance_prob - bench.joint_compliance_prob):+.1f}%",
            },
            "Cost per Liter ($)": {
                "test": self.fluid.properties.cost_per_liter_usd[0],
                "benchmark": self.benchmark_fluid.properties.cost_per_liter_usd[0],
                "advantage": f"{100*(1 - self.fluid.properties.cost_per_liter_usd[0] / 
                                    self.benchmark_fluid.properties.cost_per_liter_usd[0]):.0f}% savings",
            },
        }
        
        return comparisons
    
    def run_full_validation(
        self,
        n_samples: int = 10_000,
        random_seed: int = 42,
        include_economics: bool = True,
        include_benchmark: bool = True,
        power_MW: float = 1.0,
    ) -> ValidationResult:
        """
        Run complete validation workflow.
        
        Args:
            n_samples: Monte Carlo samples
            random_seed: Random seed
            include_economics: Include economic analysis
            include_benchmark: Include benchmark comparison
            power_MW: Data center capacity for economics
            
        Returns:
            ValidationResult with all analyses
        """
        # Monte Carlo simulation
        mc_result = self.run_monte_carlo(n_samples, random_seed)
        
        # Compliance check
        compliance = self.run_compliance_check(mc_result)
        
        # Economic analysis
        economics = None
        forex = None
        if include_economics:
            economics = self.run_economic_analysis(power_MW)
            
            # Forex projection
            from .economics import calculate_forex_savings
            forex = calculate_forex_savings(
                domestic_cost_per_liter=self.fluid.properties.cost_per_liter_usd[0],
                imported_cost_per_liter=self.benchmark_fluid.properties.cost_per_liter_usd[0],
            )
        
        # Benchmark comparison
        benchmark = None
        if include_benchmark:
            benchmark = self.run_benchmark_comparison(n_samples, random_seed)
        
        return ValidationResult(
            fluid_name=self.fluid.name,
            timestamp=datetime.now().isoformat(),
            monte_carlo=mc_result,
            compliance=compliance,
            economics=economics,
            forex_projection=forex,
            benchmark_comparison=benchmark,
        )


def validate_fluid(
    fluid: Fluid = None,
    n_samples: int = 10_000,
    include_economics: bool = True,
) -> ValidationResult:
    """
    Convenience function for quick validation.
    
    Args:
        fluid: Fluid to validate (default: Group III oil)
        n_samples: Monte Carlo samples
        include_economics: Include economic analysis
        
    Returns:
        ValidationResult
    """
    if fluid is None:
        fluid = GroupIIIOil()
    
    validator = ImmersionCoolingValidator(fluid)
    return validator.run_full_validation(
        n_samples=n_samples,
        include_economics=include_economics,
    )
