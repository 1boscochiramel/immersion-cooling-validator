"""
OCP compliance checking for immersion cooling fluids.

This module implements compliance verification against Open Compute Project
specifications for single-phase immersion cooling fluids.

Reference: Open Compute Project (2024). OCP Advanced Cooling Solutions:
           Immersion Cooling Guidelines.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class ComplianceStatus(Enum):
    """Compliance status enumeration."""
    PASS = "PASS"
    FAIL = "FAIL"
    MARGINAL = "MARGINAL"  # Within 10% of limit
    UNKNOWN = "UNKNOWN"


@dataclass
class OCPRequirement:
    """Single OCP requirement specification."""
    
    name: str
    description: str
    
    # Requirement type
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    # Units
    unit: str = ""
    
    # Criticality
    is_critical: bool = True
    
    def check(self, value: float) -> ComplianceStatus:
        """
        Check if value meets requirement.
        
        Args:
            value: Measured/simulated value
            
        Returns:
            ComplianceStatus
        """
        if self.min_value is not None and value < self.min_value:
            # Check if marginal (within 10%)
            margin = abs(value - self.min_value) / self.min_value
            if margin < 0.1:
                return ComplianceStatus.MARGINAL
            return ComplianceStatus.FAIL
        
        if self.max_value is not None and value > self.max_value:
            margin = abs(value - self.max_value) / self.max_value
            if margin < 0.1:
                return ComplianceStatus.MARGINAL
            return ComplianceStatus.FAIL
        
        return ComplianceStatus.PASS
    
    def get_margin(self, value: float) -> float:
        """
        Calculate margin relative to limit.
        
        Positive margin means compliant with headroom.
        Negative margin means non-compliant.
        
        Args:
            value: Measured value
            
        Returns:
            Margin (positive = good)
        """
        if self.min_value is not None:
            return value - self.min_value
        if self.max_value is not None:
            return self.max_value - value
        return float('inf')


@dataclass
class OCPSpecification:
    """
    Complete OCP specification for immersion cooling fluids.
    
    Based on Table 2 from the paper and OCP guidelines.
    """
    
    requirements: Dict[str, OCPRequirement] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default OCP requirements."""
        if not self.requirements:
            self.requirements = {
                # Thermal requirements
                "junction_temperature_C": OCPRequirement(
                    name="Junction Temperature",
                    description="Maximum chip junction temperature",
                    max_value=88.0,  # Thermal throttle limit
                    unit="°C",
                    is_critical=True,
                ),
                
                # Electrical requirements
                "breakdown_voltage_kV": OCPRequirement(
                    name="Breakdown Voltage",
                    description="Minimum dielectric breakdown voltage",
                    min_value=45.0,
                    unit="kV",
                    is_critical=True,
                ),
                "volume_resistivity_ohm_cm": OCPRequirement(
                    name="Volume Resistivity",
                    description="Minimum volume resistivity at 90°C",
                    min_value=1e11,
                    unit="Ω·cm",
                    is_critical=True,
                ),
                "dielectric_constant": OCPRequirement(
                    name="Dielectric Constant",
                    description="Relative permittivity range",
                    min_value=1.8,
                    max_value=2.5,
                    unit="",
                    is_critical=False,
                ),
                "loss_tangent": OCPRequirement(
                    name="Loss Tangent",
                    description="Maximum dielectric loss tangent",
                    max_value=5e-4,
                    unit="",
                    is_critical=False,
                ),
                
                # Lifetime requirement
                "p5_life_years": OCPRequirement(
                    name="P5 Service Life",
                    description="Minimum 5th percentile service life",
                    min_value=5.0,
                    unit="years",
                    is_critical=True,
                ),
                
                # Physical requirements
                "flash_point_C": OCPRequirement(
                    name="Flash Point",
                    description="Minimum flash point for safety",
                    min_value=150.0,
                    unit="°C",
                    is_critical=True,
                ),
                "pour_point_C": OCPRequirement(
                    name="Pour Point",
                    description="Maximum pour point for pumpability",
                    max_value=-10.0,
                    unit="°C",
                    is_critical=False,
                ),
                "viscosity_cSt_40C": OCPRequirement(
                    name="Kinematic Viscosity",
                    description="Viscosity range at 40°C",
                    min_value=2.0,
                    max_value=20.0,
                    unit="cSt",
                    is_critical=False,
                ),
            }
    
    def get_critical_requirements(self) -> Dict[str, OCPRequirement]:
        """Get only critical requirements."""
        return {k: v for k, v in self.requirements.items() if v.is_critical}


@dataclass
class RequirementResult:
    """Result for a single requirement check."""
    
    requirement: OCPRequirement
    value: float
    status: ComplianceStatus
    margin: float
    compliance_probability: float = 1.0  # From Monte Carlo
    
    @property
    def margin_percent(self) -> float:
        """Margin as percentage of limit."""
        if self.requirement.min_value:
            return 100 * self.margin / self.requirement.min_value
        if self.requirement.max_value:
            return 100 * self.margin / self.requirement.max_value
        return 0.0


@dataclass
class ComplianceResult:
    """
    Complete compliance assessment result.
    """
    
    # Individual results
    results: Dict[str, RequirementResult]
    
    # Overall compliance
    all_pass: bool
    critical_pass: bool
    joint_compliance_probability: float
    
    # Summary statistics
    n_requirements: int
    n_pass: int
    n_fail: int
    n_marginal: int
    
    def summary(self) -> str:
        """Generate compliance summary."""
        lines = [
            "OCP COMPLIANCE ASSESSMENT",
            "=" * 50,
            f"Overall Status: {'✅ PASS' if self.all_pass else '❌ FAIL'}",
            f"Critical Requirements: {'✅ PASS' if self.critical_pass else '❌ FAIL'}",
            f"Joint Compliance Probability: {100*self.joint_compliance_probability:.2f}%",
            "",
            f"Results: {self.n_pass} PASS, {self.n_marginal} MARGINAL, {self.n_fail} FAIL",
            "",
            "DETAILED RESULTS:",
            "-" * 50,
        ]
        
        for name, result in self.results.items():
            status_emoji = {
                ComplianceStatus.PASS: "✅",
                ComplianceStatus.FAIL: "❌",
                ComplianceStatus.MARGINAL: "⚠️",
            }.get(result.status, "❓")
            
            req = result.requirement
            limit_str = ""
            if req.min_value is not None:
                limit_str = f">{req.min_value}"
            if req.max_value is not None:
                if limit_str:
                    limit_str += f", <{req.max_value}"
                else:
                    limit_str = f"<{req.max_value}"
            
            lines.append(
                f"{status_emoji} {req.name}: {result.value:.4g} {req.unit} "
                f"(limit: {limit_str} {req.unit}, margin: {result.margin_percent:+.1f}%)"
            )
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "all_pass": self.all_pass,
            "critical_pass": self.critical_pass,
            "joint_compliance_probability": self.joint_compliance_probability,
            "n_requirements": self.n_requirements,
            "n_pass": self.n_pass,
            "n_fail": self.n_fail,
            "n_marginal": self.n_marginal,
            "results": {
                name: {
                    "value": r.value,
                    "status": r.status.value,
                    "margin": r.margin,
                    "margin_percent": r.margin_percent,
                    "compliance_probability": r.compliance_probability,
                }
                for name, r in self.results.items()
            }
        }


class ComplianceChecker:
    """
    OCP compliance checker for immersion cooling fluids.
    """
    
    def __init__(self, spec: OCPSpecification = None):
        """
        Initialize compliance checker.
        
        Args:
            spec: OCP specification (default: standard OCP requirements)
        """
        self.spec = spec or OCPSpecification()
    
    def check_single(
        self,
        requirement_name: str,
        value: float,
        compliance_prob: float = 1.0
    ) -> RequirementResult:
        """
        Check compliance for a single requirement.
        
        Args:
            requirement_name: Name of requirement
            value: Measured/simulated value
            compliance_prob: Compliance probability from Monte Carlo
            
        Returns:
            RequirementResult
        """
        if requirement_name not in self.spec.requirements:
            raise ValueError(f"Unknown requirement: {requirement_name}")
        
        req = self.spec.requirements[requirement_name]
        status = req.check(value)
        margin = req.get_margin(value)
        
        return RequirementResult(
            requirement=req,
            value=value,
            status=status,
            margin=margin,
            compliance_probability=compliance_prob,
        )
    
    def check_all(
        self,
        values: Dict[str, float],
        compliance_probs: Dict[str, float] = None
    ) -> ComplianceResult:
        """
        Check compliance for all requirements.
        
        Args:
            values: Dictionary mapping requirement names to values
            compliance_probs: Optional compliance probabilities from MC
            
        Returns:
            ComplianceResult
        """
        if compliance_probs is None:
            compliance_probs = {}
        
        results = {}
        for name, req in self.spec.requirements.items():
            if name in values:
                prob = compliance_probs.get(name, 1.0)
                results[name] = self.check_single(name, values[name], prob)
        
        # Calculate summary statistics
        n_pass = sum(1 for r in results.values() if r.status == ComplianceStatus.PASS)
        n_fail = sum(1 for r in results.values() if r.status == ComplianceStatus.FAIL)
        n_marginal = sum(1 for r in results.values() if r.status == ComplianceStatus.MARGINAL)
        
        # MARGINAL counts as failing for all_pass determination
        all_pass = (n_fail == 0) and (n_marginal == 0)
        
        # Check critical requirements
        critical_pass = all(
            results[name].status != ComplianceStatus.FAIL
            for name, req in self.spec.requirements.items()
            if req.is_critical and name in results
        )
        
        # Joint compliance probability (product of individual probabilities)
        joint_prob = 1.0
        for r in results.values():
            joint_prob *= r.compliance_probability
        
        return ComplianceResult(
            results=results,
            all_pass=all_pass,
            critical_pass=critical_pass,
            joint_compliance_probability=joint_prob,
            n_requirements=len(results),
            n_pass=n_pass,
            n_fail=n_fail,
            n_marginal=n_marginal,
        )


def check_ocp_compliance(
    junction_temp_C: float,
    breakdown_voltage_kV: float,
    volume_resistivity_ohm_cm: float,
    p5_life_years: float,
    dielectric_constant: float = 2.15,
    loss_tangent: float = 1.5e-4,
    compliance_probs: Dict[str, float] = None,
) -> ComplianceResult:
    """
    Convenience function to check OCP compliance.
    
    Args:
        junction_temp_C: Junction temperature
        breakdown_voltage_kV: Breakdown voltage
        volume_resistivity_ohm_cm: Volume resistivity
        p5_life_years: P5 service life
        dielectric_constant: Dielectric constant
        loss_tangent: Loss tangent
        compliance_probs: Optional MC compliance probabilities
        
    Returns:
        ComplianceResult
    """
    checker = ComplianceChecker()
    
    values = {
        "junction_temperature_C": junction_temp_C,
        "breakdown_voltage_kV": breakdown_voltage_kV,
        "volume_resistivity_ohm_cm": volume_resistivity_ohm_cm,
        "p5_life_years": p5_life_years,
        "dielectric_constant": dielectric_constant,
        "loss_tangent": loss_tangent,
    }
    
    return checker.check_all(values, compliance_probs)
