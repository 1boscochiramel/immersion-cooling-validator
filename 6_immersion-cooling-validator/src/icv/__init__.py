"""
Immersion Cooling Validator (ICV)
=================================

A Monte Carlo validation framework for evaluating immersion cooling fluids
against Open Compute Project (OCP) specifications.

Based on: Chiramel, B. (2024). "Monte Carlo Validation Framework for Group III
Hydrocarbon-Based Single-Phase Immersion Cooling Fluid in AI Data Center Applications"

Author: Bosco Chiramel
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Bosco Chiramel"
__email__ = "bosco8b4824@gmail.com"

from .fluids import (
    Fluid,
    GroupIIIOil,
    FluorinertFC77,
    create_fluid,
)

from .properties import (
    ThermophysicalProperties,
    ElectricalProperties,
    calculate_viscosity,
    calculate_thermal_conductivity,
    calculate_density,
    calculate_specific_heat,
)

from .thermal import (
    ThermalModel,
    ThermalResistanceNetwork,
    calculate_junction_temperature,
    calculate_convective_htc,
    GPUSpecification,
    HeatSinkGeometry,
)

from .electrical import (
    ElectricalModel,
    calculate_breakdown_voltage,
    calculate_resistivity_at_temp,
    arrhenius_resistivity,
)

from .lifetime import (
    LifetimeModel,
    weibull_survival,
    weibull_percentile,
    calculate_service_life,
)

from .monte_carlo import (
    MonteCarloEngine,
    MonteCarloResult,
    MonteCarloConfig,
    run_simulation,
)

from .compliance import (
    OCPSpecification,
    ComplianceChecker,
    ComplianceResult,
    check_ocp_compliance,
)

from .economics import (
    EconomicAnalysis,
    CostComparison,
    FluidCost,
    DataCenterConfig,
    ForexSavingsProjection,
    calculate_forex_savings,
)

from .validator import (
    ImmersionCoolingValidator,
    ValidationResult,
    validate_fluid,
)

from .sensitivity import (
    SobolIndices,
    SensitivityResult,
    SensitivityAnalyzer,
    compute_correlations,
    rank_parameters_by_influence,
    run_sensitivity_analysis,
)

from .report import (
    generate_html_report,
    save_html_report,
    ReportConfig,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Fluids
    "Fluid",
    "GroupIIIOil",
    "FluorinertFC77",
    "create_fluid",
    
    # Properties
    "ThermophysicalProperties",
    "ElectricalProperties",
    "calculate_viscosity",
    "calculate_thermal_conductivity",
    "calculate_density",
    "calculate_specific_heat",
    
    # Thermal
    "ThermalModel",
    "ThermalResistanceNetwork",
    "calculate_junction_temperature",
    "calculate_convective_htc",
    "GPUSpecification",
    "HeatSinkGeometry",
    
    # Electrical
    "ElectricalModel",
    "calculate_breakdown_voltage",
    "calculate_resistivity_at_temp",
    "arrhenius_resistivity",
    
    # Lifetime
    "LifetimeModel",
    "weibull_survival",
    "weibull_percentile",
    "calculate_service_life",
    
    # Monte Carlo
    "MonteCarloEngine",
    "MonteCarloResult",
    "MonteCarloConfig",
    "run_simulation",
    
    # Compliance
    "OCPSpecification",
    "ComplianceChecker",
    "ComplianceResult",
    "check_ocp_compliance",
    
    # Economics
    "EconomicAnalysis",
    "CostComparison",
    "FluidCost",
    "DataCenterConfig",
    "ForexSavingsProjection",
    "calculate_forex_savings",
    
    # Main Validator
    "ImmersionCoolingValidator",
    "ValidationResult",
    "validate_fluid",
    
    # Sensitivity Analysis
    "SobolIndices",
    "SensitivityResult",
    "SensitivityAnalyzer",
    "compute_correlations",
    "rank_parameters_by_influence",
    "run_sensitivity_analysis",
    
    # Reports
    "generate_html_report",
    "save_html_report",
    "ReportConfig",
]
