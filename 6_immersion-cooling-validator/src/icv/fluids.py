"""
Fluid definitions for immersion cooling applications.

This module defines base fluid classes and specific implementations for
Group III hydrocarbon base oils and Fluorinert FC-77.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class FluidProperties:
    """Container for fluid property distributions (mean, std)."""
    
    # Thermophysical properties
    density_kg_m3: Tuple[float, float] = (820.0, 5.0)  # (mean, std)
    viscosity_cSt_40C: Tuple[float, float] = (10.0, 0.5)
    thermal_conductivity_W_mK: Tuple[float, float] = (0.135, 0.005)
    specific_heat_J_kgK: Tuple[float, float] = (2100.0, 50.0)
    
    # Electrical properties
    breakdown_voltage_kV: Tuple[float, float] = (55.0, 3.0)
    volume_resistivity_ohm_cm: Tuple[float, float] = (1e14, 1e13)
    dielectric_constant: Tuple[float, float] = (2.15, 0.05)
    loss_tangent: Tuple[float, float] = (1.5e-4, 0.3e-4)
    
    # Thermal stability
    max_operating_temp_C: float = 150.0
    flash_point_C: float = 220.0
    pour_point_C: float = -20.0
    
    # Arrhenius parameters for resistivity
    activation_energy_eV: float = 0.69
    reference_temp_K: float = 298.15  # 25°C
    
    # Weibull lifetime parameters (calibrated to paper: P5=5.8, P50=6.6)
    weibull_shape_k: float = 8.8
    weibull_scale_lambda_years: float = 8.1
    
    # Economics
    cost_per_liter_usd: Tuple[float, float] = (10.0, 2.0)


@dataclass
class Fluid:
    """Base class for immersion cooling fluids."""
    
    name: str
    properties: FluidProperties
    description: str = ""
    manufacturer: str = ""
    
    def sample_properties(self, rng: np.random.Generator) -> Dict[str, float]:
        """
        Sample fluid properties from their distributions.
        
        Args:
            rng: NumPy random generator for reproducibility
            
        Returns:
            Dictionary of sampled property values
        """
        props = self.properties
        
        return {
            "density_kg_m3": rng.normal(*props.density_kg_m3),
            "viscosity_cSt_40C": max(1.0, rng.normal(*props.viscosity_cSt_40C)),
            "thermal_conductivity_W_mK": max(0.05, rng.normal(*props.thermal_conductivity_W_mK)),
            "specific_heat_J_kgK": max(500, rng.normal(*props.specific_heat_J_kgK)),
            "breakdown_voltage_kV": max(10, rng.normal(*props.breakdown_voltage_kV)),
            "volume_resistivity_ohm_cm": max(1e10, rng.lognormal(
                np.log(props.volume_resistivity_ohm_cm[0]),
                props.volume_resistivity_ohm_cm[1] / props.volume_resistivity_ohm_cm[0]
            )),
            "dielectric_constant": max(1.5, rng.normal(*props.dielectric_constant)),
            "loss_tangent": max(1e-5, rng.normal(*props.loss_tangent)),
            "cost_per_liter_usd": max(1, rng.normal(*props.cost_per_liter_usd)),
        }
    
    def get_viscosity_at_temp(self, temp_C: float, viscosity_40C: float) -> float:
        """
        Calculate viscosity at a given temperature using Walther equation.
        
        Args:
            temp_C: Temperature in Celsius
            viscosity_40C: Kinematic viscosity at 40°C in cSt
            
        Returns:
            Kinematic viscosity at temp_C in cSt
        """
        # Walther equation: log(log(v + 0.7)) = A - B * log(T)
        # Simplified approximation using viscosity index
        T1, T2 = 40.0, 100.0
        v1 = viscosity_40C
        v2 = viscosity_40C * 0.35  # Approximate VI=120 for Group III
        
        # Linear interpolation in log-log space
        if temp_C <= T1:
            m = (np.log(v1) - np.log(v2)) / (T2 - T1)
            return v1 * np.exp(m * (T1 - temp_C))
        elif temp_C >= T2:
            m = (np.log(v1) - np.log(v2)) / (T2 - T1)
            return v2 * np.exp(m * (T2 - temp_C))
        else:
            # Log-linear interpolation
            log_v = np.log(v1) + (np.log(v2) - np.log(v1)) * (temp_C - T1) / (T2 - T1)
            return np.exp(log_v)
    
    def get_density_at_temp(self, temp_C: float, density_15C: float) -> float:
        """
        Calculate density at temperature using thermal expansion.
        
        Args:
            temp_C: Temperature in Celsius
            density_15C: Density at 15°C in kg/m³
            
        Returns:
            Density at temp_C in kg/m³
        """
        # Thermal expansion coefficient ~0.00065 /°C for hydrocarbons
        alpha = 0.00065
        return density_15C * (1 - alpha * (temp_C - 15.0))
    
    def __repr__(self) -> str:
        return f"Fluid(name='{self.name}', cost=${self.properties.cost_per_liter_usd[0]:.1f}/L)"


def GroupIIIOil(
    name: str = "Group III Base Oil",
    viscosity_grade: str = "4cSt",
    custom_properties: Optional[Dict] = None
) -> Fluid:
    """
    Create a Group III hydrocarbon base oil fluid.
    
    Group III oils are produced through severe hydrocracking and isomerization.
    They offer excellent dielectric properties and thermal stability.
    
    Args:
        name: Fluid name/identifier
        viscosity_grade: "4cSt", "6cSt", or "8cSt"
        custom_properties: Optional dict to override default properties
        
    Returns:
        Configured Fluid instance
    """
    # Base properties for different viscosity grades
    grade_properties = {
        "4cSt": {
            "viscosity_cSt_40C": (4.2, 0.2),
            "density_kg_m3": (815.0, 5.0),
            "pour_point_C": -18.0,
        },
        "6cSt": {
            "viscosity_cSt_40C": (6.0, 0.3),
            "density_kg_m3": (825.0, 5.0),
            "pour_point_C": -15.0,
        },
        "8cSt": {
            "viscosity_cSt_40C": (8.0, 0.4),
            "density_kg_m3": (835.0, 5.0),
            "pour_point_C": -12.0,
        },
    }
    
    grade_props = grade_properties.get(viscosity_grade, grade_properties["4cSt"])
    
    props = FluidProperties(
        # Thermophysical (from paper and API Group III specs)
        density_kg_m3=grade_props["density_kg_m3"],
        viscosity_cSt_40C=grade_props["viscosity_cSt_40C"],
        thermal_conductivity_W_mK=(0.135, 0.005),
        specific_heat_J_kgK=(2100.0, 50.0),
        
        # Electrical (from paper validation results)
        breakdown_voltage_kV=(55.0, 3.0),
        volume_resistivity_ohm_cm=(1e14, 1e13),
        dielectric_constant=(2.15, 0.05),
        loss_tangent=(1.5e-4, 0.3e-4),
        
        # Thermal stability
        max_operating_temp_C=150.0,
        flash_point_C=220.0,
        pour_point_C=grade_props["pour_point_C"],
        
        # Arrhenius (from Oommen 2002)
        activation_energy_eV=0.69,
        reference_temp_K=298.15,
        
        # Weibull lifetime (calibrated to paper: P5=5.8 years)
        weibull_shape_k=8.8,
        weibull_scale_lambda_years=8.1,
        
        # Economics (domestic production)
        cost_per_liter_usd=(10.0, 2.0),
    )
    
    # Apply custom overrides
    if custom_properties:
        for key, value in custom_properties.items():
            if hasattr(props, key):
                setattr(props, key, value)
    
    return Fluid(
        name=name,
        properties=props,
        description=f"Group III hydrocarbon base oil ({viscosity_grade} grade) - "
                   f"Domestically produced through severe hydrocracking",
        manufacturer="Indian Refineries (HPCL/IOCL/BPCL)",
    )


def FluorinertFC77() -> Fluid:
    """
    Create a 3M Fluorinert FC-77 fluid (benchmark comparison).
    
    Fluorinert FC-77 is a widely-used perfluorocarbon fluid for
    electronics cooling with excellent dielectric properties but
    high cost and environmental concerns.
    
    Returns:
        Configured Fluid instance
    """
    props = FluidProperties(
        # Thermophysical (from 3M datasheet)
        density_kg_m3=(1780.0, 10.0),
        viscosity_cSt_40C=(1.4, 0.1),
        thermal_conductivity_W_mK=(0.063, 0.003),
        specific_heat_J_kgK=(1050.0, 30.0),
        
        # Electrical
        breakdown_voltage_kV=(40.0, 2.0),
        volume_resistivity_ohm_cm=(1e15, 1e14),
        dielectric_constant=(1.86, 0.02),
        loss_tangent=(1e-4, 0.2e-4),
        
        # Thermal stability
        max_operating_temp_C=200.0,
        flash_point_C=float('inf'),  # Non-flammable
        pour_point_C=-95.0,
        
        # Lifetime (longer due to chemical inertness)
        weibull_shape_k=10.0,
        weibull_scale_lambda_years=10.0,
        
        # Economics (imported, expensive)
        cost_per_liter_usd=(400.0, 50.0),
    )
    
    return Fluid(
        name="Fluorinert FC-77",
        properties=props,
        description="3M Fluorinert FC-77 perfluorocarbon electronic cooling fluid",
        manufacturer="3M Company",
    )


def create_fluid(
    fluid_type: str,
    **kwargs
) -> Fluid:
    """
    Factory function to create fluid instances.
    
    Args:
        fluid_type: One of "group_iii", "fluorinert_fc77", "custom"
        **kwargs: Additional arguments passed to fluid constructor
        
    Returns:
        Configured Fluid instance
    """
    fluid_map = {
        "group_iii": GroupIIIOil,
        "group_iii_4cst": lambda **kw: GroupIIIOil(viscosity_grade="4cSt", **kw),
        "group_iii_6cst": lambda **kw: GroupIIIOil(viscosity_grade="6cSt", **kw),
        "group_iii_8cst": lambda **kw: GroupIIIOil(viscosity_grade="8cSt", **kw),
        "fluorinert_fc77": FluorinertFC77,
        "fc77": FluorinertFC77,
    }
    
    if fluid_type.lower() not in fluid_map:
        available = ", ".join(fluid_map.keys())
        raise ValueError(f"Unknown fluid type: {fluid_type}. Available: {available}")
    
    creator = fluid_map[fluid_type.lower()]
    return creator(**kwargs) if callable(creator) else creator
