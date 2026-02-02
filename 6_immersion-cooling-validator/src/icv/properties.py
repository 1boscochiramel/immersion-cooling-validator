"""
Thermophysical and electrical property models for immersion cooling fluids.

This module implements temperature-dependent property correlations based on
API standards and published literature for hydrocarbon fluids.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class ThermophysicalProperties:
    """
    Container for thermophysical properties at a specific temperature.
    """
    temperature_C: float
    density_kg_m3: float
    viscosity_cSt: float
    viscosity_Pa_s: float
    thermal_conductivity_W_mK: float
    specific_heat_J_kgK: float
    prandtl_number: float
    thermal_diffusivity_m2_s: float
    
    @property
    def viscosity_cP(self) -> float:
        """Dynamic viscosity in centipoise."""
        return self.viscosity_Pa_s * 1000
    
    def __repr__(self) -> str:
        return (f"ThermophysicalProperties(T={self.temperature_C}°C, "
                f"ρ={self.density_kg_m3:.1f} kg/m³, "
                f"μ={self.viscosity_cSt:.2f} cSt, "
                f"k={self.thermal_conductivity_W_mK:.4f} W/m·K)")


@dataclass
class ElectricalProperties:
    """
    Container for electrical properties at a specific temperature.
    """
    temperature_C: float
    breakdown_voltage_kV: float
    volume_resistivity_ohm_cm: float
    dielectric_constant: float
    loss_tangent: float
    
    @property
    def dielectric_strength_kV_mm(self) -> float:
        """Dielectric strength assuming 2.5mm gap (standard test)."""
        return self.breakdown_voltage_kV / 2.5
    
    def __repr__(self) -> str:
        return (f"ElectricalProperties(T={self.temperature_C}°C, "
                f"BDV={self.breakdown_voltage_kV:.1f} kV, "
                f"ρ={self.volume_resistivity_ohm_cm:.2e} Ω·cm)")


def calculate_density(
    temp_C: float,
    density_15C: float = 825.0,
    thermal_expansion_coeff: float = 0.00065
) -> float:
    """
    Calculate fluid density at temperature using linear thermal expansion.
    
    Args:
        temp_C: Temperature in Celsius
        density_15C: Density at 15°C reference in kg/m³
        thermal_expansion_coeff: Volumetric thermal expansion coefficient (1/°C)
        
    Returns:
        Density at specified temperature in kg/m³
    """
    return density_15C * (1 - thermal_expansion_coeff * (temp_C - 15.0))


def calculate_viscosity(
    temp_C: float,
    viscosity_40C: float = 10.0,
    viscosity_100C: Optional[float] = None,
    viscosity_index: float = 120.0
) -> float:
    """
    Calculate kinematic viscosity at temperature using Walther equation.
    
    The Walther equation is the industry standard for hydrocarbon viscosity:
    log(log(ν + 0.7)) = A - B·log(T)
    
    Args:
        temp_C: Temperature in Celsius
        viscosity_40C: Kinematic viscosity at 40°C in cSt
        viscosity_100C: Kinematic viscosity at 100°C in cSt (optional)
        viscosity_index: Viscosity index if viscosity_100C not provided
        
    Returns:
        Kinematic viscosity at temp_C in cSt
    """
    # Calculate viscosity at 100°C from VI if not provided
    if viscosity_100C is None:
        # Approximate relationship for VI calculation
        # Higher VI means less viscosity change with temperature
        viscosity_100C = viscosity_40C * (0.1 + 0.4 * (viscosity_index / 150.0))
    
    # Walther equation parameters
    T1, T2 = 313.15, 373.15  # 40°C and 100°C in Kelvin
    v1, v2 = viscosity_40C, viscosity_100C
    
    # Calculate Walther constants A and B
    def walther_func(v):
        return np.log10(np.log10(v + 0.7))
    
    W1, W2 = walther_func(v1), walther_func(v2)
    B = (W1 - W2) / (np.log10(T2) - np.log10(T1))
    A = W1 + B * np.log10(T1)
    
    # Calculate viscosity at target temperature
    T = temp_C + 273.15
    W = A - B * np.log10(T)
    viscosity = 10**(10**W) - 0.7
    
    return max(0.5, viscosity)  # Minimum physical viscosity


def calculate_thermal_conductivity(
    temp_C: float,
    k_25C: float = 0.135,
    temp_coefficient: float = -0.0002
) -> float:
    """
    Calculate thermal conductivity at temperature.
    
    Hydrocarbon thermal conductivity decreases slightly with temperature.
    
    Args:
        temp_C: Temperature in Celsius
        k_25C: Thermal conductivity at 25°C in W/(m·K)
        temp_coefficient: Temperature coefficient (W/(m·K·°C))
        
    Returns:
        Thermal conductivity at temp_C in W/(m·K)
    """
    return k_25C + temp_coefficient * (temp_C - 25.0)


def calculate_specific_heat(
    temp_C: float,
    cp_25C: float = 2100.0,
    temp_coefficient: float = 3.0
) -> float:
    """
    Calculate specific heat capacity at temperature.
    
    Hydrocarbon specific heat increases slightly with temperature.
    
    Args:
        temp_C: Temperature in Celsius
        cp_25C: Specific heat at 25°C in J/(kg·K)
        temp_coefficient: Temperature coefficient (J/(kg·K·°C))
        
    Returns:
        Specific heat at temp_C in J/(kg·K)
    """
    return cp_25C + temp_coefficient * (temp_C - 25.0)


def calculate_prandtl_number(
    viscosity_Pa_s: float,
    specific_heat_J_kgK: float,
    thermal_conductivity_W_mK: float
) -> float:
    """
    Calculate Prandtl number.
    
    Pr = μ·Cp / k
    
    Args:
        viscosity_Pa_s: Dynamic viscosity in Pa·s
        specific_heat_J_kgK: Specific heat in J/(kg·K)
        thermal_conductivity_W_mK: Thermal conductivity in W/(m·K)
        
    Returns:
        Prandtl number (dimensionless)
    """
    return viscosity_Pa_s * specific_heat_J_kgK / thermal_conductivity_W_mK


def get_thermophysical_properties(
    temp_C: float,
    density_15C: float = 825.0,
    viscosity_40C: float = 10.0,
    k_25C: float = 0.135,
    cp_25C: float = 2100.0,
    viscosity_index: float = 120.0
) -> ThermophysicalProperties:
    """
    Calculate all thermophysical properties at a given temperature.
    
    Args:
        temp_C: Temperature in Celsius
        density_15C: Reference density at 15°C in kg/m³
        viscosity_40C: Reference viscosity at 40°C in cSt
        k_25C: Reference thermal conductivity at 25°C in W/(m·K)
        cp_25C: Reference specific heat at 25°C in J/(kg·K)
        viscosity_index: Viscosity index for temperature extrapolation
        
    Returns:
        ThermophysicalProperties dataclass with all properties
    """
    density = calculate_density(temp_C, density_15C)
    viscosity_cSt = calculate_viscosity(temp_C, viscosity_40C, 
                                         viscosity_index=viscosity_index)
    # Convert kinematic to dynamic viscosity
    # μ [Pa·s] = ν [cSt] × ρ [kg/m³] × 10⁻⁶
    viscosity_Pa_s = viscosity_cSt * density * 1e-6
    
    k = calculate_thermal_conductivity(temp_C, k_25C)
    cp = calculate_specific_heat(temp_C, cp_25C)
    
    Pr = calculate_prandtl_number(viscosity_Pa_s, cp, k)
    alpha = k / (density * cp)  # Thermal diffusivity
    
    return ThermophysicalProperties(
        temperature_C=temp_C,
        density_kg_m3=density,
        viscosity_cSt=viscosity_cSt,
        viscosity_Pa_s=viscosity_Pa_s,
        thermal_conductivity_W_mK=k,
        specific_heat_J_kgK=cp,
        prandtl_number=Pr,
        thermal_diffusivity_m2_s=alpha,
    )


# Physical constants
BOLTZMANN_CONSTANT_eV_K = 8.617e-5  # Boltzmann constant in eV/K


def arrhenius_resistivity(
    temp_K: float,
    rho_ref: float,
    T_ref: float = 298.15,
    activation_energy_eV: float = 0.69
) -> float:
    """
    Calculate volume resistivity at temperature using Arrhenius model.
    
    ρ(T) = ρ_ref × exp[Ea/kB × (1/T - 1/T_ref)]
    
    From: Oommen, T. (2002). IEEE Electrical Insulation Magazine.
    
    Args:
        temp_K: Temperature in Kelvin
        rho_ref: Reference resistivity at T_ref in Ω·cm
        T_ref: Reference temperature in Kelvin (default 25°C)
        activation_energy_eV: Activation energy in eV (0.69 for Group III)
        
    Returns:
        Volume resistivity at temp_K in Ω·cm
    """
    kB = BOLTZMANN_CONSTANT_eV_K
    Ea = activation_energy_eV
    
    exponent = (Ea / kB) * (1/temp_K - 1/T_ref)
    return rho_ref * np.exp(exponent)


def calculate_breakdown_voltage(
    temp_C: float,
    bdv_25C: float = 55.0,
    temp_coefficient: float = -0.1,
    moisture_ppm: float = 50.0
) -> float:
    """
    Calculate breakdown voltage at temperature with moisture correction.
    
    BDV decreases with temperature and moisture content.
    
    Args:
        temp_C: Temperature in Celsius
        bdv_25C: Breakdown voltage at 25°C in kV
        temp_coefficient: Temperature coefficient (kV/°C)
        moisture_ppm: Moisture content in ppm
        
    Returns:
        Breakdown voltage at conditions in kV
    """
    # Temperature effect
    bdv_temp = bdv_25C + temp_coefficient * (temp_C - 25.0)
    
    # Moisture correction (significant above 100 ppm)
    if moisture_ppm > 100:
        moisture_factor = 1 - 0.003 * (moisture_ppm - 100)
        bdv_temp *= max(0.5, moisture_factor)
    
    return max(10.0, bdv_temp)


def calculate_resistivity_at_temp(
    temp_C: float,
    rho_25C: float = 1e14,
    activation_energy_eV: float = 0.69
) -> float:
    """
    Calculate volume resistivity at temperature.
    
    Convenience wrapper for arrhenius_resistivity.
    
    Args:
        temp_C: Temperature in Celsius
        rho_25C: Resistivity at 25°C in Ω·cm
        activation_energy_eV: Activation energy in eV
        
    Returns:
        Volume resistivity at temp_C in Ω·cm
    """
    temp_K = temp_C + 273.15
    return arrhenius_resistivity(temp_K, rho_25C, 298.15, activation_energy_eV)


def get_electrical_properties(
    temp_C: float,
    bdv_25C: float = 55.0,
    rho_25C: float = 1e14,
    dielectric_constant: float = 2.15,
    loss_tangent: float = 1.5e-4,
    activation_energy_eV: float = 0.69,
    moisture_ppm: float = 50.0
) -> ElectricalProperties:
    """
    Calculate all electrical properties at a given temperature.
    
    Args:
        temp_C: Temperature in Celsius
        bdv_25C: Breakdown voltage at 25°C in kV
        rho_25C: Resistivity at 25°C in Ω·cm
        dielectric_constant: Relative permittivity (weakly temp-dependent)
        loss_tangent: Dielectric loss tangent
        activation_energy_eV: Arrhenius activation energy in eV
        moisture_ppm: Moisture content in ppm
        
    Returns:
        ElectricalProperties dataclass with all properties
    """
    bdv = calculate_breakdown_voltage(temp_C, bdv_25C, moisture_ppm=moisture_ppm)
    rho = calculate_resistivity_at_temp(temp_C, rho_25C, activation_energy_eV)
    
    # Dielectric constant slightly decreases with temperature
    dk_temp = dielectric_constant * (1 - 0.001 * (temp_C - 25.0))
    
    # Loss tangent increases with temperature
    tan_delta_temp = loss_tangent * (1 + 0.01 * (temp_C - 25.0))
    
    return ElectricalProperties(
        temperature_C=temp_C,
        breakdown_voltage_kV=bdv,
        volume_resistivity_ohm_cm=rho,
        dielectric_constant=dk_temp,
        loss_tangent=tan_delta_temp,
    )
