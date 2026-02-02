"""
Electrical property models for dielectric fluids.

This module implements:
- Arrhenius temperature dependence of volume resistivity (Eq. 3)
- Breakdown voltage modeling
- Dielectric property calculations
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


# Physical constants
BOLTZMANN_CONSTANT_eV_K = 8.617e-5  # eV/K


@dataclass
class ElectricalModel:
    """
    Electrical property model for dielectric cooling fluids.
    
    Implements temperature-dependent resistivity using Arrhenius model
    and breakdown voltage correlations.
    """
    
    # Reference properties at 25°C
    breakdown_voltage_25C_kV: float = 55.0
    volume_resistivity_25C_ohm_cm: float = 1e14
    dielectric_constant_25C: float = 2.15
    loss_tangent_25C: float = 1.5e-4
    
    # Arrhenius parameters (from Oommen 2002)
    activation_energy_eV: float = 0.69
    reference_temp_K: float = 298.15
    
    # Temperature coefficients
    bdv_temp_coeff_kV_per_C: float = -0.1  # BDV decreases with temp
    dk_temp_coeff_per_C: float = -0.001    # Dk slightly decreases
    
    # Aging factors
    aging_rate_per_year: float = 0.02  # 2% degradation per year
    
    def resistivity_at_temp(self, temp_C: float) -> float:
        """
        Calculate volume resistivity at temperature using Arrhenius model.
        
        ρ(T) = ρ_ref × exp[Ea/kB × (1/T - 1/T_ref)]  (Equation 3)
        
        Args:
            temp_C: Temperature in Celsius
            
        Returns:
            Volume resistivity in Ω·cm
        """
        temp_K = temp_C + 273.15
        T_ref = self.reference_temp_K
        Ea = self.activation_energy_eV
        kB = BOLTZMANN_CONSTANT_eV_K
        
        exponent = (Ea / kB) * (1/temp_K - 1/T_ref)
        return self.volume_resistivity_25C_ohm_cm * np.exp(exponent)
    
    def breakdown_voltage_at_temp(
        self,
        temp_C: float,
        moisture_ppm: float = 50.0,
        years_aged: float = 0.0
    ) -> float:
        """
        Calculate breakdown voltage at temperature with corrections.
        
        Args:
            temp_C: Temperature in Celsius
            moisture_ppm: Moisture content in ppm
            years_aged: Years of thermal aging
            
        Returns:
            Breakdown voltage in kV
        """
        # Base temperature effect
        bdv = self.breakdown_voltage_25C_kV + self.bdv_temp_coeff_kV_per_C * (temp_C - 25.0)
        
        # Moisture degradation (significant above 100 ppm)
        if moisture_ppm > 100:
            moisture_factor = 1 - 0.003 * (moisture_ppm - 100)
            bdv *= max(0.5, moisture_factor)
        
        # Aging degradation
        if years_aged > 0:
            aging_factor = 1 - self.aging_rate_per_year * years_aged
            bdv *= max(0.5, aging_factor)
        
        return max(10.0, bdv)
    
    def dielectric_constant_at_temp(self, temp_C: float) -> float:
        """
        Calculate dielectric constant at temperature.
        
        Args:
            temp_C: Temperature in Celsius
            
        Returns:
            Relative permittivity (dimensionless)
        """
        return self.dielectric_constant_25C * (1 + self.dk_temp_coeff_per_C * (temp_C - 25.0))
    
    def loss_tangent_at_temp(self, temp_C: float) -> float:
        """
        Calculate loss tangent at temperature.
        
        Loss tangent increases with temperature due to increased
        ionic mobility.
        
        Args:
            temp_C: Temperature in Celsius
            
        Returns:
            Loss tangent (dimensionless)
        """
        # Loss tangent increases ~1% per °C above reference
        temp_factor = 1 + 0.01 * (temp_C - 25.0)
        return self.loss_tangent_25C * max(0.5, temp_factor)
    
    def aged_properties(
        self,
        temp_C: float,
        years_aged: float,
        moisture_ppm: float = 50.0
    ) -> dict:
        """
        Calculate all electrical properties with aging effects.
        
        Args:
            temp_C: Operating temperature in Celsius
            years_aged: Years of service
            moisture_ppm: Moisture content
            
        Returns:
            Dictionary of electrical properties
        """
        # Resistivity decreases with aging (contamination)
        rho = self.resistivity_at_temp(temp_C)
        rho_aged = rho * (1 - 0.1 * years_aged)  # 10% decrease per year
        rho_aged = max(1e10, rho_aged)  # Physical minimum
        
        return {
            "breakdown_voltage_kV": self.breakdown_voltage_at_temp(temp_C, moisture_ppm, years_aged),
            "volume_resistivity_ohm_cm": rho_aged,
            "dielectric_constant": self.dielectric_constant_at_temp(temp_C),
            "loss_tangent": self.loss_tangent_at_temp(temp_C),
        }


def arrhenius_resistivity(
    temp_K: float,
    rho_ref: float,
    T_ref: float = 298.15,
    activation_energy_eV: float = 0.69
) -> float:
    """
    Calculate volume resistivity using Arrhenius temperature dependence.
    
    ρ(T) = ρ_ref × exp[Ea/kB × (1/T - 1/T_ref)]
    
    Reference: Oommen, T. (2002). IEEE Electrical Insulation Magazine.
    
    Args:
        temp_K: Temperature in Kelvin
        rho_ref: Reference resistivity at T_ref in Ω·cm
        T_ref: Reference temperature in Kelvin
        activation_energy_eV: Activation energy in eV
        
    Returns:
        Volume resistivity in Ω·cm
    """
    kB = BOLTZMANN_CONSTANT_eV_K
    exponent = (activation_energy_eV / kB) * (1/temp_K - 1/T_ref)
    return rho_ref * np.exp(exponent)


def calculate_breakdown_voltage(
    temp_C: float,
    bdv_25C: float = 55.0,
    temp_coeff: float = -0.1,
    moisture_ppm: float = 50.0,
    years_aged: float = 0.0,
    aging_rate: float = 0.02
) -> float:
    """
    Calculate breakdown voltage with temperature, moisture, and aging effects.
    
    Args:
        temp_C: Temperature in Celsius
        bdv_25C: Breakdown voltage at 25°C in kV
        temp_coeff: Temperature coefficient (kV/°C)
        moisture_ppm: Moisture content in ppm
        years_aged: Years of thermal aging
        aging_rate: Annual degradation rate (fraction)
        
    Returns:
        Breakdown voltage in kV
    """
    # Temperature effect
    bdv = bdv_25C + temp_coeff * (temp_C - 25.0)
    
    # Moisture effect
    if moisture_ppm > 100:
        moisture_factor = 1 - 0.003 * (moisture_ppm - 100)
        bdv *= max(0.5, moisture_factor)
    
    # Aging effect
    if years_aged > 0:
        aging_factor = 1 - aging_rate * years_aged
        bdv *= max(0.5, aging_factor)
    
    return max(10.0, bdv)


def calculate_resistivity_at_temp(
    temp_C: float,
    rho_25C: float = 1e14,
    activation_energy_eV: float = 0.69
) -> float:
    """
    Calculate volume resistivity at temperature.
    
    Convenience function wrapping arrhenius_resistivity.
    
    Args:
        temp_C: Temperature in Celsius
        rho_25C: Resistivity at 25°C in Ω·cm
        activation_energy_eV: Activation energy in eV
        
    Returns:
        Volume resistivity in Ω·cm
    """
    temp_K = temp_C + 273.15
    return arrhenius_resistivity(temp_K, rho_25C, 298.15, activation_energy_eV)


@dataclass
class ElectricalSimulationResult:
    """Results from electrical property simulation."""
    
    temperature_C: float
    breakdown_voltage_kV: float
    volume_resistivity_ohm_cm: float
    dielectric_constant: float
    loss_tangent: float
    
    # OCP requirements
    bdv_requirement_kV: float = 45.0
    resistivity_requirement_ohm_cm: float = 1e11
    
    @property
    def bdv_margin_kV(self) -> float:
        """Margin above BDV requirement."""
        return self.breakdown_voltage_kV - self.bdv_requirement_kV
    
    @property
    def bdv_compliant(self) -> bool:
        """Check BDV compliance."""
        return self.breakdown_voltage_kV >= self.bdv_requirement_kV
    
    @property
    def resistivity_compliant(self) -> bool:
        """Check resistivity compliance."""
        return self.volume_resistivity_ohm_cm >= self.resistivity_requirement_ohm_cm
    
    @property
    def resistivity_margin_decades(self) -> float:
        """Margin above resistivity requirement in decades."""
        return np.log10(self.volume_resistivity_ohm_cm / self.resistivity_requirement_ohm_cm)


def run_electrical_simulation(
    model: ElectricalModel,
    temp_C: float,
    years_aged: float = 0.0,
    moisture_ppm: float = 50.0
) -> ElectricalSimulationResult:
    """
    Run electrical property simulation.
    
    Args:
        model: ElectricalModel instance
        temp_C: Operating temperature
        years_aged: Service years
        moisture_ppm: Moisture content
        
    Returns:
        ElectricalSimulationResult
    """
    props = model.aged_properties(temp_C, years_aged, moisture_ppm)
    
    return ElectricalSimulationResult(
        temperature_C=temp_C,
        breakdown_voltage_kV=props["breakdown_voltage_kV"],
        volume_resistivity_ohm_cm=props["volume_resistivity_ohm_cm"],
        dielectric_constant=props["dielectric_constant"],
        loss_tangent=props["loss_tangent"],
    )
