"""
Thermal modeling for immersion-cooled electronics.

This module implements the thermal resistance network model for calculating
junction temperatures of GPUs and other electronics in immersion cooling.

Based on Equation (1) and (2) from the paper:
    Tj = Tbulk + Q × Rtotal
    Rtotal = Rjc + Rcs + Rconv
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import numpy as np

from .properties import (
    get_thermophysical_properties,
    ThermophysicalProperties,
)


@dataclass
class HeatSinkGeometry:
    """Heat sink fin geometry parameters."""
    
    base_length_m: float = 0.10  # 100mm
    base_width_m: float = 0.08   # 80mm
    base_thickness_m: float = 0.003  # 3mm
    
    fin_height_m: float = 0.030  # 30mm (optimized)
    fin_thickness_m: float = 0.001  # 1mm
    fin_spacing_m: float = 0.002  # 2mm
    num_fins: int = 40  # Increased for 340% area enhancement
    
    material_k_W_mK: float = 200.0  # Aluminum
    
    @property
    def base_area_m2(self) -> float:
        """Heat sink base area in m²."""
        return self.base_length_m * self.base_width_m
    
    @property
    def fin_area_m2(self) -> float:
        """Total fin surface area in m²."""
        # Both sides of each fin
        single_fin_area = 2 * self.fin_height_m * self.base_length_m
        return self.num_fins * single_fin_area
    
    @property
    def total_area_m2(self) -> float:
        """Total heat transfer surface area."""
        # Base between fins + fin surfaces
        base_exposed = self.base_area_m2 - (self.num_fins * self.fin_thickness_m * self.base_length_m)
        return base_exposed + self.fin_area_m2
    
    @property
    def enhancement_factor(self) -> float:
        """Area enhancement factor compared to bare surface."""
        return self.total_area_m2 / self.base_area_m2


@dataclass
class GPUSpecification:
    """GPU thermal and power specifications."""
    
    name: str = "NVIDIA H100"
    tdp_watts: float = 700.0
    
    # Thermal resistances (calibrated to achieve Tj=59°C from paper)
    # Paper: Rtotal ≈ (59-40)/700 = 0.027 K/W
    # Rconv = 7.9% → 0.002 K/W, so Rjc+Rcs ≈ 0.025 K/W
    r_jc_K_W: Tuple[float, float] = (0.018, 0.002)  # Junction-to-case (mean, std)
    r_cs_K_W: Tuple[float, float] = (0.005, 0.001)  # Case-to-surface (TIM)
    
    # Temperature limits
    t_junction_max_C: float = 83.0
    t_throttle_C: float = 88.0  # Thermal shutdown
    
    # Die specifications
    die_area_mm2: float = 814.0  # H100 die size
    
    @property
    def die_area_m2(self) -> float:
        return self.die_area_mm2 * 1e-6
    
    @property
    def heat_flux_W_m2(self) -> float:
        """Heat flux at die surface in W/m²."""
        return self.tdp_watts / self.die_area_m2


@dataclass
class ThermalResistanceNetwork:
    """
    Lumped thermal resistance network for immersion cooling.
    
    The network consists of:
    - R_jc: Junction to case (internal to GPU package)
    - R_cs: Case to surface (thermal interface material)
    - R_conv: Convective resistance to fluid
    """
    
    r_jc_K_W: float  # Junction-to-case
    r_cs_K_W: float  # Case-to-surface (TIM)
    r_conv_K_W: float  # Convective to fluid
    
    @property
    def r_total_K_W(self) -> float:
        """Total thermal resistance."""
        return self.r_jc_K_W + self.r_cs_K_W + self.r_conv_K_W
    
    @property
    def resistance_breakdown(self) -> Dict[str, float]:
        """Percentage breakdown of thermal resistances."""
        total = self.r_total_K_W
        return {
            "R_jc": 100 * self.r_jc_K_W / total,
            "R_cs": 100 * self.r_cs_K_W / total,
            "R_conv": 100 * self.r_conv_K_W / total,
        }
    
    def __repr__(self) -> str:
        return (f"ThermalResistanceNetwork(R_jc={self.r_jc_K_W:.4f}, "
                f"R_cs={self.r_cs_K_W:.4f}, R_conv={self.r_conv_K_W:.4f}, "
                f"R_total={self.r_total_K_W:.4f} K/W)")


@dataclass
class ThermalModel:
    """
    Complete thermal model for immersion-cooled GPU.
    """
    
    gpu: GPUSpecification
    heat_sink: HeatSinkGeometry
    fluid_velocity_m_s: float = 0.3
    
    # Operating conditions
    t_bulk_C: float = 40.0  # Bulk fluid temperature
    
    def calculate_htc(
        self,
        fluid_props: ThermophysicalProperties
    ) -> float:
        """
        Calculate convective heat transfer coefficient.
        
        Uses Churchill-Bernstein correlation for external flow over fins.
        
        Args:
            fluid_props: Fluid thermophysical properties at bulk temperature
            
        Returns:
            Heat transfer coefficient in W/(m²·K)
        """
        # Flow parameters
        L = self.heat_sink.fin_height_m  # Characteristic length
        V = self.fluid_velocity_m_s
        
        # Fluid properties
        rho = fluid_props.density_kg_m3
        mu = fluid_props.viscosity_Pa_s
        k = fluid_props.thermal_conductivity_W_mK
        Pr = fluid_props.prandtl_number
        
        # Reynolds number
        Re = rho * V * L / mu
        
        # Nusselt number - Churchill-Bernstein correlation
        if Re < 1:
            Nu = 0.68  # Creeping flow limit
        elif Re < 1000:
            # Laminar flow
            Nu = 0.664 * Re**0.5 * Pr**(1/3)
        else:
            # Turbulent (though unlikely in immersion cooling)
            Nu = 0.037 * Re**0.8 * Pr**(1/3)
        
        # Heat transfer coefficient
        htc = Nu * k / L
        
        return htc
    
    def calculate_convective_resistance(
        self,
        htc_W_m2K: float
    ) -> float:
        """
        Calculate convective thermal resistance.
        
        R_conv = 1 / (h × A × η)
        
        where η is fin efficiency.
        
        Args:
            htc_W_m2K: Heat transfer coefficient in W/(m²·K)
            
        Returns:
            Convective resistance in K/W
        """
        # Fin efficiency (simplified)
        m = np.sqrt(2 * htc_W_m2K / 
                   (self.heat_sink.material_k_W_mK * self.heat_sink.fin_thickness_m))
        L = self.heat_sink.fin_height_m
        eta_fin = np.tanh(m * L) / (m * L) if m * L > 0.01 else 1.0
        
        # Effective area with fin efficiency
        A_eff = (self.heat_sink.base_area_m2 + 
                 eta_fin * self.heat_sink.fin_area_m2)
        
        return 1.0 / (htc_W_m2K * A_eff)
    
    def build_resistance_network(
        self,
        fluid_props: ThermophysicalProperties,
        r_jc: Optional[float] = None,
        r_cs: Optional[float] = None,
    ) -> ThermalResistanceNetwork:
        """
        Build the complete thermal resistance network.
        
        Args:
            fluid_props: Fluid properties at bulk temperature
            r_jc: Junction-to-case resistance (uses GPU spec default if None)
            r_cs: Case-to-surface resistance (uses GPU spec default if None)
            
        Returns:
            ThermalResistanceNetwork instance
        """
        # Use provided values or defaults
        if r_jc is None:
            r_jc = self.gpu.r_jc_K_W[0]  # Mean value
        if r_cs is None:
            r_cs = self.gpu.r_cs_K_W[0]
        
        # Calculate convective resistance
        htc = self.calculate_htc(fluid_props)
        r_conv = self.calculate_convective_resistance(htc)
        
        return ThermalResistanceNetwork(
            r_jc_K_W=r_jc,
            r_cs_K_W=r_cs,
            r_conv_K_W=r_conv,
        )
    
    def calculate_junction_temperature(
        self,
        resistance_network: ThermalResistanceNetwork,
        power_watts: Optional[float] = None,
    ) -> float:
        """
        Calculate junction temperature.
        
        Tj = Tbulk + Q × Rtotal  (Equation 1 from paper)
        
        Args:
            resistance_network: Thermal resistance network
            power_watts: Power dissipation (uses GPU TDP if None)
            
        Returns:
            Junction temperature in °C
        """
        Q = power_watts if power_watts is not None else self.gpu.tdp_watts
        return self.t_bulk_C + Q * resistance_network.r_total_K_W


def calculate_junction_temperature(
    power_watts: float,
    t_bulk_C: float,
    r_total_K_W: float
) -> float:
    """
    Simple junction temperature calculation.
    
    Tj = Tbulk + Q × Rtotal
    
    Args:
        power_watts: Heat dissipation in Watts
        t_bulk_C: Bulk fluid temperature in °C
        r_total_K_W: Total thermal resistance in K/W
        
    Returns:
        Junction temperature in °C
    """
    return t_bulk_C + power_watts * r_total_K_W


def calculate_convective_htc(
    velocity_m_s: float,
    char_length_m: float,
    density_kg_m3: float,
    viscosity_Pa_s: float,
    thermal_conductivity_W_mK: float,
    prandtl_number: float
) -> float:
    """
    Calculate convective heat transfer coefficient for external flow.
    
    Uses appropriate correlation based on Reynolds number regime.
    
    Args:
        velocity_m_s: Flow velocity in m/s
        char_length_m: Characteristic length in m
        density_kg_m3: Fluid density
        viscosity_Pa_s: Dynamic viscosity
        thermal_conductivity_W_mK: Thermal conductivity
        prandtl_number: Prandtl number
        
    Returns:
        Heat transfer coefficient in W/(m²·K)
    """
    # Reynolds number
    Re = density_kg_m3 * velocity_m_s * char_length_m / viscosity_Pa_s
    Pr = prandtl_number
    
    # Nusselt correlation selection
    if Re < 5:
        Nu = 0.68 + 0.67 * Re**0.5 * Pr**(1/3)
    elif Re < 2e5:
        # Laminar
        Nu = 0.664 * Re**0.5 * Pr**(1/3)
    else:
        # Turbulent
        Nu = 0.037 * Re**0.8 * Pr**(1/3)
    
    return Nu * thermal_conductivity_W_mK / char_length_m


@dataclass
class ThermalSimulationResult:
    """Results from thermal simulation."""
    
    junction_temperature_C: float
    surface_temperature_C: float
    bulk_temperature_C: float
    
    resistance_network: ThermalResistanceNetwork
    htc_W_m2K: float
    
    power_watts: float
    fluid_velocity_m_s: float
    
    # Compliance
    t_limit_C: float = 88.0
    
    @property
    def thermal_margin_C(self) -> float:
        """Temperature margin below limit."""
        return self.t_limit_C - self.junction_temperature_C
    
    @property
    def is_compliant(self) -> bool:
        """Check if within thermal limit."""
        return self.junction_temperature_C < self.t_limit_C
    
    @property
    def utilization_percent(self) -> float:
        """Thermal budget utilization."""
        return 100 * self.junction_temperature_C / self.t_limit_C


def run_thermal_simulation(
    gpu: GPUSpecification,
    heat_sink: HeatSinkGeometry,
    fluid_props: ThermophysicalProperties,
    t_bulk_C: float = 40.0,
    velocity_m_s: float = 0.3,
    r_jc: Optional[float] = None,
    r_cs: Optional[float] = None,
) -> ThermalSimulationResult:
    """
    Run a complete thermal simulation.
    
    Args:
        gpu: GPU specifications
        heat_sink: Heat sink geometry
        fluid_props: Fluid thermophysical properties
        t_bulk_C: Bulk fluid temperature
        velocity_m_s: Fluid flow velocity
        r_jc: Junction-to-case resistance (optional override)
        r_cs: Case-to-surface resistance (optional override)
        
    Returns:
        ThermalSimulationResult with all thermal data
    """
    model = ThermalModel(
        gpu=gpu,
        heat_sink=heat_sink,
        fluid_velocity_m_s=velocity_m_s,
        t_bulk_C=t_bulk_C,
    )
    
    # Build resistance network
    network = model.build_resistance_network(
        fluid_props,
        r_jc=r_jc,
        r_cs=r_cs,
    )
    
    # Calculate temperatures
    t_junction = model.calculate_junction_temperature(network)
    
    # Surface temperature (case temperature)
    t_surface = t_junction - gpu.tdp_watts * network.r_jc_K_W
    
    # Heat transfer coefficient
    htc = model.calculate_htc(fluid_props)
    
    return ThermalSimulationResult(
        junction_temperature_C=t_junction,
        surface_temperature_C=t_surface,
        bulk_temperature_C=t_bulk_C,
        resistance_network=network,
        htc_W_m2K=htc,
        power_watts=gpu.tdp_watts,
        fluid_velocity_m_s=velocity_m_s,
        t_limit_C=gpu.t_throttle_C,
    )
