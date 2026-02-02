"""
Economic analysis for immersion cooling fluid selection.

This module implements cost comparison between domestic Group III oils
and imported fluorinated fluids, including forex savings projections.

From paper: 83% cost reduction, $18M USD savings over 2025-2030.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class FluidCost:
    """Cost parameters for a cooling fluid."""
    
    name: str
    cost_per_liter_usd: float
    cost_std_usd: float = 0.0
    
    # Supply chain factors
    lead_time_weeks: float = 2.0
    minimum_order_liters: float = 1000.0
    is_domestic: bool = True
    
    # Maintenance costs
    annual_top_up_percent: float = 5.0  # Percent of volume annually
    disposal_cost_per_liter_usd: float = 1.0
    
    # Exchange rate (for imports)
    import_duty_percent: float = 0.0
    forex_risk_premium: float = 0.0  # Additional % for currency risk


@dataclass
class DataCenterConfig:
    """Data center configuration for cost calculations."""
    
    power_capacity_MW: float = 1.0
    
    # Fluid requirements (liters per MW)
    fluid_per_MW_liters: float = 50_000.0  # ~50,000 L/MW typical
    
    # Service life
    fluid_service_life_years: float = 5.0
    facility_lifetime_years: float = 15.0
    
    # Growth
    annual_capacity_growth_percent: float = 30.0  # India AI DC growth rate


@dataclass
class CostComparison:
    """
    Comparison results between two cooling fluid options.
    """
    
    fluid_a_name: str
    fluid_b_name: str
    
    # Initial costs
    initial_cost_a_usd: float
    initial_cost_b_usd: float
    
    # TCO over facility lifetime
    tco_a_usd: float
    tco_b_usd: float
    
    # Savings
    absolute_savings_usd: float
    relative_savings_percent: float
    
    # Detailed breakdown
    breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Generate cost comparison summary."""
        cheaper = self.fluid_a_name if self.tco_a_usd < self.tco_b_usd else self.fluid_b_name
        
        return f"""
COST COMPARISON: {self.fluid_a_name} vs {self.fluid_b_name}
{'='*60}

INITIAL FILL COST:
  {self.fluid_a_name}: ${self.initial_cost_a_usd:,.0f}
  {self.fluid_b_name}: ${self.initial_cost_b_usd:,.0f}

TOTAL COST OF OWNERSHIP (Facility Lifetime):
  {self.fluid_a_name}: ${self.tco_a_usd:,.0f}
  {self.fluid_b_name}: ${self.tco_b_usd:,.0f}

SAVINGS WITH {cheaper}:
  Absolute: ${abs(self.absolute_savings_usd):,.0f}
  Relative: {abs(self.relative_savings_percent):.1f}%
{'='*60}
"""


@dataclass
class ForexSavingsProjection:
    """
    Foreign exchange savings projection from domestic production.
    """
    
    years: List[int]
    annual_savings_usd: List[float]
    cumulative_savings_usd: List[float]
    
    # Market assumptions
    market_growth_rate: float
    domestic_adoption_rate: float
    
    @property
    def total_savings_usd(self) -> float:
        """Total forex savings over projection period."""
        return self.cumulative_savings_usd[-1] if self.cumulative_savings_usd else 0.0
    
    @property
    def total_savings_inr_crores(self) -> float:
        """Total savings in INR Crores (1 Crore = 10M INR)."""
        # Approximate exchange rate: 83 INR/USD
        return self.total_savings_usd * 83 / 1e7


class EconomicAnalysis:
    """
    Economic analysis engine for cooling fluid selection.
    """
    
    def __init__(
        self,
        domestic_fluid: FluidCost,
        imported_fluid: FluidCost,
        dc_config: DataCenterConfig = None,
    ):
        """
        Initialize economic analysis.
        
        Args:
            domestic_fluid: Domestic fluid cost parameters
            imported_fluid: Imported fluid cost parameters
            dc_config: Data center configuration
        """
        self.domestic = domestic_fluid
        self.imported = imported_fluid
        self.dc_config = dc_config or DataCenterConfig()
    
    def calculate_initial_fill_cost(
        self,
        fluid: FluidCost,
        volume_liters: float
    ) -> float:
        """
        Calculate initial fluid fill cost.
        
        Args:
            fluid: Fluid cost parameters
            volume_liters: Required volume in liters
            
        Returns:
            Total cost in USD
        """
        base_cost = volume_liters * fluid.cost_per_liter_usd
        
        # Add import duties and forex premium for imports
        if not fluid.is_domestic:
            base_cost *= (1 + fluid.import_duty_percent / 100)
            base_cost *= (1 + fluid.forex_risk_premium / 100)
        
        return base_cost
    
    def calculate_tco(
        self,
        fluid: FluidCost,
        volume_liters: float,
        years: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total cost of ownership.
        
        Includes: initial fill, annual top-up, disposal, supply chain costs.
        
        Args:
            fluid: Fluid cost parameters
            volume_liters: Initial volume in liters
            years: Facility lifetime in years
            
        Returns:
            Tuple of (total_cost, breakdown_dict)
        """
        # Initial fill
        initial_cost = self.calculate_initial_fill_cost(fluid, volume_liters)
        
        # Number of complete fluid replacements
        n_replacements = int(years / self.dc_config.fluid_service_life_years)
        replacement_cost = n_replacements * initial_cost
        
        # Annual top-up
        annual_topup_liters = volume_liters * fluid.annual_top_up_percent / 100
        annual_topup_cost = annual_topup_liters * fluid.cost_per_liter_usd
        total_topup_cost = annual_topup_cost * years
        
        # Disposal costs
        total_disposal_volume = volume_liters * (n_replacements + 1)
        disposal_cost = total_disposal_volume * fluid.disposal_cost_per_liter_usd
        
        # Supply chain / inventory holding cost (simplified)
        inventory_cost = initial_cost * 0.02 * years  # 2% annual holding cost
        
        total_cost = (
            initial_cost +
            replacement_cost +
            total_topup_cost +
            disposal_cost +
            inventory_cost
        )
        
        breakdown = {
            "initial_fill": initial_cost,
            "replacements": replacement_cost,
            "top_up": total_topup_cost,
            "disposal": disposal_cost,
            "inventory": inventory_cost,
        }
        
        return total_cost, breakdown
    
    def compare_fluids(self, power_MW: float = None) -> CostComparison:
        """
        Compare domestic vs imported fluid costs.
        
        Args:
            power_MW: Data center power capacity (uses config default if None)
            
        Returns:
            CostComparison result
        """
        if power_MW is None:
            power_MW = self.dc_config.power_capacity_MW
        
        volume = power_MW * self.dc_config.fluid_per_MW_liters
        years = self.dc_config.facility_lifetime_years
        
        # Calculate costs for each fluid
        initial_a = self.calculate_initial_fill_cost(self.domestic, volume)
        initial_b = self.calculate_initial_fill_cost(self.imported, volume)
        
        tco_a, breakdown_a = self.calculate_tco(self.domestic, volume, years)
        tco_b, breakdown_b = self.calculate_tco(self.imported, volume, years)
        
        # Savings (positive = domestic cheaper)
        absolute_savings = tco_b - tco_a
        relative_savings = 100 * absolute_savings / tco_b if tco_b > 0 else 0
        
        return CostComparison(
            fluid_a_name=self.domestic.name,
            fluid_b_name=self.imported.name,
            initial_cost_a_usd=initial_a,
            initial_cost_b_usd=initial_b,
            tco_a_usd=tco_a,
            tco_b_usd=tco_b,
            absolute_savings_usd=absolute_savings,
            relative_savings_percent=relative_savings,
            breakdown={
                self.domestic.name: breakdown_a,
                self.imported.name: breakdown_b,
            },
        )
    
    def project_forex_savings(
        self,
        start_year: int = 2025,
        end_year: int = 2030,
        initial_market_MW: float = 100.0,
        domestic_adoption_rate: float = 0.5,
    ) -> ForexSavingsProjection:
        """
        Project foreign exchange savings from domestic production.
        
        Based on paper's $18M savings projection for 2025-2030.
        
        Args:
            start_year: Projection start year
            end_year: Projection end year
            initial_market_MW: Initial market size in MW
            domestic_adoption_rate: Fraction adopting domestic fluid
            
        Returns:
            ForexSavingsProjection
        """
        years = list(range(start_year, end_year + 1))
        growth_rate = self.dc_config.annual_capacity_growth_percent / 100
        
        # Cost difference per liter
        cost_diff = self.imported.cost_per_liter_usd - self.domestic.cost_per_liter_usd
        liters_per_MW = self.dc_config.fluid_per_MW_liters
        
        annual_savings = []
        cumulative = 0.0
        cumulative_savings = []
        
        market_MW = initial_market_MW
        
        for i, year in enumerate(years):
            # New capacity added this year
            if i == 0:
                new_capacity = market_MW
            else:
                new_capacity = market_MW * growth_rate
                market_MW += new_capacity
            
            # Forex savings from domestic adoption
            domestic_MW = new_capacity * domestic_adoption_rate
            volume_liters = domestic_MW * liters_per_MW
            savings = volume_liters * cost_diff
            
            # Add top-up savings for existing domestic installations
            if i > 0:
                existing_domestic_MW = sum([
                    initial_market_MW * domestic_adoption_rate * (1 + growth_rate) ** j
                    for j in range(i)
                ])
                topup_volume = existing_domestic_MW * liters_per_MW * 0.05
                savings += topup_volume * cost_diff
            
            annual_savings.append(savings)
            cumulative += savings
            cumulative_savings.append(cumulative)
        
        return ForexSavingsProjection(
            years=years,
            annual_savings_usd=annual_savings,
            cumulative_savings_usd=cumulative_savings,
            market_growth_rate=growth_rate,
            domestic_adoption_rate=domestic_adoption_rate,
        )


def calculate_forex_savings(
    start_year: int = 2025,
    end_year: int = 2030,
    domestic_cost_per_liter: float = 10.0,
    imported_cost_per_liter: float = 400.0,
    initial_market_MW: float = 100.0,
    growth_rate_percent: float = 30.0,
    adoption_rate: float = 0.5,
) -> ForexSavingsProjection:
    """
    Convenience function to calculate forex savings projection.
    
    Default parameters based on paper assumptions:
    - Group III: $10/L
    - Fluorinert FC-77: $400/L
    - Indian AI DC growth: 30% CAGR
    - 50% domestic adoption
    
    Args:
        start_year: Start year for projection
        end_year: End year for projection
        domestic_cost_per_liter: Domestic fluid cost ($/L)
        imported_cost_per_liter: Imported fluid cost ($/L)
        initial_market_MW: Initial market size (MW)
        growth_rate_percent: Annual growth rate (%)
        adoption_rate: Domestic fluid adoption rate (0-1)
        
    Returns:
        ForexSavingsProjection
    """
    domestic = FluidCost(
        name="Group III Oil",
        cost_per_liter_usd=domestic_cost_per_liter,
        is_domestic=True,
    )
    
    imported = FluidCost(
        name="Fluorinert FC-77",
        cost_per_liter_usd=imported_cost_per_liter,
        is_domestic=False,
        import_duty_percent=10.0,
    )
    
    dc_config = DataCenterConfig(
        annual_capacity_growth_percent=growth_rate_percent,
    )
    
    analysis = EconomicAnalysis(domestic, imported, dc_config)
    
    return analysis.project_forex_savings(
        start_year=start_year,
        end_year=end_year,
        initial_market_MW=initial_market_MW,
        domestic_adoption_rate=adoption_rate,
    )
