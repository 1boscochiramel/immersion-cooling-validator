"""
Test suite for Immersion Cooling Validator.

Run with: pytest tests/test_icv.py -v
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose


class TestFluids:
    """Test fluid definitions."""
    
    def test_group_iii_oil_creation(self):
        from icv import GroupIIIOil
        
        fluid = GroupIIIOil()
        assert fluid.name == "Group III Base Oil"
        assert fluid.properties.viscosity_cSt_40C[0] == 4.2  # 4cSt default
    
    def test_group_iii_viscosity_grades(self):
        from icv import GroupIIIOil
        
        oil_4 = GroupIIIOil(viscosity_grade="4cSt")
        oil_6 = GroupIIIOil(viscosity_grade="6cSt")
        oil_8 = GroupIIIOil(viscosity_grade="8cSt")
        
        assert oil_4.properties.viscosity_cSt_40C[0] == 4.2
        assert oil_6.properties.viscosity_cSt_40C[0] == 6.0
        assert oil_8.properties.viscosity_cSt_40C[0] == 8.0
    
    def test_fluorinert_fc77(self):
        from icv import FluorinertFC77
        
        fluid = FluorinertFC77()
        assert fluid.name == "Fluorinert FC-77"
        assert fluid.properties.cost_per_liter_usd[0] == 400.0
    
    def test_create_fluid_factory(self):
        from icv import create_fluid
        
        fluid = create_fluid("group_iii")
        assert "Group III" in fluid.name
        
        fc77 = create_fluid("fc77")
        assert "Fluorinert" in fc77.name
    
    def test_fluid_sampling(self):
        from icv import GroupIIIOil
        import numpy as np
        
        fluid = GroupIIIOil()
        rng = np.random.default_rng(42)
        
        samples = fluid.sample_properties(rng)
        
        assert "density_kg_m3" in samples
        assert "breakdown_voltage_kV" in samples
        assert samples["density_kg_m3"] > 0


class TestProperties:
    """Test thermophysical and electrical property models."""
    
    def test_density_temperature_dependence(self):
        from icv.properties import calculate_density
        
        # Density should decrease with temperature
        rho_15 = calculate_density(15.0, density_15C=825.0)
        rho_60 = calculate_density(60.0, density_15C=825.0)
        
        assert rho_15 == 825.0  # Reference point
        assert rho_60 < rho_15  # Decreases with temp
    
    def test_viscosity_temperature_dependence(self):
        from icv.properties import calculate_viscosity
        
        # Viscosity should decrease with temperature
        mu_40 = calculate_viscosity(40.0, viscosity_40C=10.0)
        mu_60 = calculate_viscosity(60.0, viscosity_40C=10.0)
        
        assert_allclose(mu_40, 10.0, rtol=0.01)  # Reference point
        assert mu_60 < mu_40  # Decreases with temp
    
    def test_arrhenius_resistivity(self):
        from icv.properties import arrhenius_resistivity
        
        # Resistivity should decrease with temperature (for Arrhenius with positive Ea)
        rho_25C = arrhenius_resistivity(298.15, rho_ref=1e14)
        rho_90C = arrhenius_resistivity(363.15, rho_ref=1e14)
        
        assert rho_90C < rho_25C  # Decreases with temp
        
        # Check magnitude is reasonable
        assert rho_90C > 1e10
    
    def test_thermophysical_properties_bundle(self):
        from icv.properties import get_thermophysical_properties
        
        props = get_thermophysical_properties(temp_C=50.0)
        
        assert props.temperature_C == 50.0
        assert props.density_kg_m3 > 0
        assert props.viscosity_cSt > 0
        assert props.thermal_conductivity_W_mK > 0
        assert props.prandtl_number > 0


class TestThermal:
    """Test thermal model."""
    
    def test_junction_temperature_calculation(self):
        from icv.thermal import calculate_junction_temperature
        
        # Tj = Tbulk + Q * Rtotal
        Tj = calculate_junction_temperature(
            power_watts=700.0,
            t_bulk_C=40.0,
            r_total_K_W=0.03
        )
        
        expected = 40.0 + 700.0 * 0.03  # = 61°C
        assert_allclose(Tj, expected, rtol=0.01)
    
    def test_thermal_resistance_network(self):
        from icv.thermal import ThermalResistanceNetwork
        
        network = ThermalResistanceNetwork(
            r_jc_K_W=0.05,
            r_cs_K_W=0.02,
            r_conv_K_W=0.01
        )
        
        assert_allclose(network.r_total_K_W, 0.08, rtol=0.01)
        
        breakdown = network.resistance_breakdown
        assert sum(breakdown.values()) == pytest.approx(100.0, rel=0.01)
    
    def test_gpu_specification(self):
        from icv.thermal import GPUSpecification
        
        gpu = GPUSpecification()
        
        assert gpu.name == "NVIDIA H100"
        assert gpu.tdp_watts == 700.0
        assert gpu.t_throttle_C == 88.0


class TestElectrical:
    """Test electrical property models."""
    
    def test_breakdown_voltage_temperature_effect(self):
        from icv.electrical import calculate_breakdown_voltage
        
        # BDV should decrease with temperature
        bdv_25 = calculate_breakdown_voltage(25.0, bdv_25C=55.0)
        bdv_60 = calculate_breakdown_voltage(60.0, bdv_25C=55.0)
        
        assert_allclose(bdv_25, 55.0, rtol=0.01)
        assert bdv_60 < bdv_25
    
    def test_resistivity_at_temperature(self):
        from icv.electrical import calculate_resistivity_at_temp
        
        rho_25 = calculate_resistivity_at_temp(25.0, rho_25C=1e14)
        rho_90 = calculate_resistivity_at_temp(90.0, rho_25C=1e14)
        
        # Resistivity should be ~1e14 at reference
        assert_allclose(rho_25, 1e14, rtol=0.01)
        
        # Should decrease at higher temp
        assert rho_90 < rho_25
        
        # From paper: should still be above 1e11 at 90°C
        assert rho_90 > 1e11
    
    def test_electrical_model_aging(self):
        from icv.electrical import ElectricalModel
        
        model = ElectricalModel()
        
        props_new = model.aged_properties(60.0, years_aged=0.0)
        props_old = model.aged_properties(60.0, years_aged=5.0)
        
        # Aged fluid should have lower BDV
        assert props_old["breakdown_voltage_kV"] < props_new["breakdown_voltage_kV"]


class TestLifetime:
    """Test Weibull lifetime model."""
    
    def test_weibull_survival(self):
        from icv.lifetime import weibull_survival
        
        # At t=0, survival should be 1.0
        assert weibull_survival(0.0) == 1.0
        
        # At t=λ, survival should be exp(-1) ≈ 0.368
        assert_allclose(weibull_survival(7.0, scale_lambda=7.0), np.exp(-1), rtol=0.01)
    
    def test_weibull_percentile(self):
        from icv.lifetime import weibull_percentile
        
        # P50 should be close to scale parameter for high shape
        p50 = weibull_percentile(50.0, shape_k=8.8, scale_lambda=7.0)
        
        # From paper: P50 = 6.6 years
        assert_allclose(p50, 6.6, rtol=0.1)
    
    def test_p5_life(self):
        from icv.lifetime import LifetimeModel
        
        model = LifetimeModel(shape_k=8.8, scale_lambda=8.1)
        
        # With calibrated scale=8.1: P5 ≈ 5.8 years
        assert_allclose(model.p5_life, 5.8, rtol=0.15)
    
    def test_temperature_acceleration(self):
        from icv.lifetime import LifetimeModel
        
        model = LifetimeModel()
        
        # Higher temperature should accelerate aging (AF > 1)
        af_high = model.temperature_acceleration_factor(100.0, reference_temp_C=90.0)
        assert af_high > 1.0
        
        # Lower temperature should slow aging (AF < 1)
        af_low = model.temperature_acceleration_factor(60.0, reference_temp_C=90.0)
        assert af_low < 1.0


class TestMonteCarlo:
    """Test Monte Carlo simulation engine."""
    
    def test_monte_carlo_basic(self):
        from icv import GroupIIIOil, run_simulation
        
        # Run with small sample size for speed
        result = run_simulation(n_samples=100, random_seed=42)
        
        assert result.n_samples == 100
        assert result.t_junction_mean > 0
        assert result.bdv_mean > 0
    
    def test_monte_carlo_reproducibility(self):
        from icv import GroupIIIOil, run_simulation
        
        result1 = run_simulation(n_samples=100, random_seed=42)
        result2 = run_simulation(n_samples=100, random_seed=42)
        
        # Same seed should give same results
        assert_allclose(result1.t_junction_mean, result2.t_junction_mean, rtol=0.001)
    
    def test_monte_carlo_compliance_probabilities(self):
        from icv import run_simulation
        
        result = run_simulation(n_samples=1000, random_seed=42)
        
        # All probabilities should be between 0 and 1
        assert 0 <= result.thermal_compliance_prob <= 1
        assert 0 <= result.bdv_compliance_prob <= 1
        assert 0 <= result.joint_compliance_prob <= 1
        
        # Joint should be <= minimum individual
        min_individual = min(
            result.thermal_compliance_prob,
            result.bdv_compliance_prob,
            result.resistivity_compliance_prob,
            result.lifetime_compliance_prob
        )
        assert result.joint_compliance_prob <= min_individual
    
    def test_monte_carlo_paper_values(self):
        """Test that results match paper values approximately."""
        from icv import run_simulation
        
        result = run_simulation(n_samples=10_000, random_seed=42)
        
        # From paper: Tj = 59 ± 5°C
        assert 50 < result.t_junction_mean < 70
        
        # From paper: BDV = 55 ± 3 kV
        assert 45 < result.bdv_mean < 65
        
        # From paper: Joint compliance = 97.6%
        # Allow some variation due to implementation differences
        assert result.joint_compliance_prob > 0.90


class TestCompliance:
    """Test OCP compliance checker."""
    
    def test_compliance_pass(self):
        from icv import check_ocp_compliance
        
        result = check_ocp_compliance(
            junction_temp_C=59.0,
            breakdown_voltage_kV=55.0,
            volume_resistivity_ohm_cm=1e13,
            p5_life_years=5.8,
        )
        
        assert result.critical_pass is True
    
    def test_compliance_fail_thermal(self):
        from icv import check_ocp_compliance
        
        result = check_ocp_compliance(
            junction_temp_C=95.0,  # Above 88°C limit
            breakdown_voltage_kV=55.0,
            volume_resistivity_ohm_cm=1e13,
            p5_life_years=5.8,
        )
        
        assert result.all_pass is False
    
    def test_compliance_fail_bdv(self):
        from icv import check_ocp_compliance
        
        result = check_ocp_compliance(
            junction_temp_C=59.0,
            breakdown_voltage_kV=40.0,  # Below 45 kV requirement
            volume_resistivity_ohm_cm=1e13,
            p5_life_years=5.8,
        )
        
        assert result.all_pass is False


class TestEconomics:
    """Test economic analysis."""
    
    def test_cost_comparison(self):
        from icv.economics import EconomicAnalysis, FluidCost
        
        domestic = FluidCost(name="Group III", cost_per_liter_usd=10.0, is_domestic=True)
        imported = FluidCost(name="FC-77", cost_per_liter_usd=400.0, is_domestic=False)
        
        analysis = EconomicAnalysis(domestic, imported)
        comparison = analysis.compare_fluids(power_MW=1.0)
        
        # Domestic should be cheaper
        assert comparison.tco_a_usd < comparison.tco_b_usd
        
        # Savings should be significant (>80%)
        assert comparison.relative_savings_percent > 80
    
    def test_forex_savings(self):
        from icv import calculate_forex_savings
        
        projection = calculate_forex_savings(
            start_year=2025,
            end_year=2030,
        )
        
        assert len(projection.years) == 6
        assert projection.total_savings_usd > 0
        
        # Cumulative should be increasing
        for i in range(1, len(projection.cumulative_savings_usd)):
            assert projection.cumulative_savings_usd[i] >= projection.cumulative_savings_usd[i-1]


class TestValidator:
    """Test main validator."""
    
    def test_full_validation(self):
        from icv import GroupIIIOil, ImmersionCoolingValidator
        
        fluid = GroupIIIOil()
        validator = ImmersionCoolingValidator(fluid)
        
        result = validator.run_full_validation(
            n_samples=100,  # Small for speed
            include_economics=True,
            include_benchmark=False,  # Skip for speed
        )
        
        assert result.fluid_name == "Group III Base Oil"
        assert result.monte_carlo is not None
        assert result.compliance is not None
        assert result.economics is not None
    
    def test_validate_fluid_convenience(self):
        from icv import validate_fluid
        
        result = validate_fluid(n_samples=100, include_economics=False)
        
        assert result is not None
        assert result.monte_carlo.n_samples == 100


class TestIntegration:
    """Integration tests matching paper results."""
    
    def test_paper_results_reproduction(self):
        """
        Verify we can approximately reproduce the paper's key findings.
        """
        from icv import GroupIIIOil, ImmersionCoolingValidator
        
        fluid = GroupIIIOil()
        validator = ImmersionCoolingValidator(fluid)
        
        result = validator.run_full_validation(
            n_samples=10_000,
            random_seed=42,
            include_economics=True,
            include_benchmark=False,
        )
        
        mc = result.monte_carlo
        
        # Paper: Tj = 59 ± 5°C (allow wider range for model differences)
        assert 55 < mc.t_junction_mean < 70, f"Tj={mc.t_junction_mean}"
        assert 1 < mc.t_junction_std < 10, f"Tj_std={mc.t_junction_std}"
        
        # Paper: Thermal compliance = 100%
        assert mc.thermal_compliance_prob > 0.99
        
        # Paper: BDV = 55 ± 3 kV, compliance > 99.9%
        assert mc.bdv_compliance_prob > 0.95
        
        # Paper: Resistivity compliance = 100%
        assert mc.resistivity_compliance_prob > 0.99
        
        # Paper: Joint compliance = 97.6%
        assert mc.joint_compliance_prob > 0.90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
