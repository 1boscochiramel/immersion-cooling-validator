"""
Test suite for Session 2 features: Sensitivity, Visualization, Reports.

Run with: pytest tests/test_session2.py -v
"""

import pytest
import numpy as np


class TestSensitivity:
    """Test sensitivity analysis module."""
    
    def test_sobol_indices_creation(self):
        from icv.sensitivity import SobolIndices
        
        indices = SobolIndices(
            parameter_names=["param1", "param2", "param3"],
            first_order={"param1": 0.3, "param2": 0.2, "param3": 0.1},
            total_order={"param1": 0.35, "param2": 0.25, "param3": 0.15},
        )
        
        assert len(indices.parameter_names) == 3
        assert indices.first_order["param1"] == 0.3
        assert indices.total_order["param1"] == 0.35
    
    def test_sobol_most_influential(self):
        from icv.sensitivity import SobolIndices
        
        indices = SobolIndices(
            parameter_names=["a", "b", "c"],
            first_order={"a": 0.1, "b": 0.5, "c": 0.2},
            total_order={"a": 0.15, "b": 0.55, "c": 0.25},
        )
        
        most_influential = indices.most_influential
        assert most_influential[0][0] == "b"  # b has highest ST
        assert most_influential[0][1] == 0.55
    
    def test_sobol_summary(self):
        from icv.sensitivity import SobolIndices
        
        indices = SobolIndices(
            parameter_names=["temp", "power"],
            first_order={"temp": 0.3, "power": 0.4},
            total_order={"temp": 0.35, "power": 0.45},
        )
        
        summary = indices.summary()
        assert "SOBOL SENSITIVITY INDICES" in summary
        assert "temp" in summary
        assert "power" in summary
    
    def test_correlation_computation(self):
        from icv.sensitivity import compute_correlations
        import numpy as np
        
        # Create correlated data
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = 2 * x + np.random.normal(0, 0.1, 100)  # Strong positive correlation
        
        samples = {"input_x": x}
        outputs = {"output_y": y}
        
        correlations = compute_correlations(samples, outputs)
        
        assert "output_y" in correlations
        assert "input_x" in correlations["output_y"]
        assert correlations["output_y"]["input_x"] > 0.9  # Strong correlation
    
    def test_rank_parameters(self):
        from icv.sensitivity import rank_parameters_by_influence
        import numpy as np
        
        np.random.seed(42)
        n = 100
        
        # Create mock MC samples
        mc_samples = {
            "important": np.random.normal(0, 1, n),
            "less_important": np.random.normal(0, 1, n),
        }
        
        # Output highly correlated with "important"
        mc_outputs = {
            "t_junction_C": 60 + 5 * mc_samples["important"] + 0.5 * mc_samples["less_important"]
        }
        
        rankings = rank_parameters_by_influence(mc_samples, mc_outputs)
        
        assert len(rankings) == 2
        assert rankings[0][0] == "important"  # Should be ranked first


class TestReport:
    """Test report generation module."""
    
    def test_html_report_generation(self):
        from icv import validate_fluid, generate_html_report
        
        result = validate_fluid(n_samples=100, include_economics=True)
        html = generate_html_report(result)
        
        assert "<!DOCTYPE html>" in html
        assert "Executive Summary" in html
        assert "Thermal Performance" in html
        assert "OCP Compliance" in html
    
    def test_html_report_contains_metrics(self):
        from icv import validate_fluid, generate_html_report
        
        result = validate_fluid(n_samples=100, include_economics=False)
        html = generate_html_report(result)
        
        # Should contain key metrics
        assert "Junction Temperature" in html
        assert "Breakdown Voltage" in html
        assert "Joint Compliance" in html
    
    def test_report_config(self):
        from icv.report import ReportConfig
        
        config = ReportConfig(
            include_executive_summary=True,
            include_economics=False,
            company_name="Test Corp",
        )
        
        assert config.include_executive_summary is True
        assert config.include_economics is False
        assert config.company_name == "Test Corp"
    
    def test_save_html_report(self, tmp_path):
        from icv import validate_fluid
        from icv.report import save_html_report
        
        result = validate_fluid(n_samples=100, include_economics=False)
        filepath = tmp_path / "test_report.html"
        
        save_html_report(result, str(filepath))
        
        assert filepath.exists()
        content = filepath.read_text()
        assert "<!DOCTYPE html>" in content


class TestVisualization:
    """Test visualization module (without actually rendering plots)."""
    
    def test_matplotlib_check(self):
        from icv.visualization import HAS_MATPLOTLIB
        # Just verify the check works
        assert isinstance(HAS_MATPLOTLIB, bool)
    
    def test_colors_defined(self):
        from icv.visualization import COLORS
        
        assert "pass" in COLORS
        assert "fail" in COLORS
        assert "primary" in COLORS


class TestExtendedLifetime:
    """Test extended lifetime model features."""
    
    def test_lifetime_temperature_acceleration(self):
        from icv.lifetime import LifetimeModel
        
        model = LifetimeModel()
        
        # At reference temperature, AF should be ~1
        af_ref = model.temperature_acceleration_factor(90.0, 90.0)
        assert 0.99 < af_ref < 1.01
        
        # Higher temp should accelerate (AF > 1)
        af_high = model.temperature_acceleration_factor(100.0, 90.0)
        assert af_high > 1.0
        
        # Lower temp should decelerate (AF < 1)
        af_low = model.temperature_acceleration_factor(70.0, 90.0)
        assert af_low < 1.0
    
    def test_lifetime_adjusted_life(self):
        from icv.lifetime import LifetimeModel
        
        model = LifetimeModel()
        
        # At lower operating temp, life should be extended
        base_p5 = model.p5_life
        adjusted_p5 = model.adjusted_life(5, 60.0, 90.0)
        
        assert adjusted_p5 > base_p5  # Lower temp = longer life
    
    def test_lifetime_sampling(self):
        from icv.lifetime import LifetimeModel
        import numpy as np
        
        model = LifetimeModel()
        rng = np.random.default_rng(42)
        
        samples = model.sample_lifetime(rng, n_samples=1000)
        
        assert len(samples) == 1000
        assert np.mean(samples) > 0
        assert np.all(samples > 0)  # All lifetimes positive


class TestExtendedElectrical:
    """Test extended electrical model features."""
    
    def test_electrical_model_aging(self):
        from icv.electrical import ElectricalModel
        
        model = ElectricalModel()
        
        # Properties should degrade with aging
        props_new = model.aged_properties(60.0, years_aged=0)
        props_5yr = model.aged_properties(60.0, years_aged=5)
        
        assert props_5yr["breakdown_voltage_kV"] < props_new["breakdown_voltage_kV"]
    
    def test_electrical_moisture_effect(self):
        from icv.electrical import calculate_breakdown_voltage
        
        # Higher moisture should reduce BDV
        bdv_dry = calculate_breakdown_voltage(60.0, moisture_ppm=50)
        bdv_wet = calculate_breakdown_voltage(60.0, moisture_ppm=200)
        
        assert bdv_wet < bdv_dry


class TestExtendedThermal:
    """Test extended thermal model features."""
    
    def test_heatsink_area_enhancement(self):
        from icv.thermal import HeatSinkGeometry
        
        heatsink = HeatSinkGeometry()
        
        # Enhanced design should have significant area increase
        assert heatsink.enhancement_factor > 3.0  # >340% per paper
    
    def test_gpu_specification(self):
        from icv.thermal import GPUSpecification
        
        gpu = GPUSpecification()
        
        assert gpu.name == "NVIDIA H100"
        assert gpu.tdp_watts == 700.0
        assert gpu.t_throttle_C == 88.0
        assert gpu.die_area_m2 > 0
        assert gpu.heat_flux_W_m2 > 0


class TestValidatorExtended:
    """Test extended validator features."""
    
    def test_validator_with_custom_gpu(self):
        from icv import GroupIIIOil, ImmersionCoolingValidator
        from icv.thermal import GPUSpecification
        
        # Custom lower-power GPU
        gpu = GPUSpecification(name="Custom GPU", tdp_watts=400)
        fluid = GroupIIIOil()
        
        validator = ImmersionCoolingValidator(fluid, gpu=gpu)
        result = validator.run_monte_carlo(n_samples=100)
        
        # Lower TDP should give lower junction temp
        assert result.t_junction_mean < 80
    
    def test_validator_benchmark_comparison(self):
        from icv import GroupIIIOil, ImmersionCoolingValidator
        
        fluid = GroupIIIOil()
        validator = ImmersionCoolingValidator(fluid)
        
        comparison = validator.run_benchmark_comparison(n_samples=100)
        
        assert "Junction Temperature (°C)" in comparison
        assert "Cost per Liter ($)" in comparison
        assert comparison["Cost per Liter ($)"]["test"] < comparison["Cost per Liter ($)"]["benchmark"]


class TestIntegrationSession2:
    """Integration tests for Session 2 complete workflow."""
    
    def test_full_workflow_with_report(self, tmp_path):
        """Test complete workflow: validate → generate report → save."""
        from icv import (
            GroupIIIOil,
            ImmersionCoolingValidator,
            generate_html_report,
        )
        from icv.report import save_html_report
        from icv.sensitivity import run_sensitivity_analysis
        
        # Create fluid and validator
        fluid = GroupIIIOil(name="Test Oil")
        validator = ImmersionCoolingValidator(fluid)
        
        # Run validation
        result = validator.run_full_validation(
            n_samples=500,
            include_economics=True,
            include_benchmark=False,
        )
        
        # Verify key metrics
        assert result.monte_carlo.t_junction_mean > 0
        assert result.monte_carlo.joint_compliance_prob > 0.9
        
        # Run sensitivity analysis
        sensitivity = run_sensitivity_analysis(result.monte_carlo)
        assert sensitivity.junction_temperature is not None
        
        # Generate and save report
        html = generate_html_report(result)
        assert len(html) > 5000  # Substantial content
        
        report_path = tmp_path / "validation_report.html"
        save_html_report(result, str(report_path))
        assert report_path.exists()
        
        # Save JSON results
        json_path = tmp_path / "results.json"
        result.save(str(json_path))
        assert json_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
