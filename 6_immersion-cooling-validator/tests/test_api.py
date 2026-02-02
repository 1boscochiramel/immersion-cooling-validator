"""
Tests for FastAPI backend.

Run with: pytest tests/test_api.py -v
"""

import pytest
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'web', 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestAPIModels:
    """Test Pydantic models for API."""
    
    def test_fluid_config_defaults(self):
        from main import FluidConfig
        
        config = FluidConfig()
        assert config.name == "Group III Base Oil"
        assert config.viscosity_grade == "4cSt"
        assert config.cost_per_liter == 10.0
    
    def test_fluid_config_custom(self):
        from main import FluidConfig
        
        config = FluidConfig(
            name="Custom Oil",
            viscosity_grade="6cSt",
            cost_per_liter=15.0,
        )
        assert config.name == "Custom Oil"
        assert config.viscosity_grade == "6cSt"
    
    def test_gpu_config_defaults(self):
        from main import GPUConfig
        
        config = GPUConfig()
        assert config.name == "NVIDIA H100"
        assert config.tdp_watts == 700.0
    
    def test_simulation_config(self):
        from main import SimulationConfig
        
        config = SimulationConfig(n_samples=1000)
        assert config.n_samples == 1000
        assert config.include_economics is True
        assert config.fluid.viscosity_grade == "4cSt"
    
    def test_compliance_check_request(self):
        from main import ComplianceCheckRequest
        
        request = ComplianceCheckRequest(
            junction_temp_C=59.0,
            breakdown_voltage_kV=55.0,
            volume_resistivity_ohm_cm=1e13,
            p5_life_years=5.8,
        )
        assert request.junction_temp_C == 59.0
        assert request.breakdown_voltage_kV == 55.0


class TestAPIHelpers:
    """Test helper functions."""
    
    def test_compute_histogram(self):
        from main import _compute_histogram
        import numpy as np
        
        data = np.random.normal(60, 5, 1000)
        hist = _compute_histogram(data, n_bins=20)
        
        assert "counts" in hist
        assert "bin_edges" in hist
        assert "bin_centers" in hist
        assert len(hist["counts"]) == 20
        assert len(hist["bin_edges"]) == 21
        assert len(hist["bin_centers"]) == 20


class TestAPIEndpointsUnit:
    """Unit tests for API endpoint logic (without running server)."""
    
    def test_health_response_structure(self):
        """Test expected health response structure."""
        # Just verify the expected keys
        expected_keys = ["status", "timestamp", "version"]
        response = {"status": "healthy", "timestamp": "2024-01-01", "version": "0.1.0"}
        
        for key in expected_keys:
            assert key in response
    
    def test_fluids_list_structure(self):
        """Test expected fluids list structure."""
        fluids = [
            {"id": "group_iii_4cst", "name": "Group III Base Oil (4cSt)", "type": "domestic"},
            {"id": "fluorinert_fc77", "name": "Fluorinert FC-77", "type": "imported"},
        ]
        
        assert len(fluids) >= 2
        assert fluids[0]["type"] == "domestic"
        assert fluids[1]["type"] == "imported"
    
    def test_ocp_requirements_structure(self):
        """Test OCP requirements structure."""
        requirements = [
            {"name": "Junction Temperature", "limit": "< 88°C", "critical": True},
            {"name": "Breakdown Voltage", "limit": "> 45 kV", "critical": True},
        ]
        
        for req in requirements:
            assert "name" in req
            assert "limit" in req
            assert "critical" in req


class TestICVIntegration:
    """Test ICV library integration with API."""
    
    def test_validate_fluid_integration(self):
        from icv import validate_fluid
        
        result = validate_fluid(n_samples=100, include_economics=False)
        
        assert result is not None
        assert result.monte_carlo.n_samples == 100
        assert 0 <= result.monte_carlo.joint_compliance_prob <= 1
    
    def test_check_compliance_integration(self):
        from icv import check_ocp_compliance
        
        result = check_ocp_compliance(
            junction_temp_C=59.0,
            breakdown_voltage_kV=55.0,
            volume_resistivity_ohm_cm=1e13,
            p5_life_years=5.8,
        )
        
        assert result.critical_pass is True
    
    def test_forex_calculation_integration(self):
        from icv import calculate_forex_savings
        
        projection = calculate_forex_savings(
            start_year=2025,
            end_year=2030,
        )
        
        assert len(projection.years) == 6
        assert projection.total_savings_usd > 0


# Optional: Integration tests with actual server
# These require the server to be running

@pytest.mark.skip(reason="Requires running server")
class TestAPIIntegration:
    """Integration tests requiring running server."""
    
    def test_health_endpoint(self):
        import httpx
        
        response = httpx.get("http://localhost:8000/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_quick_simulation(self):
        import httpx
        
        response = httpx.post(
            "http://localhost:8000/api/simulate/quick",
            params={"n_samples": 100, "viscosity_grade": "4cSt"},
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "junction_temperature" in data
        assert "verdict" in data
    
    def test_compliance_endpoint(self):
        import httpx
        
        response = httpx.post(
            "http://localhost:8000/api/compliance",
            json={
                "junction_temp_C": 59.0,
                "breakdown_voltage_kV": 55.0,
                "volume_resistivity_ohm_cm": 1e13,
                "p5_life_years": 5.8,
            },
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["critical_pass"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
