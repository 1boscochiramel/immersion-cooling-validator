"""
FastAPI Backend for Immersion Cooling Validator Web App.

Provides REST API endpoints for:
- Running Monte Carlo simulations
- OCP compliance checking
- Economic analysis
- Sensitivity analysis
- Report generation

Run with: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import uuid
import os
import sys

# Add parent directory to path for icv imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from icv import (
    GroupIIIOil,
    FluorinertFC77,
    ImmersionCoolingValidator,
    validate_fluid,
    check_ocp_compliance,
    calculate_forex_savings,
    GPUSpecification,
)
from icv.sensitivity import run_sensitivity_analysis
from icv.report import generate_html_report, save_html_report


# ============================================================================
# FastAPI App Configuration
# ============================================================================

app = FastAPI(
    title="Immersion Cooling Validator API",
    description="Monte Carlo validation framework for immersion cooling fluids",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store for running simulations
simulations: Dict[str, Dict] = {}


# ============================================================================
# Pydantic Models
# ============================================================================

class FluidConfig(BaseModel):
    """Fluid configuration parameters."""
    
    name: str = Field(default="Group III Base Oil", description="Fluid name")
    viscosity_grade: str = Field(default="4cSt", description="Viscosity grade: 4cSt, 6cSt, 8cSt")
    cost_per_liter: float = Field(default=10.0, ge=0, description="Cost per liter in USD")
    
    # Optional overrides for fluid properties
    density_kg_m3: Optional[float] = Field(default=None, ge=700, le=1000)
    breakdown_voltage_kV: Optional[float] = Field(default=None, ge=20, le=100)
    volume_resistivity_ohm_cm: Optional[float] = Field(default=None, ge=1e10, le=1e16)


class GPUConfig(BaseModel):
    """GPU configuration parameters."""
    
    name: str = Field(default="NVIDIA H100", description="GPU model name")
    tdp_watts: float = Field(default=700.0, ge=100, le=2000, description="Thermal design power")
    r_jc_K_W: float = Field(default=0.018, ge=0.001, le=0.1, description="Junction-to-case resistance")
    r_cs_K_W: float = Field(default=0.005, ge=0.001, le=0.05, description="Case-to-surface resistance")


class SimulationConfig(BaseModel):
    """Monte Carlo simulation configuration."""
    
    n_samples: int = Field(default=10000, ge=100, le=100000, description="Number of MC samples")
    random_seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")
    include_economics: bool = Field(default=True, description="Include economic analysis")
    include_benchmark: bool = Field(default=True, description="Include FC-77 benchmark comparison")
    
    # Fluid and GPU configs
    fluid: FluidConfig = Field(default_factory=FluidConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)


class ComplianceCheckRequest(BaseModel):
    """Request for OCP compliance check."""
    
    junction_temp_C: float = Field(..., description="Junction temperature in Celsius")
    breakdown_voltage_kV: float = Field(..., description="Breakdown voltage in kV")
    volume_resistivity_ohm_cm: float = Field(..., description="Volume resistivity in Ω·cm")
    p5_life_years: float = Field(..., description="P5 service life in years")
    dielectric_constant: float = Field(default=2.15, description="Dielectric constant")
    loss_tangent: float = Field(default=1.5e-4, description="Loss tangent")


class ForexRequest(BaseModel):
    """Request for forex savings projection."""
    
    start_year: int = Field(default=2025, ge=2024, le=2030)
    end_year: int = Field(default=2030, ge=2025, le=2040)
    initial_market_MW: float = Field(default=100.0, ge=10, le=10000)
    cagr: float = Field(default=0.30, ge=0, le=1.0, description="Compound annual growth rate")
    adoption_rate: float = Field(default=0.5, ge=0, le=1.0, description="Domestic fluid adoption rate")


class SimulationStatus(BaseModel):
    """Status of a running simulation."""
    
    id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float  # 0-100
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    return """
    <html>
        <head>
            <title>ICV API</title>
            <style>
                body { font-family: system-ui; max-width: 800px; margin: 50px auto; padding: 20px; }
                h1 { color: #3b82f6; }
                a { color: #3b82f6; }
                code { background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }
            </style>
        </head>
        <body>
            <h1>🌡️ Immersion Cooling Validator API</h1>
            <p>Monte Carlo validation framework for immersion cooling fluids.</p>
            <h2>Quick Links</h2>
            <ul>
                <li><a href="/docs">📚 Interactive API Documentation (Swagger)</a></li>
                <li><a href="/redoc">📖 API Reference (ReDoc)</a></li>
                <li><a href="/health">💚 Health Check</a></li>
            </ul>
            <h2>Key Endpoints</h2>
            <ul>
                <li><code>POST /api/simulate</code> - Run Monte Carlo simulation</li>
                <li><code>POST /api/compliance</code> - Check OCP compliance</li>
                <li><code>POST /api/economics/forex</code> - Calculate forex savings</li>
                <li><code>GET /api/sensitivity</code> - Run sensitivity analysis</li>
            </ul>
        </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0",
    }


@app.post("/api/simulate", response_model=SimulationStatus)
async def run_simulation(
    config: SimulationConfig,
    background_tasks: BackgroundTasks,
):
    """
    Start a Monte Carlo simulation.
    
    Returns immediately with simulation ID. Use /api/simulate/{id} to check status.
    """
    sim_id = str(uuid.uuid4())[:8]
    
    simulations[sim_id] = {
        "id": sim_id,
        "status": "pending",
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "result": None,
        "config": config.dict(),
    }
    
    # Run simulation in background
    background_tasks.add_task(execute_simulation, sim_id, config)
    
    return SimulationStatus(**simulations[sim_id])


async def execute_simulation(sim_id: str, config: SimulationConfig):
    """Execute simulation in background."""
    try:
        simulations[sim_id]["status"] = "running"
        simulations[sim_id]["progress"] = 10
        
        # Create fluid
        fluid = GroupIIIOil(
            name=config.fluid.name,
            viscosity_grade=config.fluid.viscosity_grade,
        )
        
        # Create GPU spec
        gpu = GPUSpecification(
            name=config.gpu.name,
            tdp_watts=config.gpu.tdp_watts,
            r_jc_K_W=(config.gpu.r_jc_K_W, 0.002),
            r_cs_K_W=(config.gpu.r_cs_K_W, 0.001),
        )
        
        simulations[sim_id]["progress"] = 20
        
        # Run validation
        validator = ImmersionCoolingValidator(fluid, gpu=gpu)
        result = validator.run_full_validation(
            n_samples=config.n_samples,
            random_seed=config.random_seed,
            include_economics=config.include_economics,
            include_benchmark=config.include_benchmark,
        )
        
        simulations[sim_id]["progress"] = 90
        
        # Extract results for JSON response
        mc = result.monte_carlo
        
        result_dict = {
            "fluid_name": result.fluid_name,
            "timestamp": result.timestamp,
            "monte_carlo": {
                "n_samples": mc.n_samples,
                "junction_temperature": {
                    "mean": float(mc.t_junction_mean),
                    "std": float(mc.t_junction_std),
                    "percentiles": {
                        "p5": float(mc.t_junction_percentiles.get(5, 0)),
                        "p50": float(mc.t_junction_percentiles.get(50, 0)),
                        "p95": float(mc.t_junction_percentiles.get(95, 0)),
                    },
                },
                "breakdown_voltage": {
                    "mean": float(mc.bdv_mean),
                    "std": float(mc.bdv_std),
                },
                "resistivity_mean": float(mc.resistivity_mean),
                "p5_life_years": float(mc.p5_life_mean),
                "compliance": {
                    "thermal": float(mc.thermal_compliance_prob),
                    "bdv": float(mc.bdv_compliance_prob),
                    "resistivity": float(mc.resistivity_compliance_prob),
                    "lifetime": float(mc.lifetime_compliance_prob),
                    "joint": float(mc.joint_compliance_prob),
                },
            },
            "verdict": "PASS" if result.compliance.critical_pass else "FAIL",
        }
        
        # Add economics if included
        if result.economics:
            result_dict["economics"] = {
                "domestic_tco": float(result.economics.tco_a_usd),
                "imported_tco": float(result.economics.tco_b_usd),
                "savings_usd": float(result.economics.absolute_savings_usd),
                "savings_percent": float(result.economics.relative_savings_percent),
            }
        
        # Add benchmark comparison if included
        if result.benchmark_comparison:
            result_dict["benchmark"] = result.benchmark_comparison
        
        # Add histogram data for frontend
        if mc.samples:
            result_dict["histograms"] = {
                "junction_temperature": _compute_histogram(mc.samples["t_junction_C"]),
                "breakdown_voltage": _compute_histogram(mc.samples["breakdown_voltage_kV"]),
            }
        
        simulations[sim_id]["result"] = result_dict
        simulations[sim_id]["status"] = "completed"
        simulations[sim_id]["progress"] = 100
        simulations[sim_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        simulations[sim_id]["status"] = "failed"
        simulations[sim_id]["error"] = str(e)


def _compute_histogram(data, n_bins: int = 50) -> Dict:
    """Compute histogram data for frontend visualization."""
    import numpy as np
    counts, bin_edges = np.histogram(data, bins=n_bins)
    return {
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist(),
        "bin_centers": ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist(),
    }


@app.get("/api/simulate/{sim_id}", response_model=SimulationStatus)
async def get_simulation_status(sim_id: str):
    """Get status of a running or completed simulation."""
    if sim_id not in simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    return SimulationStatus(**simulations[sim_id])


@app.post("/api/simulate/quick")
async def quick_simulation(
    n_samples: int = 1000,
    viscosity_grade: str = "4cSt",
    tdp_watts: float = 700.0,
):
    """
    Run a quick synchronous simulation (blocking).
    
    For small sample sizes only. Use /api/simulate for larger runs.
    """
    if n_samples > 5000:
        raise HTTPException(
            status_code=400,
            detail="For n_samples > 5000, use async endpoint /api/simulate"
        )
    
    fluid = GroupIIIOil(viscosity_grade=viscosity_grade)
    gpu = GPUSpecification(tdp_watts=tdp_watts)
    
    validator = ImmersionCoolingValidator(fluid, gpu=gpu)
    result = validator.run_full_validation(
        n_samples=n_samples,
        include_economics=True,
        include_benchmark=False,
    )
    
    mc = result.monte_carlo
    
    return {
        "fluid_name": result.fluid_name,
        "junction_temperature": {
            "mean": float(mc.t_junction_mean),
            "std": float(mc.t_junction_std),
        },
        "breakdown_voltage_mean": float(mc.bdv_mean),
        "joint_compliance": float(mc.joint_compliance_prob),
        "verdict": "PASS" if result.compliance.critical_pass else "FAIL",
    }


@app.post("/api/compliance")
async def check_compliance(request: ComplianceCheckRequest):
    """Check OCP compliance for given parameter values."""
    
    result = check_ocp_compliance(
        junction_temp_C=request.junction_temp_C,
        breakdown_voltage_kV=request.breakdown_voltage_kV,
        volume_resistivity_ohm_cm=request.volume_resistivity_ohm_cm,
        p5_life_years=request.p5_life_years,
        dielectric_constant=request.dielectric_constant,
        loss_tangent=request.loss_tangent,
    )
    
    return {
        "all_pass": result.all_pass,
        "critical_pass": result.critical_pass,
        "n_pass": result.n_pass,
        "n_fail": result.n_fail,
        "n_marginal": result.n_marginal,
        "details": {
            name: {
                "status": res.status.value,
                "value": float(res.value),
                "margin_percent": float(res.margin_percent),
            }
            for name, res in result.results.items()
        },
    }


@app.post("/api/economics/forex")
async def calculate_forex(request: ForexRequest):
    """Calculate forex savings projection."""
    
    projection = calculate_forex_savings(
        start_year=request.start_year,
        end_year=request.end_year,
        initial_market_MW=request.initial_market_MW,
        cagr=request.cagr,
        adoption_rate=request.adoption_rate,
    )
    
    return {
        "years": projection.years,
        "annual_savings_usd": [float(s) for s in projection.annual_savings_usd],
        "cumulative_savings_usd": [float(s) for s in projection.cumulative_savings_usd],
        "total_savings_usd": float(projection.total_savings_usd),
        "total_savings_inr_crores": float(projection.total_savings_inr_crores),
    }


@app.get("/api/sensitivity")
async def get_sensitivity_analysis(n_samples: int = 512):
    """
    Run Sobol sensitivity analysis.
    
    Returns sensitivity indices showing which parameters most affect outputs.
    """
    if n_samples > 5000:
        raise HTTPException(
            status_code=400,
            detail="n_samples must be <= 5000 for real-time analysis"
        )
    
    # Run a quick MC simulation to get samples for sensitivity analysis
    fluid = GroupIIIOil()
    validator = ImmersionCoolingValidator(fluid)
    mc_result = validator.run_monte_carlo(n_samples=n_samples)
    
    # Run sensitivity analysis on the MC results
    sensitivity = run_sensitivity_analysis(mc_result)
    
    return {
        "method": "sobol_approximation",
        "total_samples": n_samples,
        "junction_temperature": {
            "first_order": sensitivity.junction_temperature.first_order,
            "total_order": sensitivity.junction_temperature.total_order,
        },
        "parameter_ranking": [
            {"name": name, "total_order": val}
            for name, val in sensitivity.junction_temperature.most_influential
        ],
    }


@app.get("/api/fluids")
async def list_fluids():
    """List available fluid configurations."""
    return {
        "fluids": [
            {
                "id": "group_iii_4cst",
                "name": "Group III Base Oil (4cSt)",
                "type": "domestic",
                "cost_per_liter": 10.0,
                "viscosity_grade": "4cSt",
            },
            {
                "id": "group_iii_6cst",
                "name": "Group III Base Oil (6cSt)",
                "type": "domestic",
                "cost_per_liter": 10.0,
                "viscosity_grade": "6cSt",
            },
            {
                "id": "group_iii_8cst",
                "name": "Group III Base Oil (8cSt)",
                "type": "domestic",
                "cost_per_liter": 10.0,
                "viscosity_grade": "8cSt",
            },
            {
                "id": "fluorinert_fc77",
                "name": "Fluorinert FC-77",
                "type": "imported",
                "cost_per_liter": 400.0,
                "viscosity_grade": None,
            },
        ]
    }


@app.get("/api/gpus")
async def list_gpus():
    """List available GPU configurations."""
    return {
        "gpus": [
            {
                "id": "h100",
                "name": "NVIDIA H100",
                "tdp_watts": 700,
                "architecture": "Hopper",
            },
            {
                "id": "h200",
                "name": "NVIDIA H200",
                "tdp_watts": 700,
                "architecture": "Hopper",
            },
            {
                "id": "a100_80gb",
                "name": "NVIDIA A100 (80GB)",
                "tdp_watts": 400,
                "architecture": "Ampere",
            },
            {
                "id": "l40s",
                "name": "NVIDIA L40S",
                "tdp_watts": 350,
                "architecture": "Ada Lovelace",
            },
            {
                "id": "mi300x",
                "name": "AMD Instinct MI300X",
                "tdp_watts": 750,
                "architecture": "CDNA 3",
            },
        ]
    }


@app.get("/api/ocp-requirements")
async def get_ocp_requirements():
    """Get OCP specification requirements."""
    return {
        "requirements": [
            {
                "name": "Junction Temperature",
                "limit": "< 88°C",
                "unit": "°C",
                "critical": True,
            },
            {
                "name": "Breakdown Voltage",
                "limit": "> 45 kV",
                "unit": "kV",
                "critical": True,
            },
            {
                "name": "Volume Resistivity",
                "limit": "> 10¹¹ Ω·cm",
                "unit": "Ω·cm",
                "critical": True,
            },
            {
                "name": "P5 Service Life",
                "limit": "> 5 years",
                "unit": "years",
                "critical": True,
            },
            {
                "name": "Dielectric Constant",
                "limit": "1.8 - 2.5",
                "unit": "",
                "critical": False,
            },
            {
                "name": "Loss Tangent",
                "limit": "< 5×10⁻⁴",
                "unit": "",
                "critical": False,
            },
        ]
    }


# ============================================================================
# Report Generation
# ============================================================================

@app.post("/api/report/html/{sim_id}")
async def generate_report(sim_id: str):
    """Generate HTML report for a completed simulation."""
    if sim_id not in simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    sim = simulations[sim_id]
    if sim["status"] != "completed":
        raise HTTPException(status_code=400, detail="Simulation not completed")
    
    # For now, return a simple HTML summary
    # Full report generation would require storing the ValidationResult object
    result = sim["result"]
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ICV Report - {result['fluid_name']}</title>
        <style>
            body {{ font-family: system-ui; max-width: 900px; margin: 50px auto; padding: 20px; }}
            h1 {{ color: #3b82f6; }}
            .metric {{ background: #f1f5f9; padding: 15px; border-radius: 8px; margin: 10px 0; }}
            .pass {{ color: #22c55e; }}
            .fail {{ color: #ef4444; }}
        </style>
    </head>
    <body>
        <h1>🌡️ Immersion Cooling Validation Report</h1>
        <p>Fluid: {result['fluid_name']}</p>
        <p>Generated: {result['timestamp']}</p>
        
        <h2>Monte Carlo Results (N={result['monte_carlo']['n_samples']:,})</h2>
        
        <div class="metric">
            <h3>Junction Temperature</h3>
            <p>Mean: {result['monte_carlo']['junction_temperature']['mean']:.1f}°C</p>
            <p>Std: {result['monte_carlo']['junction_temperature']['std']:.1f}°C</p>
            <p>P95: {result['monte_carlo']['junction_temperature']['percentiles']['p95']:.1f}°C</p>
        </div>
        
        <div class="metric">
            <h3>Breakdown Voltage</h3>
            <p>Mean: {result['monte_carlo']['breakdown_voltage']['mean']:.1f} kV</p>
        </div>
        
        <div class="metric">
            <h3>Compliance Probabilities</h3>
            <p>Thermal: {result['monte_carlo']['compliance']['thermal']*100:.1f}%</p>
            <p>BDV: {result['monte_carlo']['compliance']['bdv']*100:.1f}%</p>
            <p>Resistivity: {result['monte_carlo']['compliance']['resistivity']*100:.1f}%</p>
            <p>Lifetime: {result['monte_carlo']['compliance']['lifetime']*100:.1f}%</p>
            <p><strong>Joint: {result['monte_carlo']['compliance']['joint']*100:.1f}%</strong></p>
        </div>
        
        <h2>Verdict: <span class="{'pass' if result['verdict'] == 'PASS' else 'fail'}">{result['verdict']}</span></h2>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html)


# ============================================================================
# Run server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
