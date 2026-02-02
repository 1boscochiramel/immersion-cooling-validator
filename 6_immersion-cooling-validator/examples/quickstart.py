#!/usr/bin/env python3
"""
ICV Quickstart Example
======================
Demonstrates core functionality in 20 lines.
"""

from icv import validate_fluid, GroupIIIOil, check_ocp_compliance

# 1. Quick validation (one line)
result = validate_fluid(n_samples=5000)
print(f"Verdict: {result.compliance.critical_pass} | Compliance: {result.monte_carlo.joint_compliance_prob:.1%}")

# 2. Custom fluid
fluid = GroupIIIOil(viscosity_grade="6cSt")
result = validate_fluid(fluid=fluid, n_samples=5000)
print(f"6cSt Fluid - Tj: {result.monte_carlo.t_junction_mean:.1f}°C")

# 3. Direct compliance check
compliance = check_ocp_compliance(
    junction_temp_C=65.0,
    breakdown_voltage_kV=50.0,
    volume_resistivity_ohm_cm=5e12,
    p5_life_years=45.0,
)
print(f"Direct Check: {'PASS' if compliance.critical_pass else 'FAIL'}")

# 4. Cost savings
if result.economics:
    print(f"Cost Savings: {result.economics.relative_savings_percent:.0f}%")
