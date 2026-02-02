#!/usr/bin/env python3
"""
Example: Running a complete immersion cooling fluid validation.

This script demonstrates how to use the ICV library to validate
Group III hydrocarbon oils against OCP specifications.

Usage:
    python examples/basic_validation.py
"""

from icv import (
    GroupIIIOil,
    FluorinertFC77,
    ImmersionCoolingValidator,
    validate_fluid,
    run_simulation,
    check_ocp_compliance,
    calculate_forex_savings,
)


def main():
    print("=" * 70)
    print("IMMERSION COOLING VALIDATOR - EXAMPLE")
    print("=" * 70)
    print()
    
    # =========================================================================
    # Example 1: Quick validation with convenience function
    # =========================================================================
    print("1. QUICK VALIDATION")
    print("-" * 50)
    
    result = validate_fluid(n_samples=10_000)
    
    print(f"   Fluid: {result.fluid_name}")
    print(f"   Junction Temperature: {result.monte_carlo.t_junction_mean:.1f} ± "
          f"{result.monte_carlo.t_junction_std:.1f}°C")
    print(f"   Joint Compliance: {100*result.monte_carlo.joint_compliance_prob:.1f}%")
    print(f"   Verdict: {'✅ PASS' if result.compliance.critical_pass else '❌ FAIL'}")
    print()
    
    # =========================================================================
    # Example 2: Custom fluid configuration
    # =========================================================================
    print("2. CUSTOM FLUID CONFIGURATION")
    print("-" * 50)
    
    # Create fluids with different viscosity grades
    oil_4cst = GroupIIIOil(name="HPCL 4cSt", viscosity_grade="4cSt")
    oil_6cst = GroupIIIOil(name="IOCL 6cSt", viscosity_grade="6cSt")
    oil_8cst = GroupIIIOil(name="BPCL 8cSt", viscosity_grade="8cSt")
    
    for fluid in [oil_4cst, oil_6cst, oil_8cst]:
        result = run_simulation(fluid=fluid, n_samples=1000)
        print(f"   {fluid.name}: Tj={result.t_junction_mean:.1f}°C, "
              f"Compliance={100*result.joint_compliance_prob:.1f}%")
    print()
    
    # =========================================================================
    # Example 3: Detailed compliance checking
    # =========================================================================
    print("3. OCP COMPLIANCE CHECK")
    print("-" * 50)
    
    compliance = check_ocp_compliance(
        junction_temp_C=59.0,
        breakdown_voltage_kV=55.0,
        volume_resistivity_ohm_cm=1.5e13,
        p5_life_years=5.8,
    )
    
    for name, res in compliance.results.items():
        status = "✅" if res.status.value == "PASS" else "❌"
        print(f"   {status} {res.requirement.name}: {res.value:.4g} "
              f"(margin: {res.margin_percent:+.1f}%)")
    
    print(f"\n   Overall: {'✅ ALL PASS' if compliance.all_pass else '❌ SOME FAILED'}")
    print()
    
    # =========================================================================
    # Example 4: Economic analysis
    # =========================================================================
    print("4. ECONOMIC ANALYSIS")
    print("-" * 50)
    
    from icv.economics import EconomicAnalysis, FluidCost, DataCenterConfig
    
    domestic = FluidCost(
        name="Group III Oil",
        cost_per_liter_usd=10.0,
        is_domestic=True,
    )
    
    imported = FluidCost(
        name="Fluorinert FC-77",
        cost_per_liter_usd=400.0,
        is_domestic=False,
        import_duty_percent=10.0,
    )
    
    dc_config = DataCenterConfig(power_capacity_MW=10.0)
    
    analysis = EconomicAnalysis(domestic, imported, dc_config)
    comparison = analysis.compare_fluids()
    
    print(f"   Data Center: {dc_config.power_capacity_MW} MW")
    print(f"   Group III TCO: ${comparison.tco_a_usd:,.0f}")
    print(f"   FC-77 TCO: ${comparison.tco_b_usd:,.0f}")
    print(f"   💰 Savings: ${comparison.absolute_savings_usd:,.0f} "
          f"({comparison.relative_savings_percent:.0f}%)")
    print()
    
    # =========================================================================
    # Example 5: Forex savings projection
    # =========================================================================
    print("5. FOREX SAVINGS PROJECTION (2025-2030)")
    print("-" * 50)
    
    forex = calculate_forex_savings(
        start_year=2025,
        end_year=2030,
        initial_market_MW=100.0,
        adoption_rate=0.5,
    )
    
    print(f"   {'Year':<6} {'Annual':>15} {'Cumulative':>15}")
    print(f"   {'-'*6} {'-'*15} {'-'*15}")
    for i, year in enumerate(forex.years):
        print(f"   {year:<6} ${forex.annual_savings_usd[i]:>13,.0f} "
              f"${forex.cumulative_savings_usd[i]:>13,.0f}")
    
    print(f"\n   Total: ${forex.total_savings_usd:,.0f} USD")
    print(f"          ₹{forex.total_savings_inr_crores:.0f} Crores")
    print()
    
    # =========================================================================
    # Example 6: Full validation with benchmark comparison
    # =========================================================================
    print("6. FULL VALIDATION WITH BENCHMARK")
    print("-" * 50)
    
    fluid = GroupIIIOil()
    validator = ImmersionCoolingValidator(fluid)
    
    result = validator.run_full_validation(
        n_samples=10_000,
        random_seed=42,
        include_economics=True,
        include_benchmark=True,
    )
    
    # Print benchmark comparison
    if result.benchmark_comparison:
        for metric, data in result.benchmark_comparison.items():
            print(f"   {metric}:")
            print(f"      Group III: {data['test']:.4g}")
            print(f"      FC-77:     {data['benchmark']:.4g}")
            print(f"      Advantage: {data['advantage']}")
            print()
    
    # =========================================================================
    # Example 7: Export results
    # =========================================================================
    print("7. EXPORT RESULTS")
    print("-" * 50)
    
    # Export to JSON
    json_output = result.to_json()
    print(f"   JSON output length: {len(json_output)} characters")
    
    # Save to file
    result.save("validation_results.json")
    print("   Saved to: validation_results.json")
    print()
    
    # =========================================================================
    # Print full summary
    # =========================================================================
    print("=" * 70)
    print("FULL VALIDATION REPORT")
    print("=" * 70)
    print(result.summary())


if __name__ == "__main__":
    main()
