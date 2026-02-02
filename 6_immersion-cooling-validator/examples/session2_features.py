#!/usr/bin/env python3
"""
Example: Session 2 Features - Sensitivity Analysis & Reports

This script demonstrates:
- Sensitivity analysis (parameter importance)
- HTML report generation
- Visualization (if matplotlib installed)
- Extended validation options

Usage:
    python examples/session2_features.py
"""

from icv import (
    GroupIIIOil,
    FluorinertFC77,
    ImmersionCoolingValidator,
    validate_fluid,
    generate_html_report,
    GPUSpecification,
    HeatSinkGeometry,
)
from icv.sensitivity import (
    SobolIndices,
    run_sensitivity_analysis,
    rank_parameters_by_influence,
)
from icv.report import save_html_report, ReportConfig


def main():
    print("=" * 70)
    print("SESSION 2: SENSITIVITY ANALYSIS & REPORTING")
    print("=" * 70)
    print()
    
    # =========================================================================
    # 1. Run validation with full sample storage
    # =========================================================================
    print("1. RUNNING MONTE CARLO VALIDATION")
    print("-" * 50)
    
    fluid = GroupIIIOil(name="HPCL Group III 4cSt")
    validator = ImmersionCoolingValidator(fluid)
    
    result = validator.run_full_validation(
        n_samples=10_000,
        random_seed=42,
        include_economics=True,
        include_benchmark=True,
    )
    
    print(f"   ✅ Validation complete")
    print(f"   Junction Temperature: {result.monte_carlo.t_junction_mean:.1f} ± "
          f"{result.monte_carlo.t_junction_std:.1f}°C")
    print(f"   Joint Compliance: {result.monte_carlo.joint_compliance_prob*100:.1f}%")
    print()
    
    # =========================================================================
    # 2. Sensitivity Analysis
    # =========================================================================
    print("2. SENSITIVITY ANALYSIS")
    print("-" * 50)
    
    sensitivity = run_sensitivity_analysis(result.monte_carlo)
    
    print("   Sobol Sensitivity Indices for Junction Temperature:")
    print()
    print(f"   {'Parameter':<25} {'S1 (First-order)':>15} {'ST (Total)':>15}")
    print(f"   {'-'*25} {'-'*15} {'-'*15}")
    
    for name, st in sensitivity.junction_temperature.most_influential:
        s1 = sensitivity.junction_temperature.first_order.get(name, 0)
        print(f"   {name:<25} {s1:>15.3f} {st:>15.3f}")
    
    print()
    print("   Key Findings:")
    critical = [name for name, st in sensitivity.junction_temperature.most_influential if st > 0.1]
    print(f"   - {len(critical)} parameters with ST > 0.1 (significant)")
    print(f"   - Most influential: {sensitivity.junction_temperature.most_influential[0][0]}")
    print()
    
    # =========================================================================
    # 3. Custom GPU Configuration Analysis
    # =========================================================================
    print("3. CUSTOM GPU CONFIGURATION ANALYSIS")
    print("-" * 50)
    
    gpu_configs = [
        GPUSpecification(name="H100 (700W)", tdp_watts=700),
        GPUSpecification(name="A100 (400W)", tdp_watts=400),
        GPUSpecification(name="L40S (300W)", tdp_watts=300),
    ]
    
    print(f"   {'GPU':<20} {'Tj Mean':>12} {'Tj P95':>12} {'Compliance':>12}")
    print(f"   {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    
    for gpu in gpu_configs:
        v = ImmersionCoolingValidator(fluid, gpu=gpu)
        mc = v.run_monte_carlo(n_samples=1000)
        
        p95 = mc.t_junction_percentiles.get(95, 0)
        print(f"   {gpu.name:<20} {mc.t_junction_mean:>10.1f}°C {p95:>10.1f}°C "
              f"{mc.thermal_compliance_prob*100:>10.1f}%")
    
    print()
    
    # =========================================================================
    # 4. Benchmark Comparison Summary
    # =========================================================================
    print("4. BENCHMARK COMPARISON (vs Fluorinert FC-77)")
    print("-" * 50)
    
    if result.benchmark_comparison:
        for metric, data in result.benchmark_comparison.items():
            print(f"\n   {metric}:")
            print(f"      Group III:  {data['test']:.4g}")
            print(f"      FC-77:      {data['benchmark']:.4g}")
            print(f"      Advantage:  {data['advantage']}")
    
    print()
    
    # =========================================================================
    # 5. Generate HTML Report
    # =========================================================================
    print("5. HTML REPORT GENERATION")
    print("-" * 50)
    
    html_report = generate_html_report(result)
    
    report_path = "validation_report.html"
    save_html_report(result, report_path)
    
    print(f"   ✅ Report saved to: {report_path}")
    print(f"   Report size: {len(html_report):,} characters")
    print()
    
    # =========================================================================
    # 6. Export Results
    # =========================================================================
    print("6. EXPORT RESULTS")
    print("-" * 50)
    
    # JSON export
    json_path = "validation_results.json"
    result.save(json_path)
    print(f"   ✅ JSON results saved to: {json_path}")
    
    # Print summary of exported files
    print()
    print("   Exported files:")
    print(f"   - {report_path} (HTML report)")
    print(f"   - {json_path} (JSON data)")
    print()
    
    # =========================================================================
    # 7. Quick Parameter Sensitivity Check
    # =========================================================================
    print("7. PARAMETER IMPORTANCE RANKING")
    print("-" * 50)
    
    # If we have stored samples, we can rank parameters
    if result.monte_carlo.samples:
        print("   Based on correlation with junction temperature:")
        print()
        
        # Extract available samples
        mc = result.monte_carlo
        
        # Show distribution statistics
        print(f"   Temperature distribution:")
        print(f"      Min:    {mc.samples['t_junction_C'].min():.1f}°C")
        print(f"      Max:    {mc.samples['t_junction_C'].max():.1f}°C")
        print(f"      Range:  {mc.samples['t_junction_C'].max() - mc.samples['t_junction_C'].min():.1f}°C")
    
    print()
    
    # =========================================================================
    # Print full validation summary
    # =========================================================================
    print("=" * 70)
    print("COMPLETE VALIDATION SUMMARY")
    print("=" * 70)
    print(result.summary())


def demonstrate_visualization():
    """Demonstrate visualization features (if matplotlib available)."""
    try:
        from icv.visualization import (
            plot_temperature_distribution,
            plot_compliance_probabilities,
            plot_weibull_distribution,
            plot_forex_projection,
            HAS_MATPLOTLIB,
        )
        
        if not HAS_MATPLOTLIB:
            print("\n⚠️  matplotlib not installed. Skipping visualization demo.")
            print("   Install with: pip install matplotlib")
            return
        
        print("\nGENERATING VISUALIZATIONS")
        print("-" * 50)
        
        # Run quick validation
        result = validate_fluid(n_samples=5000)
        mc = result.monte_carlo
        
        # Temperature distribution
        if mc.samples:
            fig = plot_temperature_distribution(
                mc.samples["t_junction_C"],
                limit=88.0,
                save_path="temperature_distribution.png"
            )
            print("   ✅ Saved: temperature_distribution.png")
        
        # Compliance probabilities
        compliance_dict = {
            "Thermal": mc.thermal_compliance_prob,
            "BDV": mc.bdv_compliance_prob,
            "Resistivity": mc.resistivity_compliance_prob,
            "Lifetime": mc.lifetime_compliance_prob,
            "Joint": mc.joint_compliance_prob,
        }
        fig = plot_compliance_probabilities(
            compliance_dict,
            save_path="compliance_probabilities.png"
        )
        print("   ✅ Saved: compliance_probabilities.png")
        
        # Weibull distribution
        fig = plot_weibull_distribution(
            save_path="weibull_distribution.png"
        )
        print("   ✅ Saved: weibull_distribution.png")
        
        # Forex projection
        if result.forex_projection:
            forex = result.forex_projection
            fig = plot_forex_projection(
                forex.years,
                forex.annual_savings_usd,
                forex.cumulative_savings_usd,
                save_path="forex_projection.png"
            )
            print("   ✅ Saved: forex_projection.png")
        
        print()
        
    except Exception as e:
        print(f"\n⚠️  Visualization error: {e}")


if __name__ == "__main__":
    main()
    
    # Try visualization (optional)
    demonstrate_visualization()
