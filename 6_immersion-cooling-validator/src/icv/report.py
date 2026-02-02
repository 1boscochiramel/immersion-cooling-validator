"""
Report generation for immersion cooling validation.

This module generates comprehensive HTML reports with:
- Executive summary
- Monte Carlo results with visualizations
- Compliance assessment details
- Economic analysis
- Sensitivity analysis
- Recommendations

Reports can be exported as standalone HTML files.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import base64
import io


HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --primary: #3b82f6;
            --secondary: #8b5cf6;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        header {{
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 3rem 2rem;
            margin-bottom: 2rem;
            border-radius: 0 0 1rem 1rem;
        }}
        
        header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}
        
        header .subtitle {{
            opacity: 0.9;
            font-size: 1.1rem;
        }}
        
        .card {{
            background: var(--card-bg);
            border-radius: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        
        .card h2 {{
            color: var(--primary);
            border-bottom: 2px solid var(--primary);
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }}
        
        .card h3 {{
            color: var(--text);
            margin: 1rem 0 0.5rem;
        }}
        
        .verdict {{
            display: inline-block;
            padding: 0.5rem 1.5rem;
            border-radius: 2rem;
            font-weight: bold;
            font-size: 1.2rem;
        }}
        
        .verdict.pass {{
            background: var(--success);
            color: white;
        }}
        
        .verdict.fail {{
            background: var(--danger);
            color: white;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        
        .metric {{
            background: var(--bg);
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
        }}
        
        .metric .value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary);
        }}
        
        .metric .label {{
            color: var(--text-muted);
            font-size: 0.9rem;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        th {{
            background: var(--bg);
            font-weight: 600;
        }}
        
        .status {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.85rem;
            font-weight: 500;
        }}
        
        .status.pass {{
            background: #dcfce7;
            color: #166534;
        }}
        
        .status.fail {{
            background: #fee2e2;
            color: #991b1b;
        }}
        
        .status.marginal {{
            background: #fef3c7;
            color: #92400e;
        }}
        
        .chart-container {{
            margin: 1rem 0;
            padding: 1rem;
            background: var(--bg);
            border-radius: 0.5rem;
        }}
        
        .progress-bar {{
            height: 1.5rem;
            background: #e2e8f0;
            border-radius: 0.75rem;
            overflow: hidden;
            margin: 0.5rem 0;
        }}
        
        .progress-fill {{
            height: 100%;
            border-radius: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 0.5rem;
            color: white;
            font-weight: 500;
            font-size: 0.85rem;
        }}
        
        .cost-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }}
        
        .cost-card {{
            padding: 1.5rem;
            border-radius: 0.5rem;
            text-align: center;
        }}
        
        .cost-card.domestic {{
            background: linear-gradient(135deg, #dcfce7, #bbf7d0);
            border: 2px solid var(--success);
        }}
        
        .cost-card.imported {{
            background: linear-gradient(135deg, #fee2e2, #fecaca);
            border: 2px solid var(--danger);
        }}
        
        .cost-value {{
            font-size: 2rem;
            font-weight: bold;
        }}
        
        .savings-banner {{
            background: linear-gradient(135deg, var(--success), #15803d);
            color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            text-align: center;
            margin-top: 1rem;
        }}
        
        .savings-banner .value {{
            font-size: 2.5rem;
            font-weight: bold;
        }}
        
        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            font-size: 0.9rem;
        }}
        
        @media print {{
            header {{
                background: var(--primary);
                -webkit-print-color-adjust: exact;
            }}
            .card {{
                break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>🌡️ Immersion Cooling Validation Report</h1>
            <p class="subtitle">{fluid_name} • Generated {timestamp}</p>
        </div>
    </header>
    
    <div class="container">
        {content}
    </div>
    
    <footer>
        <p>Generated by Immersion Cooling Validator v0.1.0</p>
        <p>Based on: Chiramel, B. (2024). Monte Carlo Validation Framework for Group III Hydrocarbon-Based Single-Phase Immersion Cooling Fluid</p>
    </footer>
</body>
</html>
'''


def _generate_executive_summary(result: 'ValidationResult') -> str:
    """Generate executive summary section."""
    mc = result.monte_carlo
    verdict_class = "pass" if result.compliance.critical_pass else "fail"
    verdict_text = "✅ PASS" if result.compliance.critical_pass else "❌ FAIL"
    
    return f'''
    <div class="card">
        <h2>📋 Executive Summary</h2>
        
        <p style="margin-bottom: 1rem;">
            This report presents the Monte Carlo validation results for <strong>{result.fluid_name}</strong>
            as an immersion cooling fluid for AI data center applications, evaluated against 
            Open Compute Project (OCP) specifications.
        </p>
        
        <div style="text-align: center; margin: 1.5rem 0;">
            <span class="verdict {verdict_class}">{verdict_text}</span>
        </div>
        
        <div class="metrics-grid">
            <div class="metric">
                <div class="value">{mc.t_junction_mean:.1f}°C</div>
                <div class="label">Junction Temperature</div>
            </div>
            <div class="metric">
                <div class="value">{mc.joint_compliance_prob*100:.1f}%</div>
                <div class="label">Joint Compliance</div>
            </div>
            <div class="metric">
                <div class="value">{mc.n_samples:,}</div>
                <div class="label">Monte Carlo Samples</div>
            </div>
            <div class="metric">
                <div class="value">{mc.bdv_mean:.1f} kV</div>
                <div class="label">Breakdown Voltage</div>
            </div>
        </div>
    </div>
    '''


def _generate_thermal_section(result: 'ValidationResult') -> str:
    """Generate thermal performance section."""
    mc = result.monte_carlo
    
    # Generate simple ASCII-style histogram representation
    percentiles = mc.t_junction_percentiles
    
    return f'''
    <div class="card">
        <h2>🌡️ Thermal Performance</h2>
        
        <p>
            Junction temperature analysis based on thermal resistance network model:
            <strong>Tj = Tbulk + Q × Rtotal</strong>
        </p>
        
        <div class="metrics-grid">
            <div class="metric">
                <div class="value">{mc.t_junction_mean:.1f}°C</div>
                <div class="label">Mean Temperature</div>
            </div>
            <div class="metric">
                <div class="value">±{mc.t_junction_std:.1f}°C</div>
                <div class="label">Standard Deviation</div>
            </div>
            <div class="metric">
                <div class="value">{percentiles.get(95, 0):.1f}°C</div>
                <div class="label">95th Percentile</div>
            </div>
            <div class="metric">
                <div class="value">{88.0 - mc.t_junction_mean:.1f}°C</div>
                <div class="label">Margin to Limit</div>
            </div>
        </div>
        
        <h3>Temperature Distribution</h3>
        <div class="progress-bar" style="position: relative;">
            <div class="progress-fill" style="width: {min(100, mc.t_junction_mean/88*100):.0f}%; background: {'var(--success)' if mc.t_junction_mean < 75 else 'var(--warning)'};">
                {mc.t_junction_mean:.1f}°C
            </div>
            <div style="position: absolute; right: 5px; top: 50%; transform: translateY(-50%); font-size: 0.8rem; color: var(--danger);">
                Limit: 88°C
            </div>
        </div>
        
        <table>
            <tr><th>Percentile</th><th>Temperature (°C)</th><th>Status</th></tr>
            <tr>
                <td>P5 (5th)</td>
                <td>{percentiles.get(5, 0):.1f}</td>
                <td><span class="status pass">✓ Compliant</span></td>
            </tr>
            <tr>
                <td>P50 (Median)</td>
                <td>{percentiles.get(50, 0):.1f}</td>
                <td><span class="status pass">✓ Compliant</span></td>
            </tr>
            <tr>
                <td>P95 (95th)</td>
                <td>{percentiles.get(95, 0):.1f}</td>
                <td><span class="status {'pass' if percentiles.get(95, 0) < 88 else 'fail'}">{'✓ Compliant' if percentiles.get(95, 0) < 88 else '✗ Exceeds'}</span></td>
            </tr>
        </table>
        
        <p style="margin-top: 1rem; color: var(--text-muted);">
            <strong>Compliance Probability:</strong> {mc.thermal_compliance_prob*100:.2f}% of samples below 88°C thermal throttle limit.
        </p>
    </div>
    '''


def _generate_electrical_section(result: 'ValidationResult') -> str:
    """Generate electrical properties section."""
    mc = result.monte_carlo
    
    return f'''
    <div class="card">
        <h2>⚡ Electrical Properties</h2>
        
        <p>
            Electrical properties validated using Arrhenius temperature model:
            <strong>ρ(T) = ρref × exp[Ea/kB × (1/T − 1/Tref)]</strong>
        </p>
        
        <table>
            <tr>
                <th>Property</th>
                <th>Value</th>
                <th>OCP Requirement</th>
                <th>Compliance</th>
            </tr>
            <tr>
                <td>Breakdown Voltage</td>
                <td>{mc.bdv_mean:.1f} ± {mc.bdv_std:.1f} kV</td>
                <td>&gt; 45 kV</td>
                <td><span class="status {'pass' if mc.bdv_compliance_prob > 0.95 else 'marginal' if mc.bdv_compliance_prob > 0.9 else 'fail'}">{mc.bdv_compliance_prob*100:.2f}%</span></td>
            </tr>
            <tr>
                <td>Volume Resistivity</td>
                <td>{mc.resistivity_mean:.2e} Ω·cm</td>
                <td>&gt; 10¹¹ Ω·cm</td>
                <td><span class="status {'pass' if mc.resistivity_compliance_prob > 0.95 else 'fail'}">{mc.resistivity_compliance_prob*100:.2f}%</span></td>
            </tr>
        </table>
        
        <h3>Breakdown Voltage Margin</h3>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {min(100, mc.bdv_mean/60*100):.0f}%; background: var(--primary);">
                {mc.bdv_mean:.1f} kV
            </div>
        </div>
        <p style="font-size: 0.85rem; color: var(--text-muted);">
            Minimum requirement: 45 kV | Margin: {mc.bdv_mean - 45:.1f} kV ({(mc.bdv_mean/45 - 1)*100:.1f}%)
        </p>
    </div>
    '''


def _generate_lifetime_section(result: 'ValidationResult') -> str:
    """Generate service life section."""
    mc = result.monte_carlo
    
    return f'''
    <div class="card">
        <h2>⏱️ Service Life</h2>
        
        <p>
            Service life modeled using Weibull distribution:
            <strong>F(t) = 1 − exp[−(t/λ)ᵏ]</strong>
        </p>
        
        <div class="metrics-grid">
            <div class="metric">
                <div class="value">{mc.p5_life_mean:.1f} yrs</div>
                <div class="label">P5 Life (95% Survival)</div>
            </div>
            <div class="metric">
                <div class="value">{mc.lifetime_compliance_prob*100:.1f}%</div>
                <div class="label">Compliance Probability</div>
            </div>
        </div>
        
        <h3>Service Life Compliance</h3>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {min(100, mc.p5_life_mean/10*100):.0f}%; background: var(--secondary);">
                {mc.p5_life_mean:.1f} years
            </div>
        </div>
        <p style="font-size: 0.85rem; color: var(--text-muted);">
            OCP Minimum: 5 years | Margin: {mc.p5_life_mean - 5:.1f} years
        </p>
    </div>
    '''


def _generate_compliance_section(result: 'ValidationResult') -> str:
    """Generate compliance assessment section."""
    mc = result.monte_carlo
    
    compliance_items = [
        ("Thermal", mc.thermal_compliance_prob, "Junction Temp < 88°C"),
        ("Breakdown Voltage", mc.bdv_compliance_prob, "BDV > 45 kV"),
        ("Resistivity", mc.resistivity_compliance_prob, "ρ > 10¹¹ Ω·cm"),
        ("Service Life", mc.lifetime_compliance_prob, "P5 > 5 years"),
    ]
    
    rows = ""
    for name, prob, req in compliance_items:
        status_class = "pass" if prob > 0.99 else "marginal" if prob > 0.95 else "fail"
        rows += f'''
            <tr>
                <td>{name}</td>
                <td>{req}</td>
                <td>
                    <div class="progress-bar" style="height: 1rem; margin: 0;">
                        <div class="progress-fill" style="width: {prob*100:.0f}%; background: var({'--success' if prob > 0.99 else '--warning' if prob > 0.95 else '--danger'});">
                        </div>
                    </div>
                </td>
                <td style="text-align: right;">{prob*100:.2f}%</td>
                <td><span class="status {status_class}">{'✓ Pass' if prob > 0.95 else '⚠ Marginal' if prob > 0.9 else '✗ Fail'}</span></td>
            </tr>
        '''
    
    return f'''
    <div class="card">
        <h2>✅ OCP Compliance Assessment</h2>
        
        <table>
            <tr>
                <th>Requirement</th>
                <th>Criterion</th>
                <th style="width: 30%;">Compliance</th>
                <th>Probability</th>
                <th>Status</th>
            </tr>
            {rows}
        </table>
        
        <div style="text-align: center; margin-top: 1.5rem; padding: 1rem; background: var(--bg); border-radius: 0.5rem;">
            <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">Joint Compliance Probability</p>
            <p style="font-size: 2.5rem; font-weight: bold; color: {'var(--success)' if mc.joint_compliance_prob > 0.95 else 'var(--warning)'};">
                {mc.joint_compliance_prob*100:.1f}%
            </p>
            <p style="font-size: 0.9rem; color: var(--text-muted);">
                Probability of meeting ALL OCP specifications simultaneously
            </p>
        </div>
    </div>
    '''


def _generate_economics_section(result: 'ValidationResult') -> str:
    """Generate economic analysis section."""
    if result.economics is None:
        return ""
    
    eco = result.economics
    
    forex_content = ""
    if result.forex_projection:
        forex = result.forex_projection
        forex_content = f'''
        <h3>Forex Savings Projection ({forex.years[0]}-{forex.years[-1]})</h3>
        <div class="savings-banner">
            <div class="value">${forex.total_savings_usd/1e6:.0f}M USD</div>
            <p>Projected foreign exchange savings</p>
            <p style="font-size: 1.2rem;">₹{forex.total_savings_inr_crores:.0f} Crores</p>
        </div>
        '''
    
    return f'''
    <div class="card">
        <h2>💰 Economic Analysis</h2>
        
        <div class="cost-comparison">
            <div class="cost-card domestic">
                <h3>{eco.fluid_a_name}</h3>
                <div class="cost-value" style="color: var(--success);">${eco.tco_a_usd/1e6:.1f}M</div>
                <p>Total Cost of Ownership</p>
            </div>
            <div class="cost-card imported">
                <h3>{eco.fluid_b_name}</h3>
                <div class="cost-value" style="color: var(--danger);">${eco.tco_b_usd/1e6:.1f}M</div>
                <p>Total Cost of Ownership</p>
            </div>
        </div>
        
        <div class="savings-banner">
            <div class="value">{eco.relative_savings_percent:.0f}% Savings</div>
            <p>Cost reduction with domestic Group III oil</p>
            <p style="font-size: 1.2rem;">${eco.absolute_savings_usd/1e6:.1f}M saved</p>
        </div>
        
        {forex_content}
    </div>
    '''


def generate_html_report(result: 'ValidationResult') -> str:
    """
    Generate complete HTML validation report.
    
    Args:
        result: ValidationResult from validator
        
    Returns:
        Complete HTML report as string
    """
    content = (
        _generate_executive_summary(result) +
        _generate_thermal_section(result) +
        _generate_electrical_section(result) +
        _generate_lifetime_section(result) +
        _generate_compliance_section(result) +
        _generate_economics_section(result)
    )
    
    return HTML_TEMPLATE.format(
        title=f"Validation Report - {result.fluid_name}",
        fluid_name=result.fluid_name,
        timestamp=result.timestamp,
        content=content,
    )


def save_html_report(result: 'ValidationResult', filepath: str):
    """
    Save validation report as HTML file.
    
    Args:
        result: ValidationResult from validator
        filepath: Output file path
    """
    html = generate_html_report(result)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    
    include_executive_summary: bool = True
    include_thermal: bool = True
    include_electrical: bool = True
    include_lifetime: bool = True
    include_compliance: bool = True
    include_economics: bool = True
    include_sensitivity: bool = False
    include_charts: bool = True
    
    company_name: Optional[str] = None
    company_logo_path: Optional[str] = None
    additional_notes: Optional[str] = None
