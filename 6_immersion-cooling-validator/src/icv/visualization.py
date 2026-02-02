"""
Visualization module for immersion cooling validation results.

This module provides plotting functions for:
- Monte Carlo distributions (histograms, KDE)
- Compliance probability charts
- Tornado diagrams for sensitivity
- Economic comparison charts
- Benchmark comparisons

Supports both matplotlib and plotly backends.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Check for visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# Color schemes
COLORS = {
    "pass": "#22c55e",      # Green
    "fail": "#ef4444",      # Red
    "marginal": "#f59e0b",  # Amber
    "primary": "#3b82f6",   # Blue
    "secondary": "#8b5cf6", # Purple
    "domestic": "#10b981",  # Emerald
    "imported": "#f43f5e",  # Rose
}


def check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )


def check_plotly():
    """Check if plotly is available."""
    if not HAS_PLOTLY:
        raise ImportError(
            "plotly is required for interactive plots. "
            "Install with: pip install plotly"
        )


# =============================================================================
# Matplotlib-based plots
# =============================================================================

def plot_temperature_distribution(
    samples: np.ndarray,
    limit: float = 88.0,
    title: str = "Junction Temperature Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot Monte Carlo distribution of junction temperature.
    
    Args:
        samples: Array of junction temperature samples
        limit: OCP thermal limit (default 88°C)
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        matplotlib figure
    """
    check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Histogram
    n, bins, patches = ax.hist(
        samples, bins=50, density=True, alpha=0.7,
        color=COLORS["primary"], edgecolor='white', linewidth=0.5
    )
    
    # Color bars based on compliance
    for i, (patch, left_edge) in enumerate(zip(patches, bins[:-1])):
        if left_edge >= limit:
            patch.set_facecolor(COLORS["fail"])
    
    # Add limit line
    ax.axvline(limit, color=COLORS["fail"], linestyle='--', linewidth=2, 
               label=f'OCP Limit ({limit}°C)')
    
    # Statistics
    mean = np.mean(samples)
    std = np.std(samples)
    p95 = np.percentile(samples, 95)
    compliance = 100 * np.sum(samples < limit) / len(samples)
    
    ax.axvline(mean, color=COLORS["secondary"], linestyle='-', linewidth=2,
               label=f'Mean: {mean:.1f}°C')
    
    # Add text box with stats
    stats_text = (
        f"Mean: {mean:.1f}°C\n"
        f"Std: {std:.1f}°C\n"
        f"P95: {p95:.1f}°C\n"
        f"Compliance: {compliance:.1f}%"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Junction Temperature (°C)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_compliance_probabilities(
    compliance_dict: Dict[str, float],
    title: str = "OCP Compliance Probabilities",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot bar chart of compliance probabilities.
    
    Args:
        compliance_dict: Dict mapping requirement names to probabilities
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        matplotlib figure
    """
    check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    names = list(compliance_dict.keys())
    values = [v * 100 for v in compliance_dict.values()]
    
    # Color based on compliance level
    colors = []
    for v in values:
        if v >= 99:
            colors.append(COLORS["pass"])
        elif v >= 95:
            colors.append(COLORS["marginal"])
        else:
            colors.append(COLORS["fail"])
    
    bars = ax.barh(names, values, color=colors, edgecolor='white', linewidth=1)
    
    # Add value labels
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{value:.1f}%', va='center', fontsize=10)
    
    # Add threshold line
    ax.axvline(95, color='gray', linestyle='--', linewidth=1, alpha=0.7,
               label='95% threshold')
    
    ax.set_xlim(0, 105)
    ax.set_xlabel('Compliance Probability (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_tornado_diagram(
    tornado_data: 'TornadoData',
    title: str = "Parameter Sensitivity (Tornado Diagram)",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot tornado diagram showing parameter sensitivity.
    
    Args:
        tornado_data: TornadoData from sensitivity analysis
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        matplotlib figure
    """
    check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by swing
    sorted_data = tornado_data.sorted_by_swing[:10]  # Top 10
    
    names = [d[0] for d in sorted_data]
    lows = [d[1] for d in sorted_data]
    highs = [d[2] for d in sorted_data]
    baseline = tornado_data.baseline
    
    y_pos = np.arange(len(names))
    
    # Plot bars from baseline
    for i, (name, low, high) in enumerate(zip(names, lows, highs)):
        # Low side (left of baseline)
        if low < baseline:
            ax.barh(i, baseline - low, left=low, color=COLORS["primary"], alpha=0.7)
        else:
            ax.barh(i, low - baseline, left=baseline, color=COLORS["primary"], alpha=0.7)
        
        # High side (right of baseline)
        if high > baseline:
            ax.barh(i, high - baseline, left=baseline, color=COLORS["secondary"], alpha=0.7)
        else:
            ax.barh(i, baseline - high, left=high, color=COLORS["secondary"], alpha=0.7)
    
    # Baseline line
    ax.axvline(baseline, color='black', linestyle='-', linewidth=2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Junction Temperature (°C)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Legend
    low_patch = mpatches.Patch(color=COLORS["primary"], alpha=0.7, label='Low value')
    high_patch = mpatches.Patch(color=COLORS["secondary"], alpha=0.7, label='High value')
    ax.legend(handles=[low_patch, high_patch], loc='lower right')
    
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_cost_comparison(
    domestic_costs: Dict[str, float],
    imported_costs: Dict[str, float],
    domestic_name: str = "Group III Oil",
    imported_name: str = "Fluorinert FC-77",
    title: str = "Total Cost of Ownership Comparison",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot cost comparison bar chart.
    
    Args:
        domestic_costs: Dict of cost components for domestic fluid
        imported_costs: Dict of cost components for imported fluid
        domestic_name: Name of domestic fluid
        imported_name: Name of imported fluid
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        matplotlib figure
    """
    check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    categories = list(domestic_costs.keys())
    x = np.arange(len(categories))
    width = 0.35
    
    domestic_vals = [domestic_costs[c] / 1e6 for c in categories]  # In millions
    imported_vals = [imported_costs[c] / 1e6 for c in categories]
    
    bars1 = ax.bar(x - width/2, domestic_vals, width, label=domestic_name,
                   color=COLORS["domestic"])
    bars2 = ax.bar(x + width/2, imported_vals, width, label=imported_name,
                   color=COLORS["imported"])
    
    ax.set_ylabel('Cost ($ Million)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'${height:.1f}M',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_forex_projection(
    years: List[int],
    annual_savings: List[float],
    cumulative_savings: List[float],
    title: str = "Foreign Exchange Savings Projection",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot forex savings projection over time.
    
    Args:
        years: List of years
        annual_savings: Annual savings for each year
        cumulative_savings: Cumulative savings
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        matplotlib figure
    """
    check_matplotlib()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Annual savings bar chart
    colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(years)))
    bars = ax1.bar(years, [s/1e6 for s in annual_savings], color=colors)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Annual Savings ($ Million)', fontsize=12)
    ax1.set_title('Annual Forex Savings', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, annual_savings):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'${val/1e6:.0f}M', ha='center', va='bottom', fontsize=9)
    
    # Cumulative savings line chart
    ax2.fill_between(years, [s/1e6 for s in cumulative_savings], 
                     alpha=0.3, color=COLORS["domestic"])
    ax2.plot(years, [s/1e6 for s in cumulative_savings], 
             marker='o', color=COLORS["domestic"], linewidth=2, markersize=8)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Cumulative Savings ($ Million)', fontsize=12)
    ax2.set_title('Cumulative Forex Savings', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add final value annotation
    final_val = cumulative_savings[-1]
    ax2.annotate(f'Total: ${final_val/1e6:.0f}M',
                xy=(years[-1], final_val/1e6),
                xytext=(10, 0), textcoords='offset points',
                fontsize=11, fontweight='bold',
                color=COLORS["domestic"])
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_weibull_distribution(
    shape_k: float = 8.8,
    scale_lambda: float = 8.1,
    min_life_requirement: float = 5.0,
    title: str = "Weibull Service Life Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot Weibull service life distribution.
    
    Args:
        shape_k: Weibull shape parameter
        scale_lambda: Weibull scale parameter (years)
        min_life_requirement: OCP minimum life requirement
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        matplotlib figure
    """
    check_matplotlib()
    from scipy import stats
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate Weibull distribution
    x = np.linspace(0, 12, 200)
    weibull = stats.weibull_min(shape_k, scale=scale_lambda)
    pdf = weibull.pdf(x)
    
    # Plot PDF
    ax.plot(x, pdf, color=COLORS["primary"], linewidth=2, label='PDF')
    ax.fill_between(x, pdf, alpha=0.3, color=COLORS["primary"])
    
    # Mark P5 and P50
    p5 = weibull.ppf(0.05)
    p50 = weibull.ppf(0.50)
    
    ax.axvline(p5, color=COLORS["secondary"], linestyle='--', linewidth=2,
               label=f'P5 = {p5:.1f} years')
    ax.axvline(p50, color=COLORS["marginal"], linestyle='--', linewidth=2,
               label=f'P50 = {p50:.1f} years')
    
    # Mark requirement
    ax.axvline(min_life_requirement, color=COLORS["fail"], linestyle='-', linewidth=2,
               label=f'Min Requirement = {min_life_requirement} years')
    
    # Shade compliant region
    compliance_x = x[x >= min_life_requirement]
    compliance_pdf = pdf[x >= min_life_requirement]
    ax.fill_between(compliance_x, compliance_pdf, alpha=0.2, color=COLORS["pass"])
    
    ax.set_xlabel('Service Life (years)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# Plotly-based interactive plots
# =============================================================================

def plot_temperature_distribution_interactive(
    samples: np.ndarray,
    limit: float = 88.0,
    title: str = "Junction Temperature Distribution",
) -> Any:
    """
    Create interactive Plotly histogram of junction temperature.
    
    Args:
        samples: Array of junction temperature samples
        limit: OCP thermal limit
        title: Plot title
        
    Returns:
        Plotly figure
    """
    check_plotly()
    
    mean = np.mean(samples)
    std = np.std(samples)
    compliance = 100 * np.sum(samples < limit) / len(samples)
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=samples,
        nbinsx=50,
        name='Temperature Distribution',
        marker_color=COLORS["primary"],
        opacity=0.7,
    ))
    
    # Limit line
    fig.add_vline(x=limit, line_dash="dash", line_color=COLORS["fail"],
                  annotation_text=f"OCP Limit: {limit}°C")
    
    # Mean line
    fig.add_vline(x=mean, line_color=COLORS["secondary"],
                  annotation_text=f"Mean: {mean:.1f}°C")
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Junction Temperature (°C)",
        yaxis_title="Count",
        showlegend=True,
        annotations=[
            dict(
                x=0.98, y=0.98,
                xref='paper', yref='paper',
                text=f"Mean: {mean:.1f}°C<br>Std: {std:.1f}°C<br>Compliance: {compliance:.1f}%",
                showarrow=False,
                bgcolor='white',
                bordercolor='gray',
                borderwidth=1,
            )
        ]
    )
    
    return fig


def plot_compliance_radar(
    compliance_dict: Dict[str, float],
    title: str = "OCP Compliance Radar",
) -> Any:
    """
    Create radar chart of compliance probabilities.
    
    Args:
        compliance_dict: Dict mapping requirements to probabilities
        title: Plot title
        
    Returns:
        Plotly figure
    """
    check_plotly()
    
    categories = list(compliance_dict.keys())
    values = [v * 100 for v in compliance_dict.values()]
    
    # Close the radar
    categories = categories + [categories[0]]
    values = values + [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor=f'rgba(59, 130, 246, 0.3)',
        line_color=COLORS["primary"],
        name='Compliance',
    ))
    
    # Add 95% threshold
    threshold_values = [95] * len(categories)
    fig.add_trace(go.Scatterpolar(
        r=threshold_values,
        theta=categories,
        line_color=COLORS["fail"],
        line_dash='dash',
        name='95% Threshold',
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 105],
            )
        ),
        showlegend=True,
        title=dict(text=title, font=dict(size=16)),
    )
    
    return fig


def create_validation_dashboard(
    mc_result: 'MonteCarloResult',
    compliance_result: 'ComplianceResult',
    economics: Optional['CostComparison'] = None,
) -> Any:
    """
    Create comprehensive validation dashboard with multiple plots.
    
    Args:
        mc_result: Monte Carlo simulation results
        compliance_result: Compliance assessment results
        economics: Optional economic analysis results
        
    Returns:
        Plotly figure with subplots
    """
    check_plotly()
    
    n_rows = 2 if economics is None else 3
    
    fig = make_subplots(
        rows=n_rows, cols=2,
        subplot_titles=(
            'Junction Temperature Distribution',
            'Compliance Probabilities',
            'Service Life Distribution',
            'Key Metrics Summary',
        ),
        specs=[
            [{"type": "histogram"}, {"type": "bar"}],
            [{"type": "histogram"}, {"type": "table"}],
        ] if economics is None else [
            [{"type": "histogram"}, {"type": "bar"}],
            [{"type": "histogram"}, {"type": "table"}],
            [{"type": "bar"}, {"type": "bar"}],
        ]
    )
    
    # Temperature distribution
    if mc_result.samples is not None:
        tj_samples = mc_result.samples["t_junction_C"]
        fig.add_trace(
            go.Histogram(x=tj_samples, nbinsx=40, marker_color=COLORS["primary"]),
            row=1, col=1
        )
        fig.add_vline(x=88, line_dash="dash", line_color=COLORS["fail"], row=1, col=1)
    
    # Compliance bars
    compliance_names = ['Thermal', 'BDV', 'Resistivity', 'Lifetime', 'Joint']
    compliance_values = [
        mc_result.thermal_compliance_prob * 100,
        mc_result.bdv_compliance_prob * 100,
        mc_result.resistivity_compliance_prob * 100,
        mc_result.lifetime_compliance_prob * 100,
        mc_result.joint_compliance_prob * 100,
    ]
    colors = [COLORS["pass"] if v >= 95 else COLORS["marginal"] if v >= 90 else COLORS["fail"]
              for v in compliance_values]
    
    fig.add_trace(
        go.Bar(x=compliance_names, y=compliance_values, marker_color=colors),
        row=1, col=2
    )
    
    # Lifetime distribution
    if mc_result.samples is not None:
        life_samples = mc_result.samples["p5_life_years"]
        fig.add_trace(
            go.Histogram(x=life_samples, nbinsx=40, marker_color=COLORS["secondary"]),
            row=2, col=1
        )
    
    # Metrics table
    fig.add_trace(
        go.Table(
            header=dict(values=['Metric', 'Value']),
            cells=dict(values=[
                ['Tj Mean', 'Tj Std', 'BDV Mean', 'Joint Compliance'],
                [f'{mc_result.t_junction_mean:.1f}°C',
                 f'{mc_result.t_junction_std:.1f}°C',
                 f'{mc_result.bdv_mean:.1f} kV',
                 f'{mc_result.joint_compliance_prob*100:.1f}%']
            ])
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600 if economics is None else 900,
        showlegend=False,
        title_text="Immersion Cooling Validation Dashboard",
    )
    
    return fig
