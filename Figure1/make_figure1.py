"""
Auxiliary figures for Inkscape schematics.
Creates simple, clean figures that can be imported into Inkscape for method visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from pathlib import Path

# Color scheme
NS_COLOR = "#47d0ecff"  # Light blue for neutron star distributions (RGBA)

# Source diagram parameters
RADIUS_LARGER_OBJECT = 0.45   # Radius for upper-left object in BNS/NSBH diagrams
RADIUS_SMALLER_OBJECT = 0.45  # Radius for lower-right object in BNS/NSBH diagrams

# Column width ratios for the three-column layout
WIDTH_SOURCE_COLUMN = 0.8      # Width ratio for Source column (left)
WIDTH_POPULATION_COLUMN = 1.2  # Width ratio for Population column (middle)
WIDTH_EOS_COLUMN = 1.5         # Width ratio for EOS column (right)

# Output directory
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

fs_labels_x = 18
fs_labels_y = 24
fs_ticks_x = 20
fs_ticks_y = 14
fs_titles = 24
fs_legend = 16
fs_corner_labels = 16
fs_corner_titles = 16

params = {"axes.grid": False,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": fs_ticks_x,
        "ytick.labelsize": fs_ticks_y,
        "axes.titlesize": fs_titles,
        "figure.titlesize": fs_titles
        }

plt.rcParams.update(params)

# Improved corner kwargs
default_corner_kwargs = dict(bins=40,
                        smooth=1.,
                        show_titles=False,
                        label_kwargs=dict(fontsize=fs_corner_labels),
                        title_kwargs=dict(fontsize=fs_corner_titles), 
                        color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "red",
                        save=False)


def gaussian_mass_pdf(m: np.ndarray) -> np.ndarray:
    """
    Single Gaussian distribution for neutron star masses.
    Hyperparameters from https://arxiv.org/pdf/2407.16669

    Args:
        m: Array of mass values in solar masses

    Returns:
        Probability density at each mass value
    """
    mu = 1.33
    sigma = 0.09
    return norm.pdf(m, loc=mu, scale=sigma)


def double_gaussian_mass_pdf(m: np.ndarray) -> np.ndarray:
    """
    Double Gaussian mixture distribution for neutron star masses.
    Hyperparameters from https://arxiv.org/pdf/2407.16669

    Args:
        m: Array of mass values in solar masses

    Returns:
        Probability density at each mass value
    """
    mu_1 = 1.34
    sigma_1 = 0.07

    mu_2 = 1.80
    sigma_2 = 0.21
    w = 0.65  # Weight for first component

    # Mixture of two Gaussians
    pdf = w * norm.pdf(m, loc=mu_1, scale=sigma_1) + \
          (1 - w) * norm.pdf(m, loc=mu_2, scale=sigma_2)

    return pdf


def uniform_mass_pdf(m: np.ndarray, training_data_path: str = None) -> np.ndarray:
    """
    KDE-based distribution for neutron star masses from uniform population training data.
    Loads m1 and m2 from NFPrior training data and creates a KDE with boundary reflection.

    Args:
        m: Array of mass values in solar masses
        training_data_path: Path to training_data.npz file. If None, uses default uniform/bns/radio_chiEFT

    Returns:
        Probability density at each mass value
    """
    if training_data_path is None:
        # Default to uniform BNS training data
        script_dir = Path(__file__).parent
        training_data_path = script_dir.parent / "NFprior" / "models" / "uniform" / "bns" / "radio_chiEFT" / "training_data.npz"

    # Load training data
    data = np.load(training_data_path)
    m1 = data["m1"]
    m2 = data["m2"]

    # Combine m1 and m2 into single mass array
    all_masses = np.concatenate([m1, m2])

    # Apply boundary reflection at lower bound (1.0 M_sun)
    m_min = 1.0
    reflected_masses = np.concatenate([all_masses, 2 * m_min - all_masses])

    # Create KDE on reflected data
    kde = gaussian_kde(reflected_masses)

    # Evaluate KDE at requested mass values and multiply by 2 to account for reflection
    pdf = 2 * kde(m)

    return pdf


def load_eos_data(eos_path: str) -> dict:
    """
    Load EOS samples from npz file.

    Args:
        eos_path: Path to eos_samples.npz file

    Returns:
        Dictionary with masses, radii, lambdas, and log_prob arrays
    """
    data = np.load(eos_path)
    return {
        'masses': data['masses_EOS'],
        'radii': data['radii_EOS'],
        'lambdas': data['Lambdas_EOS'],
        'log_prob': data['log_prob']
    }


def create_mass_distributions(
    m_min: float = 1.0,
    m_max: float = 2.4,
    n_points: int = 1000,
    save: bool = True,
    show: bool = False
) -> None:
    """
    Create clean visualizations of neutron star mass distributions.
    Creates a three-column figure: Source (BNS/NSBH drawings), Population (mass distributions), and EOS.

    Args:
        m_min: Minimum mass for x-axis (M_sun)
        m_max: Maximum mass for x-axis (M_sun)
        n_points: Number of points for smooth curve
        save: Whether to save the figure
        show: Whether to display the figure
    """
    # Create mass array
    masses = np.linspace(m_min, m_max, n_points)

    # Create figure with GridSpec for custom layout
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Circle

    fig = plt.figure(figsize=(13.5, 10.5))
    # 3 columns: Source (2 rows), Population (3 rows), EOS (3 rows)
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.4,
                  width_ratios=[WIDTH_SOURCE_COLUMN, WIDTH_POPULATION_COLUMN, WIDTH_EOS_COLUMN])

    # Left column: Source drawings (BNS spans rows 0-1.5, NSBH spans rows 1.5-3)
    ax_bns = fig.add_subplot(gs[0:2, 0])
    ax_nsbh = fig.add_subplot(gs[2, 0])

    # Middle column: Population mass distributions
    ax_uniform = fig.add_subplot(gs[0, 1])
    ax_gaussian = fig.add_subplot(gs[1, 1], sharex=ax_uniform)
    ax_double = fig.add_subplot(gs[2, 1], sharex=ax_uniform)

    # Right column: EOS placeholder spanning all rows
    ax_eos = fig.add_subplot(gs[:, 2])

    # ===== Source Column: BNS Drawing =====
    ax_bns.set_xlim(-1.5, 1.5)
    ax_bns.set_ylim(-1.5, 1.5)
    ax_bns.set_aspect('equal')
    ax_bns.axis('off')

    # BNS orbital parameters
    orbit_radius_bns = 0.9
    angle_diag = np.pi / 4  # 45 degrees

    # Position circles on diagonal (upper-left and lower-right)
    # Upper left position
    x1_bns = -orbit_radius_bns * np.cos(angle_diag)
    y1_bns = orbit_radius_bns * np.sin(angle_diag)
    # Lower right position
    x2_bns = orbit_radius_bns * np.cos(angle_diag)
    y2_bns = -orbit_radius_bns * np.sin(angle_diag)

    # Draw orbit line first (lower zorder so it appears behind circles)
    orbit_bns = Circle((0, 0), orbit_radius_bns, fill=False, ec='black', linewidth=1.5,
                       linestyle='--', alpha=0.5, zorder=1)
    ax_bns.add_patch(orbit_bns)

    # Draw BNS system (two light blue circles on diagonal, larger on upper-left)
    circle1_bns = Circle((x1_bns, y1_bns), RADIUS_LARGER_OBJECT, color=NS_COLOR, ec='black', linewidth=2, zorder=2)
    circle2_bns = Circle((x2_bns, y2_bns), RADIUS_SMALLER_OBJECT, color=NS_COLOR, ec='black', linewidth=2, zorder=2)
    ax_bns.add_patch(circle1_bns)
    ax_bns.add_patch(circle2_bns)

    # Add BNS label
    ax_bns.text(0, -1.2, 'BNS', ha='center', va='top', fontsize=fs_titles, fontweight='bold')

    # ===== Source Column: NSBH Drawing =====
    ax_nsbh.set_xlim(-1.5, 1.5)
    ax_nsbh.set_ylim(-1.5, 1.5)
    ax_nsbh.set_aspect('equal')
    ax_nsbh.axis('off')

    # NSBH orbital parameters
    orbit_radius_nsbh = 0.85

    # Position circles on diagonal (upper-left BH, lower-right NS)
    # Upper left position (black hole)
    x1_nsbh = -orbit_radius_nsbh * np.cos(angle_diag)
    y1_nsbh = orbit_radius_nsbh * np.sin(angle_diag)
    # Lower right position (neutron star)
    x2_nsbh = orbit_radius_nsbh * np.cos(angle_diag)
    y2_nsbh = -orbit_radius_nsbh * np.sin(angle_diag)

    # Draw orbit line first (lower zorder so it appears behind circles)
    orbit_nsbh = Circle((0, 0), orbit_radius_nsbh, fill=False, ec='black', linewidth=1.5,
                        linestyle='--', alpha=0.5, zorder=1)
    ax_nsbh.add_patch(orbit_nsbh)

    # Draw NSBH system (black hole upper-left, neutron star lower-right)
    # Use same sizes as BNS for consistency
    circle1_nsbh = Circle((x1_nsbh, y1_nsbh), RADIUS_LARGER_OBJECT, color='black', ec='black', linewidth=2, zorder=2)
    circle2_nsbh = Circle((x2_nsbh, y2_nsbh), RADIUS_SMALLER_OBJECT, color=NS_COLOR, ec='black', linewidth=2, zorder=2)
    ax_nsbh.add_patch(circle1_nsbh)
    ax_nsbh.add_patch(circle2_nsbh)

    # Add NSBH label
    ax_nsbh.text(0, -1.2, 'NSBH', ha='center', va='top', fontsize=fs_titles, fontweight='bold')

    # ===== Population Column: Mass Distributions =====
    # Uniform distribution (KDE from training data)
    pdf_uniform = uniform_mass_pdf(masses)
    ax_uniform.fill_between(masses, pdf_uniform, alpha=0.7, color=NS_COLOR, linewidth=2)
    ax_uniform.plot(masses, pdf_uniform, color=NS_COLOR, linewidth=2.5)
    ax_uniform.set_ylabel('Probability Density', fontsize=fs_labels_y)
    ax_uniform.set_title('Uniform', fontsize=fs_titles)
    ax_uniform.set_xlim(m_min, m_max)
    ax_uniform.set_ylim(0, None)
    ax_uniform.tick_params(labelbottom=False)

    # Single Gaussian distribution
    pdf_gaussian = gaussian_mass_pdf(masses)
    ax_gaussian.fill_between(masses, pdf_gaussian, alpha=0.7, color=NS_COLOR, linewidth=2)
    ax_gaussian.plot(masses, pdf_gaussian, color=NS_COLOR, linewidth=2.5)
    ax_gaussian.set_ylabel('Probability Density', fontsize=fs_labels_y)
    ax_gaussian.set_title('Gaussian', fontsize=fs_titles)
    ax_gaussian.set_xlim(m_min, m_max)
    ax_gaussian.set_ylim(0, None)
    ax_gaussian.tick_params(labelbottom=False)

    # Double Gaussian distribution
    pdf_double = double_gaussian_mass_pdf(masses)
    ax_double.fill_between(masses, pdf_double, alpha=0.7, color=NS_COLOR, linewidth=2)
    ax_double.plot(masses, pdf_double, color=NS_COLOR, linewidth=2.5)
    ax_double.set_xlabel(r'Mass [$M_\odot$]', fontsize=fs_labels_x)
    ax_double.set_ylabel('Probability Density', fontsize=fs_labels_y)
    ax_double.set_title('Double Gaussian', fontsize=fs_titles)
    ax_double.set_xlim(m_min, m_max)
    ax_double.set_ylim(0, None)

    # ===== EOS Column: Lambda(M) Curves =====
    # Define EOS color scheme (from plots/utils.py)
    EOS_COLORS = {
        "radio": "#0472b0",           # Blue
        "radio_chiEFT": "#de8f05",    # Orange
        "radio_NICER": "#ca7abc",     # Pink/purple
    }

    EOS_LABELS = {
        "radio": r"Heavy PSRs",
        "radio_chiEFT": r"+$\chi_{\rm{EFT}}$",
        "radio_NICER": r"+NICER"
    }

    # Load EOS data from each dataset
    script_dir = Path(__file__).parent
    eos_base_dir = script_dir.parent / "data" / "eos"

    eos_datasets = ["radio", "radio_chiEFT", "radio_NICER"]

    # Plot parameters
    lambda_min_eos, lambda_max_eos = 3, 5000
    m_min_eos, m_max_eos = 1.0, 2.8

    # Create mass array for computing credible intervals
    masses_array = np.linspace(m_min_eos, m_max_eos, 100)

    # Plot 90% credible intervals for each dataset
    for eos_name in eos_datasets:
        eos_path = eos_base_dir / eos_name / "eos_samples.npz"

        if not eos_path.exists():
            print(f"Warning: EOS data not found at {eos_path}")
            continue

        eos_data = load_eos_data(str(eos_path))
        masses = eos_data['masses']
        lambdas = eos_data['lambdas']

        color = EOS_COLORS[eos_name]

        # Compute credible intervals at each mass point
        lambda_low = np.empty_like(masses_array)
        lambda_high = np.empty_like(masses_array)

        for i, mass_point in enumerate(masses_array):
            # Gather all lambdas at this mass point by interpolation
            lambdas_at_mass = []
            for mass_curve, lambda_curve in zip(masses, lambdas):
                # Skip invalid samples
                if np.any(np.isnan(mass_curve)) or np.any(np.isnan(lambda_curve)):
                    continue

                # Skip samples with negative lambdas
                if np.any(lambda_curve < 0):
                    continue

                # Check for unphysical mass-radius values
                # (we still have masses available for this check if needed)

                # Interpolate lambda at this mass point
                try:
                    lambda_interp = np.interp(mass_point, mass_curve, lambda_curve)
                    if lambda_interp > 0:  # Only keep positive lambdas
                        lambdas_at_mass.append(lambda_interp)
                except:
                    continue

            lambdas_at_mass = np.array(lambdas_at_mass)

            # Compute 90% credible interval using percentiles
            if len(lambdas_at_mass) > 0:
                lambda_low[i] = np.percentile(lambdas_at_mass, 5)
                lambda_high[i] = np.percentile(lambdas_at_mass, 95)
            else:
                lambda_low[i] = np.nan
                lambda_high[i] = np.nan

        # Plot credible interval with fill_between
        ax_eos.fill_between(masses_array, lambda_low, lambda_high,
                            alpha=0.3, color=color, label=EOS_LABELS[eos_name])

        # Draw boundary lines
        ax_eos.plot(masses_array, lambda_low, color=color, linewidth=1.5, alpha=0.8)
        ax_eos.plot(masses_array, lambda_high, color=color, linewidth=1.5, alpha=0.8)

    # Styling
    ax_eos.set_xlabel(r'$M$ [$M_\odot$]', fontsize=fs_labels_x)
    ax_eos.set_ylabel(r'$\Lambda$', fontsize=fs_labels_y)
    ax_eos.set_xlim(m_min_eos, m_max_eos)
    ax_eos.set_ylim(lambda_min_eos, lambda_max_eos)
    ax_eos.set_yscale('log')
    ax_eos.tick_params(labelsize=fs_ticks_x)

    # Add legend for EOS datasets
    ax_eos.legend(loc='upper right', fontsize=fs_legend, framealpha=0.9)

    # ===== Add column headers =====
    # Source header (left column)
    fig.text(0.15, 0.96, 'Source', ha='center', va='center',
             fontsize=fs_titles+2, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', edgecolor='black', linewidth=2))

    # Population header (middle column)
    fig.text(0.45, 0.96, 'Population', ha='center', va='center',
             fontsize=fs_titles+2, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', edgecolor='black', linewidth=2))

    # EOS header (right column)
    fig.text(0.80, 0.96, 'EOS', ha='center', va='center',
             fontsize=fs_titles+2, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', edgecolor='black', linewidth=2))

    if save:
        output_path = OUTPUT_DIR / "mass_distributions.pdf"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved figure to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    create_mass_distributions(save=True, show=False)