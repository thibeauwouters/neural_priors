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

# Column width ratios for the three-column layout
WIDTH_SOURCE_COLUMN = 0.7      # Width ratio for Source column (left)
WIDTH_POPULATION_COLUMN = 1.2  # Width ratio for Population column (middle)
WIDTH_EOS_COLUMN = 1.5         # Width ratio for EOS column (right)

# Output directory
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

LAMBDA_MIN_EOS, LAMBDA_MAX_EOS = 3, 7000
M_MIN_EOS, M_MAX_EOS = 1.0, 2.4
N_MASSES_LAMBDAS_PLOT = 20

# ===== Font sizes and spacing parameters =====

# GridSpec spacing parameters
HSPACE_POPULATION = 0.55        # Vertical spacing between population plots (higher = more whitespace)
WSPACE_COLUMNS = 0.4            # Horizontal spacing between columns

######################
### SOURCE DRAWING ###
######################

source_orbit_radius_bns = 6.0           # Orbital radius for BNS system (controls drawing size)
source_orbit_radius_nsbh = source_orbit_radius_bns          # Orbital radius for NSBH system (controls drawing size)
RADIUS_LARGER_OBJECT = 3.0   # Radius for upper-left object in BNS/NSBH diagrams
RADIUS_SMALLER_OBJECT = RADIUS_LARGER_OBJECT  # Radius for lower-right object in BNS/NSBH diagrams

# Drawing position parameters (vertical positioning within each subplot)
source_vertical_offset_bns = 1.0        # Vertical offset for BNS drawing (positive = up)
source_vertical_offset_nsbh = 1.0       # Vertical offset for NSBH drawing (positive = up)

# Label text parameters
fs_source_label = 36                    # Font size for "BNS" and "NSBH" text labels
source_label_x_bns = 0.09               # X position for BNS label in figure coordinates (0-1 scale)
source_label_y_bns = 0.60               # Y position for BNS label in figure coordinates (0-1 scale)
source_label_x_nsbh = 0.09              # X position for NSBH label in figure coordinates (0-1 scale)
source_label_y_nsbh = 0.175              # Y position for NSBH label in figure coordinates (0-1 scale)

# Population column (middle) - Mass distribution plots
fs_population_xlabel = 32      # Font size for x-axis label (Mass)
fs_population_ylabel = 26      # Font size for y-axis labels (Probability Density)
fs_population_title = 30       # Font size for plot titles (Uniform, Gaussian, etc.)
fs_xticks_population = 26      # Font size for all population plots x-axis ticks
fs_yticks_population = 20      # Font size for all population plots y-axis ticks

# EOS column (right) - Lambda(M) plot
fs_eos_xlabel = 32             # Font size for x-axis label (M)
fs_eos_ylabel = 32             # Font size for y-axis label (Lambda)
fs_xticks_eos = 26             # Font size for EOS plot x-axis tick labels
fs_yticks_eos = 26             # Font size for EOS plot y-axis tick labels
fs_eos_legend = 26             # Font size for legend text

# EOS credible interval styling
eos_border_linewidth = 3.0     # Line width for credible interval borders
eos_border_alpha = 0.9         # Alpha (transparency) for border lines (0=transparent, 1=opaque)
eos_fill_alpha = 0.25          # Alpha (transparency) for fill_between shading

# Column headers (Source, Population, EOS)
fs_column_headers = 40         # Font size for the column header text at top

# Column header positions (x-coordinate in figure coordinates, 0-1 scale)
header_x_source = 0.10          # X position for Source header
header_x_population = 0.415     # X position for Population header
header_x_eos = 0.80             # X position for EOS header
header_y = 0.96                 # Y position for all headers (vertical position)

# Column header underline parameters
underline_width_source = 0.10       # Width of underline for Source header
underline_width_population = 0.15   # Width of underline for Population header
underline_width_eos = 0.08          # Width of underline for EOS header
underline_offset = 0.03             # Distance below text to underline

# Bottom curly bracket parameters
bracket_x_left = 0.08               # Left edge of bracket (figure coordinates 0-1)
bracket_x_right = 0.92              # Right edge of bracket (figure coordinates 0-1)
bracket_y = -0.02                   # Y position for bracket top edge (negative moves it down below bottom)
bracket_height = 0.025              # Depth of the center dip (increased for steeper/deeper)
bracket_text_y = -0.08              # Y position for text below bracket (adjusted accordingly)
bracket_flat_length = 0.42          # Length of flat horizontal sections (fraction of total width, increased for tighter dip)
fs_bracket_text = 50                # Font size for pi_NF text below bracket

# Corner plot kwargs (if used elsewhere)
fs_corner_labels = 20
fs_corner_titles = 20

params = {"axes.grid": False,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"]
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
    n_points: int = 250,
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

    fig = plt.figure(figsize=(13.5, 12.0))
    # 6 rows for better control: Source gets 2 equal rows (0-2 for BNS, 3-5 for NSBH)
    # Population and EOS span all 6 rows but we'll create 3 subplots each
    gs = GridSpec(6, 3, figure=fig, hspace=HSPACE_POPULATION, wspace=WSPACE_COLUMNS,
                  width_ratios=[WIDTH_SOURCE_COLUMN, WIDTH_POPULATION_COLUMN, WIDTH_EOS_COLUMN],
                  left=0.02, right=0.99)

    # Left column: Source drawings - each gets 3 rows for equal space
    ax_bns = fig.add_subplot(gs[0:3, 0])
    ax_nsbh = fig.add_subplot(gs[3:6, 0])

    # Middle column: Population mass distributions - distribute across 6 rows
    ax_uniform = fig.add_subplot(gs[0:2, 1])
    ax_gaussian = fig.add_subplot(gs[2:4, 1], sharex=ax_uniform)
    ax_double = fig.add_subplot(gs[4:6, 1], sharex=ax_uniform)

    # Right column: EOS placeholder spanning all 6 rows
    ax_eos = fig.add_subplot(gs[:, 2])

    # ===== Source Column: BNS Drawing =====
    # Calculate axis limits automatically to fit drawing only (no label spacing)
    # Need to accommodate: orbit radius + circle radius + offset
    ylim_bns = source_orbit_radius_bns + RADIUS_LARGER_OBJECT + abs(source_vertical_offset_bns) + 0.3
    ax_bns.set_xlim(-ylim_bns, ylim_bns)
    ax_bns.set_ylim(-ylim_bns, ylim_bns)
    ax_bns.set_aspect('equal')
    ax_bns.axis('off')

    # BNS orbital parameters
    angle_diag = np.pi / 4  # 45 degrees

    # Position circles on diagonal (upper-left and lower-right)
    # Upper left position
    x1_bns = -source_orbit_radius_bns * np.cos(angle_diag)
    y1_bns = source_orbit_radius_bns * np.sin(angle_diag) + source_vertical_offset_bns
    # Lower right position
    x2_bns = source_orbit_radius_bns * np.cos(angle_diag)
    y2_bns = -source_orbit_radius_bns * np.sin(angle_diag) + source_vertical_offset_bns

    # Draw orbit line first (lower zorder so it appears behind circles)
    orbit_bns = Circle((0, source_vertical_offset_bns), source_orbit_radius_bns, fill=False, ec='black', linewidth=1.5,
                       linestyle='--', alpha=0.5, zorder=1)
    ax_bns.add_patch(orbit_bns)

    # Draw BNS system (two light blue circles on diagonal, larger on upper-left)
    circle1_bns = Circle((x1_bns, y1_bns), RADIUS_LARGER_OBJECT, color=NS_COLOR, ec='black', linewidth=2, zorder=2)
    circle2_bns = Circle((x2_bns, y2_bns), RADIUS_SMALLER_OBJECT, color=NS_COLOR, ec='black', linewidth=2, zorder=2)
    ax_bns.add_patch(circle1_bns)
    ax_bns.add_patch(circle2_bns)

    # ===== Source Column: NSBH Drawing =====
    # Calculate axis limits automatically to fit drawing only (no label spacing)
    # Need to accommodate: orbit radius + circle radius + offset
    ylim_nsbh = source_orbit_radius_nsbh + RADIUS_LARGER_OBJECT + abs(source_vertical_offset_nsbh) + 0.3
    ax_nsbh.set_xlim(-ylim_nsbh, ylim_nsbh)
    ax_nsbh.set_ylim(-ylim_nsbh, ylim_nsbh)
    ax_nsbh.set_aspect('equal')
    ax_nsbh.axis('off')

    # Position circles on diagonal (upper-left BH, lower-right NS)
    # Upper left position (black hole)
    x1_nsbh = -source_orbit_radius_nsbh * np.cos(angle_diag)
    y1_nsbh = source_orbit_radius_nsbh * np.sin(angle_diag) + source_vertical_offset_nsbh
    # Lower right position (neutron star)
    x2_nsbh = source_orbit_radius_nsbh * np.cos(angle_diag)
    y2_nsbh = -source_orbit_radius_nsbh * np.sin(angle_diag) + source_vertical_offset_nsbh

    # Draw orbit line first (lower zorder so it appears behind circles)
    orbit_nsbh = Circle((0, source_vertical_offset_nsbh), source_orbit_radius_nsbh, fill=False, ec='black', linewidth=1.5,
                        linestyle='--', alpha=0.5, zorder=1)
    ax_nsbh.add_patch(orbit_nsbh)

    # Draw NSBH system (black hole upper-left, neutron star lower-right)
    # Use same sizes as BNS for consistency
    circle1_nsbh = Circle((x1_nsbh, y1_nsbh), RADIUS_LARGER_OBJECT, color='black', ec='black', linewidth=2, zorder=2)
    circle2_nsbh = Circle((x2_nsbh, y2_nsbh), RADIUS_SMALLER_OBJECT, color=NS_COLOR, ec='black', linewidth=2, zorder=2)
    ax_nsbh.add_patch(circle1_nsbh)
    ax_nsbh.add_patch(circle2_nsbh)

    # ===== Population Column: Mass Distributions =====
    # Uniform distribution (KDE from training data)
    pdf_uniform = uniform_mass_pdf(masses)
    ax_uniform.fill_between(masses, pdf_uniform, alpha=0.7, color=NS_COLOR, linewidth=2)
    ax_uniform.plot(masses, pdf_uniform, color=NS_COLOR, linewidth=2.5)
    ax_uniform.set_ylabel('Prob. density', fontsize=fs_population_ylabel)
    ax_uniform.set_title('Uniform', fontsize=fs_population_title)
    ax_uniform.set_xlim(m_min, m_max)
    ax_uniform.set_ylim(0, None)
    ax_uniform.tick_params(axis='x', labelbottom=False, labelsize=fs_xticks_population)
    ax_uniform.tick_params(axis='y', labelsize=fs_yticks_population)

    # Single Gaussian distribution
    pdf_gaussian = gaussian_mass_pdf(masses)
    ax_gaussian.fill_between(masses, pdf_gaussian, alpha=0.7, color=NS_COLOR, linewidth=2)
    ax_gaussian.plot(masses, pdf_gaussian, color=NS_COLOR, linewidth=2.5)
    ax_gaussian.set_ylabel('Prob. density', fontsize=fs_population_ylabel)
    ax_gaussian.set_title('Gaussian', fontsize=fs_population_title)
    ax_gaussian.set_xlim(m_min, m_max)
    ax_gaussian.set_ylim(0, None)
    ax_gaussian.tick_params(axis='x', labelbottom=False, labelsize=fs_xticks_population)
    ax_gaussian.tick_params(axis='y', labelsize=fs_yticks_population)

    # Double Gaussian distribution
    pdf_double = double_gaussian_mass_pdf(masses)
    ax_double.fill_between(masses, pdf_double, alpha=0.7, color=NS_COLOR, linewidth=2)
    ax_double.plot(masses, pdf_double, color=NS_COLOR, linewidth=2.5)
    ax_double.set_xlabel(r'Mass [$M_\odot$]', fontsize=fs_population_xlabel)
    ax_double.set_ylabel('Prob. density', fontsize=fs_population_ylabel)
    ax_double.set_title('Double Gaussian', fontsize=fs_population_title)
    ax_double.set_xlim(m_min, m_max)
    ax_double.set_ylim(0, None)
    ax_double.tick_params(axis='x', labelsize=fs_xticks_population)
    ax_double.tick_params(axis='y', labelsize=fs_yticks_population)

    # ===== EOS Column: Lambda(M) Curves =====
    # Define EOS color scheme (from plots/utils.py)
    EOS_COLORS = {
        "radio": "#0472b0",           # Blue
        "radio_chiEFT": "#de8f05",    # Orange
        "radio_NICER": "#ca7abc",     # Pink/purple
    }

    EOS_LABELS = {
        "radio": r"PSRs",
        "radio_chiEFT": r"PSRs+$\chi_{\rm{EFT}}$",
        "radio_NICER": r"PSRs+NICER"
    }

    # Load EOS data from each dataset
    script_dir = Path(__file__).parent
    eos_base_dir = script_dir.parent / "data" / "eos"

    eos_datasets = ["radio", "radio_chiEFT", "radio_NICER"]

    # Create mass array for computing credible intervals
    masses_array = np.linspace(M_MIN_EOS, M_MAX_EOS, N_MASSES_LAMBDAS_PLOT)

    # Store legend handles for solid color patches
    from matplotlib.patches import Patch
    legend_handles = []

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

        # Plot credible interval with fill_between (no label here to avoid alpha in legend)
        ax_eos.fill_between(masses_array, lambda_low, lambda_high,
                            alpha=eos_fill_alpha, color=color)

        # Draw boundary lines
        ax_eos.plot(masses_array, lambda_low, color=color, linewidth=eos_border_linewidth, alpha=eos_border_alpha)
        ax_eos.plot(masses_array, lambda_high, color=color, linewidth=eos_border_linewidth, alpha=eos_border_alpha)

        # Create custom legend handle with solid color (no alpha)
        legend_handles.append(Patch(facecolor=color, edgecolor=color, label=EOS_LABELS[eos_name]))

    # Styling
    ax_eos.set_xlabel(r'Mass [$M_\odot$]', fontsize=fs_eos_xlabel)
    ax_eos.set_ylabel(r'$\Lambda$', fontsize=fs_eos_ylabel)
    ax_eos.set_xlim(M_MIN_EOS, M_MAX_EOS)
    ax_eos.set_ylim(LAMBDA_MIN_EOS, LAMBDA_MAX_EOS)
    ax_eos.set_yscale('log')
    ax_eos.tick_params(axis='x', labelsize=fs_xticks_eos)
    ax_eos.tick_params(axis='y', labelsize=fs_yticks_eos)

    # Add legend for EOS datasets with custom handles (solid colors, no alpha)
    ax_eos.legend(handles=legend_handles, loc='upper right', fontsize=fs_eos_legend, framealpha=0.9)

    # ===== Add column headers and source labels =====
    # Import line for drawing underlines
    from matplotlib.lines import Line2D

    # Add BNS and NSBH labels (using fig.text to avoid affecting axis limits)
    fig.text(source_label_x_bns, source_label_y_bns, 'BNS', ha='center', va='center',
             fontsize=fs_source_label, fontweight='bold')
    fig.text(source_label_x_nsbh, source_label_y_nsbh, 'NSBH', ha='center', va='center',
             fontsize=fs_source_label, fontweight='bold')

    # Source header (left column)
    fig.text(header_x_source, header_y, 'Source', ha='center', va='center',
             fontsize=fs_column_headers, fontweight='bold')
    # Add underline for Source
    line_source = Line2D([header_x_source - underline_width_source/2, header_x_source + underline_width_source/2],
                         [header_y - underline_offset, header_y - underline_offset],
                         transform=fig.transFigure, color='black', linewidth=2)
    fig.add_artist(line_source)

    # Population header (middle column)
    fig.text(header_x_population, header_y, 'Population', ha='center', va='center',
             fontsize=fs_column_headers, fontweight='bold')
    # Add underline for Population
    line_population = Line2D([header_x_population - underline_width_population/2, header_x_population + underline_width_population/2],
                             [header_y - underline_offset, header_y - underline_offset],
                             transform=fig.transFigure, color='black', linewidth=2)
    fig.add_artist(line_population)

    # EOS header (right column)
    fig.text(header_x_eos, header_y, 'EOS', ha='center', va='center',
             fontsize=fs_column_headers, fontweight='bold')
    # Add underline for EOS
    line_eos = Line2D([header_x_eos - underline_width_eos/2, header_x_eos + underline_width_eos/2],
                      [header_y - underline_offset, header_y - underline_offset],
                      transform=fig.transFigure, color='black', linewidth=2)
    fig.add_artist(line_eos)

    # ===== Add curly bracket at bottom spanning all columns =====
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patches as mpatches

    # Create curly bracket using annotation
    # We'll draw it as two annotations forming a brace shape
    ax_bracket = fig.add_axes([0, 0, 1, 1], facecolor='none')
    ax_bracket.set_xlim(0, 1)
    ax_bracket.set_ylim(0, 1)
    ax_bracket.axis('off')

    # Draw curly bracket using FancyBboxPatch
    # Alternative: draw it manually with path or use annotation with bracket connectionstyle
    from matplotlib.patches import ConnectionPatch

    # Use annotate to create bracket spanning the columns
    ax_bracket.annotate('', xy=(bracket_x_left, bracket_y), xytext=(bracket_x_right, bracket_y),
                       xycoords='figure fraction', textcoords='figure fraction',
                       arrowprops=dict(arrowstyle='-', lw=0, shrinkA=0, shrinkB=0),
                       annotation_clip=False)

    # Manual curly bracket drawing using path
    from matplotlib.path import Path as MplPath
    import matplotlib.patches as patches

    # Define curly bracket path with flat sides and sharp center dip
    bracket_center = (bracket_x_left + bracket_x_right) / 2
    total_width = bracket_x_right - bracket_x_left

    # Define turning points
    left_turn = bracket_x_left + bracket_flat_length * total_width
    right_turn = bracket_x_right - bracket_flat_length * total_width

    # Curly bracket with sharp inflection: flat sides, sharp downward turn
    verts = [
        (bracket_x_left, bracket_y),  # Start (left)
        (left_turn, bracket_y),  # Flat section left
        (left_turn + 0.01, bracket_y),  # Control point for sharp turn
        (bracket_center - 0.01, bracket_y - bracket_height),  # Control point approaching center
        (bracket_center, bracket_y - bracket_height),  # Center bottom point
        (bracket_center + 0.01, bracket_y - bracket_height),  # Control point leaving center
        (right_turn - 0.01, bracket_y),  # Control point for sharp turn
        (right_turn, bracket_y),  # Flat section right
        (bracket_x_right, bracket_y),  # End (right)
    ]

    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,  # Flat left section
        MplPath.CURVE4,  # Sharp turn down
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,  # Sharp turn up
        MplPath.CURVE4,
        MplPath.LINETO,  # Flat right section
        MplPath.LINETO,
    ]

    path = MplPath(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', edgecolor='black', lw=2,
                              transform=fig.transFigure, clip_on=False)
    fig.add_artist(patch)

    # Add text below the bracket
    fig.text(bracket_center, bracket_text_y, r'$\pi_{\rm{NF}}(m_1^{\rm{src}}, m_2^{\rm{src}}, \Lambda_1, \Lambda_2)$',
             ha='center', va='top', fontsize=fs_bracket_text, transform=fig.transFigure)

    if save:
        output_path = OUTPUT_DIR / "Figure1.pdf"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved figure to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    create_mass_distributions(save=True, show=False)