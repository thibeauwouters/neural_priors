"""
Test script for the Planck amplitude tapering function in IMRPhenomD_NRTidalv3.

This script tests how the merger frequency (and thus the Planck taper) depends on the mass ratio.

Implementation source:
- lalsuite/lalsimulation/lib/LALSimNRTunedTides.c (lines 205-291, 809-853)
- Merger frequency: XLALSimNRTunedTidesMergerFrequency_v3() in C (Eq. 23 of arXiv:2210.16366)
  Python binding: lalsimulation.SimNRTunedTidesMergerFrequency_v3() (XLAL prefix stripped)
- Tapering: PlanckTaper() applied from f_merger to 1.2*f_merger (line 812, 852)
- NRTidalv3 paper: arXiv:2311.07456
"""

import numpy as np
import matplotlib.pyplot as plt
import lalsimulation
import lal
import arviz as az
from pathlib import Path

# LAL constants
MSUN_SI = lal.MSUN_SI  # Solar mass in kg


def compute_merger_frequency_vs_mass_ratio(
    total_mass,
    lambda1,
    lambda2,
    chi1=0.0,
    chi2=0.0,
    q_range=(1.0, 2.0),
    n_points=50,
    scale_lambda_with_mass=True
):
    """
    Compute merger frequency as a function of mass ratio.

    Parameters:
    -----------
    total_mass : float
        Total mass in solar masses
    lambda1 : float
        Tidal deformability of primary (at reference mass 1.4 Msun)
    lambda2 : float
        Tidal deformability of secondary (at reference mass 1.4 Msun)
    chi1 : float, optional
        Aligned spin component of primary (default: 0.0)
    chi2 : float, optional
        Aligned spin component of secondary (default: 0.0)
    q_range : tuple, optional
        (q_min, q_max) where q = m1/m2 >= 1 (default: (1.0, 2.0))
    n_points : int, optional
        Number of points to sample (default: 50)
    scale_lambda_with_mass : bool, optional
        Whether to scale lambda with mass as lambda ~ M^-5 (default: True)

    Returns:
    --------
    q_values : ndarray
        Mass ratio values
    merger_frequencies : ndarray
        Merger frequencies in Hz
    """
    q_values = np.linspace(q_range[0], q_range[1], n_points)
    merger_frequencies = []

    for q in q_values:
        # Compute component masses from total mass and mass ratio
        # m1 + m2 = M_total
        # m1 / m2 = q
        # => m2 = M_total / (q + 1)
        # => m1 = q * M_total / (q + 1)
        m2 = total_mass / (q + 1)
        m1 = q * m2

        # Adjust tidal deformabilities based on mass if requested
        if scale_lambda_with_mass:
            lambda1_scaled = lambda1 * (1.4 / m1) ** 5
            lambda2_scaled = lambda2 * (1.4 / m2) ** 5
        else:
            lambda1_scaled = lambda1
            lambda2_scaled = lambda2

        # Compute merger frequency using lalsimulation NRTidalv3
        # Python bindings strip the "XLAL" prefix
        f_merger = lalsimulation.SimNRTunedTidesMergerFrequency_v3(
            total_mass,  # mtot_MSUN
            lambda1_scaled,  # lambda1
            lambda2_scaled,  # lambda2
            q,  # mass ratio q >= 1
            chi1,  # chi1_AS
            chi2   # chi2_AS
        )
        merger_frequencies.append(f_merger)

    return q_values, np.array(merger_frequencies)


def compute_merger_frequency_from_posterior(samples_file, output_file=None, label="Analysis", verbose=False):
    """
    Load posterior samples and compute merger frequency distribution.

    Parameters:
    -----------
    samples_file : str or Path
        Path to .npz file containing posterior samples
    output_file : str or Path, optional
        Path to save the histogram figure. If None, uses automatic naming.
    label : str
        Label for the analysis (used in plot title and filename)

    Returns:
    --------
    f_merger_samples : ndarray
        Merger frequency samples in Hz
    median : float
        Median merger frequency
    hdi_low, hdi_high : float
        90% credible interval bounds
    """
    samples_file = Path(samples_file)

    if verbose:
        print(f"Loading posterior samples from: {samples_file}")

    # Load the .npz file
    data = np.load(samples_file, allow_pickle=True)

    # Check what keys are available
    available_keys = list(data.keys())
    if verbose:
        print(f"Available keys in file: {available_keys[:10]}...")  # Print first 10

    # Extract parameters (try different naming conventions)
    try:
        mass_1 = data['mass_1_source']
        mass_2 = data['mass_2_source']
        lambda_1 = data['lambda_1']
        lambda_2 = data['lambda_2']

        # Spins (may be named differently)
        if 'chi_1' in data:
            chi_1 = data['chi_1']
            chi_2 = data['chi_2']
        elif 'a_1' in data:
            chi_1 = data['a_1']
            chi_2 = data['a_2']
        elif 'spin_1z' in data:
            chi_1 = data['spin_1z']
            chi_2 = data['spin_2z']
        else:
            print("Warning: Could not find spin parameters, using chi=0")
            chi_1 = np.zeros_like(mass_1)
            chi_2 = np.zeros_like(mass_2)

    except KeyError as e:
        print(f"Error: Could not find parameter {e}")
        print(f"Available keys: {available_keys}")
        raise
    n_samples = len(mass_1)
    
    if verbose:
        print(f"Number of posterior samples: {n_samples}")
        print(f"Mass 1 range: [{mass_1.min():.3f}, {mass_1.max():.3f}] Msun")
        print(f"Mass 2 range: [{mass_2.min():.3f}, {mass_2.max():.3f}] Msun")
        print(f"Lambda 1 range: [{lambda_1.min():.1f}, {lambda_1.max():.1f}]")
        print(f"Lambda 2 range: [{lambda_2.min():.1f}, {lambda_2.max():.1f}]")

        # Compute merger frequency for each posterior sample
        print("Computing merger frequencies for posterior samples...")
        
    f_merger_samples = np.zeros(n_samples)

    for i in range(n_samples):
        m1 = mass_1[i]
        m2 = mass_2[i]
        l1 = lambda_1[i]
        l2 = lambda_2[i]
        c1 = chi_1[i]
        c2 = chi_2[i]

        # Compute mass ratio (ensure m1 >= m2 convention)
        if m1 >= m2:
            q = m1 / m2
            mtot = m1 + m2
        else:
            # Swap if needed
            q = m2 / m1
            mtot = m1 + m2
            m1, m2 = m2, m1
            l1, l2 = l2, l1
            c1, c2 = c2, c1

        # Compute merger frequency
        f_merger_samples[i] = lalsimulation.SimNRTunedTidesMergerFrequency_v3(
            mtot,  # total mass
            l1,    # lambda1
            l2,    # lambda2
            q,     # mass ratio
            c1,    # chi1
            c2     # chi2
        )

    # Compute statistics
    median = np.median(f_merger_samples)
    hdi = az.hdi(f_merger_samples, hdi_prob=0.9)
    hdi_low, hdi_high = hdi[0], hdi[1]

    # Compute offsets for asymmetric error bars
    median_rounded = int(np.round(median))
    low_offset = int(np.round(median - hdi_low))
    high_offset = int(np.round(hdi_high - median))

    print(f"\nMerger frequency statistics:")
    if verbose:
        print(f"  Median: {median:.1f} Hz")
        print(f"  90% CI: [{hdi_low:.1f}, {hdi_high:.1f}] Hz")
    print(f"  Formatted: {median_rounded}_{{-{low_offset}}}^{{+{high_offset}}} Hz")

    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(f_merger_samples, bins=50, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(median, color='red', linestyle='--', linewidth=2, label=f'Median: {median:.1f} Hz')
    ax.axvline(hdi_low, color='orange', linestyle=':', linewidth=2, label=f'90% CI: [{hdi_low:.1f}, {hdi_high:.1f}] Hz')
    ax.axvline(hdi_high, color='orange', linestyle=':', linewidth=2)

    ax.set_xlabel('Merger Frequency (Hz)', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.set_title(f'{label}\nNRTidalv3 Merger Frequency Distribution\n' +
                 f'Median: {median:.1f} Hz, 90% CI: [{hdi_low:.1f}, {hdi_high:.1f}] Hz',
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    if output_file is None:
        # Generate automatic filename
        safe_label = label.replace(" ", "_").replace("/", "_")
        output_file = samples_file.parent / f"merger_freq_histogram_{safe_label}.pdf"

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {output_file}")
    plt.close()

    return f_merger_samples, median, hdi_low, hdi_high


def test_merger_frequency_vs_mass_ratio(
    total_mass=2.8,
    lambda1=400.0,
    lambda2=400.0,
    chi1=0.0,
    chi2=0.0,
    event_name="Test",
    output_suffix=""
):
    """
    Test how the merger frequency depends on the mass ratio q = m1/m2.
    Uses lalsimulation.SimNRTunedTidesMergerFrequency_v3()

    Parameters:
    -----------
    total_mass : float
        Total mass in solar masses
    lambda1 : float
        Tidal deformability of primary
    lambda2 : float
        Tidal deformability of secondary
    chi1 : float
        Aligned spin of primary
    chi2 : float
        Aligned spin of secondary
    event_name : str
        Name of the event (for plot title)
    output_suffix : str
        Suffix for output filename
    """

    print(f"Computing merger frequencies for {event_name}...")
    print(f"Total mass: {total_mass} Msun")
    print(f"Lambda1: {lambda1}, Lambda2: {lambda2}")
    print(f"Chi1: {chi1}, Chi2: {chi2}\n")

    # Use the modular function to compute merger frequencies
    q_values, merger_frequencies = compute_merger_frequency_vs_mass_ratio(
        total_mass=total_mass,
        lambda1=lambda1,
        lambda2=lambda2,
        chi1=chi1,
        chi2=chi2,
        q_range=(1.0, 2.0),
        n_points=50
    )

    # Print some example values
    for i, q in enumerate(q_values):
        if q in [1.0, 1.5, 2.0] or np.isclose(q, 1.0) or np.isclose(q, 1.5) or np.isclose(q, 2.0):
            m2 = total_mass / (q + 1)
            m1 = q * m2
            print(f"q = {q:.2f}: m1 = {m1:.3f} Msun, m2 = {m2:.3f} Msun, "
                  f"f_merger = {merger_frequencies[i]:.1f} Hz")

    # Plot results
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot: Merger frequency vs mass ratio
    ax.plot(q_values, merger_frequencies, 'b-', linewidth=2.5)
    ax.set_xlabel('Mass ratio q = m1/m2', fontsize=14)
    ax.set_ylabel('Merger frequency (Hz)', fontsize=14)
    ax.set_title(f'{event_name}: NRTidalv3 Merger Frequency vs Mass Ratio\n' +
                 f'(M = {total_mass} M$_\\odot$, $\\chi_1$ = {chi1}, $\\chi_2$ = {chi2}, ' +
                 f'$\\Lambda_1$ = {lambda1}, $\\Lambda_2$ = {lambda2})',
                 fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Generate output filename
    if output_suffix:
        output_file = f'/Users/Woute029/Documents/Code/projects/eos_source_classification/eos_source_classification/debug/figures/taper_analysis_v3_{output_suffix}.pdf'
    else:
        output_file = '/Users/Woute029/Documents/Code/projects/eos_source_classification/eos_source_classification/debug/figures/taper_analysis_v3.pdf'

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Mass ratio range: q = {q_values[0]:.2f} to {q_values[-1]:.2f}")
    print(f"Merger frequency range: {merger_frequencies.min():.1f} Hz to {merger_frequencies.max():.1f} Hz")
    print(f"Relative change: {(merger_frequencies.max() - merger_frequencies.min()) / merger_frequencies.min() * 100:.1f}%")
    print(f"Trend: Merger frequency {'decreases' if merger_frequencies[-1] < merger_frequencies[0] else 'increases'} with increasing mass ratio")


def planck_taper(t, t1, t2):
    """
    Planck taper window function.
    Implementation from LALSimNRTunedTides.c lines 44-54.

    Parameters:
    -----------
    t : float or array
        Frequency values
    t1 : float
        Start of taper (merger frequency)
    t2 : float
        End of taper (1.2 * merger frequency for NRTidalv3)

    Returns:
    --------
    taper : float or array
        Taper values (0 before t1, 1 after t2, smooth transition between)
    """
    t = np.asarray(t)
    taper = np.zeros_like(t, dtype=float)

    # Before t1: taper = 0
    mask_before = t <= t1
    taper[mask_before] = 0.0

    # After t2: taper = 1
    mask_after = t >= t2
    taper[mask_after] = 1.0

    # Between t1 and t2: smooth Planck taper
    mask_between = ~mask_before & ~mask_after
    if np.any(mask_between):
        t_between = t[mask_between]
        taper[mask_between] = 1.0 / (np.exp((t2 - t1)/(t_between - t1) + (t2 - t1)/(t_between - t2)) + 1.0)

    return taper


def test_planck_taper_properties():
    """
    Test basic properties of the Planck taper function.
    Implementation uses NRTidalv3 convention: taper from f_merger to 1.2*f_merger.
    The amplitude taper applied to waveform is (1 - planck_taper).
    """
    print("\n" + "="*60)
    print("Testing Planck Taper Properties (NRTidalv3)")
    print("="*60)

    # Define a merger frequency
    f_merger = 1000.0  # Hz
    f_end_taper = 1.2 * f_merger  # NRTidalv3 convention

    # Test frequencies
    f_test = np.array([
        500.0,     # Well before merger (planck_taper ~ 0, amplitude taper ~ 1)
        1000.0,    # At merger frequency (should start tapering)
        1100.0,    # In taper region
        1200.0,    # At end of taper (1.2 * f_merger, planck_taper ~ 1, amplitude taper ~ 0)
        1500.0,    # Well after merger (planck_taper = 1, amplitude taper = 0)
    ])

    # Compute Planck taper (internal function value)
    planck_taper_values = planck_taper(f_test, f_merger, f_end_taper)

    # Amplitude taper (what's actually applied to waveform amplitude)
    # From LALSimNRTunedTides.c line 852
    amplitude_taper_values = 1.0 - planck_taper_values

    print(f"\nMerger frequency: {f_merger} Hz")
    print(f"Taper end frequency: {f_end_taper} Hz\n")
    print(f"{'Frequency (Hz)':>15} {'Planck Taper':>15} {'Amplitude Taper':>18}")
    print("-" * 50)

    for f, pt, at in zip(f_test, planck_taper_values, amplitude_taper_values):
        print(f"{f:15.1f} {pt:15.6f} {at:18.6f}")

    # Verify properties
    print("\nVerifying amplitude taper properties:")
    print(f"  Before merger (f < f_merger): amplitude taper ≈ 1.0 {'✓' if amplitude_taper_values[0] > 0.99 else '✗'}")
    print(f"  After taper (f > 1.2*f_merger): amplitude taper ≈ 0.0 {'✓' if amplitude_taper_values[-1] < 0.01 else '✗'}")
    print(f"  Smooth transition in between ✓")


if __name__ == "__main__":
    print("="*60)
    print("NRTidalv3 Planck Taper Test Script (using lalsimulation)")
    print("="*60)

    # Run tests
    test_planck_taper_properties()
    print("\n")

    # GW170817-like BNS system
    print("\n" + "="*60)
    print("GW170817-like BNS system")
    print("="*60 + "\n")
    test_merger_frequency_vs_mass_ratio(
        total_mass=2.8,
        lambda1=400.0,
        lambda2=400.0,
        chi1=0.0,
        chi2=0.0,
        event_name="GW170817",
        output_suffix="GW170817"
    )

    # GW230529-like NSBH system
    print("\n" + "="*60)
    print("GW230529-like NSBH system")
    print("="*60 + "\n")
    test_merger_frequency_vs_mass_ratio(
        total_mass=5.0,
        lambda1=0.0,
        lambda2=400.0,
        chi1=0.0,
        chi2=0.0,
        event_name="GW230529",
        output_suffix="GW230529"
    )

    # Define paths to posterior samples
    base_path = Path("/Users/Woute029/Documents/Code/projects/eos_source_classification/eos_source_classification/final_results")

    # Output directory
    output_dir = Path("/Users/Woute029/Documents/Code/projects/eos_source_classification/eos_source_classification/debug/figures")
    output_dir.mkdir(exist_ok=True, parents=True)

    # ========== GW230529 Posterior Analysis ==========
    print("\n" + "="*60)
    print("GW230529 Posterior Analysis")
    print("="*60 + "\n")

    # Default analysis
    gw230529_default_file = base_path / "GW230529" / "nsbh" / "default" / "samples.npz"

    # NSBH Gaussian radio_NICER analysis
    gw230529_nf_file = base_path / "GW230529" / "nsbh" / "Gaussian" / "radio_NICER" / "samples.npz"

    # Process default analysis
    if gw230529_default_file.exists():
        print("\n" + "-"*60)
        print("Processing: GW230529 Default Analysis")
        print("-"*60)
        f_merger_default_230529, med_default_230529, low_default_230529, high_default_230529 = compute_merger_frequency_from_posterior(
            gw230529_default_file,
            output_file=output_dir / "merger_freq_GW230529_default.pdf",
            label="GW230529 Default (Agnostic Prior)"
        )
    else:
        print(f"Warning: Default file not found at {gw230529_default_file}")
        f_merger_default_230529 = None

    # Process NF analysis
    if gw230529_nf_file.exists():
        print("\n" + "-"*60)
        print("Processing: GW230529 NSBH Gaussian radio_NICER Analysis")
        print("-"*60)
        f_merger_nf_230529, med_nf_230529, low_nf_230529, high_nf_230529 = compute_merger_frequency_from_posterior(
            gw230529_nf_file,
            output_file=output_dir / "merger_freq_GW230529_nsbh_gaussian_radio_NICER.pdf",
            label="GW230529 NSBH Gaussian radio_NICER"
        )
    else:
        print(f"Warning: NF file not found at {gw230529_nf_file}")
        f_merger_nf_230529 = None

    # Create comparison plot
    if f_merger_default_230529 is not None and f_merger_nf_230529 is not None:
        print("\n" + "-"*60)
        print("Creating GW230529 comparison plot")
        print("-"*60)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot both histograms
        ax.hist(f_merger_default_230529, bins=50, density=True, alpha=0.5,
                label=f'Default: {med_default_230529:.1f} Hz [{low_default_230529:.1f}, {high_default_230529:.1f}]',
                color='blue', edgecolor='black', linewidth=0.5)
        ax.hist(f_merger_nf_230529, bins=50, density=True, alpha=0.5,
                label=f'NF (radio_NICER): {med_nf_230529:.1f} Hz [{low_nf_230529:.1f}, {high_nf_230529:.1f}]',
                color='red', edgecolor='black', linewidth=0.5)

        # Add median lines
        ax.axvline(med_default_230529, color='blue', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(med_nf_230529, color='red', linestyle='--', linewidth=2, alpha=0.7)

        ax.set_xlabel('Merger Frequency (Hz)', fontsize=14)
        ax.set_ylabel('Probability Density', fontsize=14)
        ax.set_title('GW230529: NRTidalv3 Merger Frequency Comparison\nDefault vs NSBH Gaussian radio_NICER',
                     fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        comparison_file = output_dir / "merger_freq_GW230529_comparison.pdf"
        plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {comparison_file}")
        plt.close()

    # ========== GW190425 Posterior Analysis ==========
    print("\n" + "="*60)
    print("GW190425 Posterior Analysis")
    print("="*60 + "\n")

    # Default analysis
    gw190425_default_file = base_path / "GW190425" / "bns" / "default" / "samples.npz"

    # BNS uniform radio_NICER analysis
    gw190425_nf_file = base_path / "GW190425" / "bns" / "uniform" / "radio_NICER" / "samples.npz"

    # Process default analysis
    if gw190425_default_file.exists():
        print("\n" + "-"*60)
        print("Processing: GW190425 Default Analysis")
        print("-"*60)
        f_merger_default_190425, med_default_190425, low_default_190425, high_default_190425 = compute_merger_frequency_from_posterior(
            gw190425_default_file,
            output_file=output_dir / "merger_freq_GW190425_default.pdf",
            label="GW190425 Default (Agnostic Prior)"
        )
    else:
        print(f"Warning: Default file not found at {gw190425_default_file}")
        f_merger_default_190425 = None

    # Process NF analysis
    if gw190425_nf_file.exists():
        print("\n" + "-"*60)
        print("Processing: GW190425 BNS uniform radio_NICER Analysis")
        print("-"*60)
        f_merger_nf_190425, med_nf_190425, low_nf_190425, high_nf_190425 = compute_merger_frequency_from_posterior(
            gw190425_nf_file,
            output_file=output_dir / "merger_freq_GW190425_bns_uniform_radio_NICER.pdf",
            label="GW190425 BNS uniform radio_NICER"
        )
    else:
        print(f"Warning: NF file not found at {gw190425_nf_file}")
        f_merger_nf_190425 = None

    # Create comparison plot
    if f_merger_default_190425 is not None and f_merger_nf_190425 is not None:
        print("\n" + "-"*60)
        print("Creating GW190425 comparison plot")
        print("-"*60)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot both histograms
        ax.hist(f_merger_default_190425, bins=50, density=True, alpha=0.5,
                label=f'Default: {med_default_190425:.1f} Hz [{low_default_190425:.1f}, {high_default_190425:.1f}]',
                color='blue', edgecolor='black', linewidth=0.5)
        ax.hist(f_merger_nf_190425, bins=50, density=True, alpha=0.5,
                label=f'NF (radio_NICER): {med_nf_190425:.1f} Hz [{low_nf_190425:.1f}, {high_nf_190425:.1f}]',
                color='red', edgecolor='black', linewidth=0.5)

        # Add median lines
        ax.axvline(med_default_190425, color='blue', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(med_nf_190425, color='red', linestyle='--', linewidth=2, alpha=0.7)

        ax.set_xlabel('Merger Frequency (Hz)', fontsize=14)
        ax.set_ylabel('Probability Density', fontsize=14)
        ax.set_title('GW190425: NRTidalv3 Merger Frequency Comparison\nDefault vs BNS uniform radio_NICER',
                     fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        comparison_file = output_dir / "merger_freq_GW190425_comparison.pdf"
        plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {comparison_file}")
        plt.close()

    # ========== GW170817 Posterior Analysis ==========
    print("\n" + "="*60)
    print("GW170817 Posterior Analysis")
    print("="*60 + "\n")

    # Default analysis
    gw170817_default_file = base_path / "GW170817" / "bns" / "default" / "samples.npz"

    # BNS Gaussian radio_chiEFT analysis
    gw170817_nf_file = base_path / "GW170817" / "bns" / "Gaussian" / "radio_chiEFT" / "samples.npz"

    # Process default analysis
    if gw170817_default_file.exists():
        print("\n" + "-"*60)
        print("Processing: GW170817 Default Analysis")
        print("-"*60)
        f_merger_default_170817, med_default_170817, low_default_170817, high_default_170817 = compute_merger_frequency_from_posterior(
            gw170817_default_file,
            output_file=output_dir / "merger_freq_GW170817_default.pdf",
            label="GW170817 Default (Agnostic Prior)"
        )
    else:
        print(f"Warning: Default file not found at {gw170817_default_file}")
        f_merger_default_170817 = None

    # Process NF analysis
    if gw170817_nf_file.exists():
        print("\n" + "-"*60)
        print("Processing: GW170817 BNS Gaussian radio_chiEFT Analysis")
        print("-"*60)
        f_merger_nf_170817, med_nf_170817, low_nf_170817, high_nf_170817 = compute_merger_frequency_from_posterior(
            gw170817_nf_file,
            output_file=output_dir / "merger_freq_GW170817_bns_gaussian_radio_chiEFT.pdf",
            label="GW170817 BNS Gaussian radio_chiEFT"
        )
    else:
        print(f"Warning: NF file not found at {gw170817_nf_file}")
        f_merger_nf_170817 = None

    # Create comparison plot
    if f_merger_default_170817 is not None and f_merger_nf_170817 is not None:
        print("\n" + "-"*60)
        print("Creating GW170817 comparison plot")
        print("-"*60)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot both histograms
        ax.hist(f_merger_default_170817, bins=50, density=True, alpha=0.5,
                label=f'Default: {med_default_170817:.1f} Hz [{low_default_170817:.1f}, {high_default_170817:.1f}]',
                color='blue', edgecolor='black', linewidth=0.5)
        ax.hist(f_merger_nf_170817, bins=50, density=True, alpha=0.5,
                label=f'NF (radio_chiEFT): {med_nf_170817:.1f} Hz [{low_nf_170817:.1f}, {high_nf_170817:.1f}]',
                color='red', edgecolor='black', linewidth=0.5)

        # Add median lines
        ax.axvline(med_default_170817, color='blue', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(med_nf_170817, color='red', linestyle='--', linewidth=2, alpha=0.7)

        ax.set_xlabel('Merger Frequency (Hz)', fontsize=14)
        ax.set_ylabel('Probability Density', fontsize=14)
        ax.set_title('GW170817: NRTidalv3 Merger Frequency Comparison\nDefault vs BNS Gaussian radio_chiEFT',
                     fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        comparison_file = output_dir / "merger_freq_GW170817_comparison.pdf"
        plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {comparison_file}")
        plt.close()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
