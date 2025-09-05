#!/usr/bin/env python3
"""
Create a vertical colorbar showing Jeffrey's scale colors with evidence conclusions.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Jeffrey's scale colors from the TeX definitions
jeffreys_colors = {
    'jeffreysred1': '#f6cdb0',  # barely worth mentioning
    'jeffreysred2': '#fc9074',  # substantial  
    'jeffreysred3': '#f4744c',  # strong
    'jeffreysred4': '#ef5b43',  # very strong
    'jeffreysred5': '#f52a44'   # decisive
}

# Evidence conclusions for each color level
evidence_labels = [
    'Barely worth mentioning',
    'Substantial', 
    'Strong',
    'Very strong',
    'Decisive'
]

# Corresponding Jeffrey's scale ranges (log10 Bayes factor)
bf_ranges = [
    '0.0 - 0.5',
    '0.5 - 1.0', 
    '1.0 - 1.5',
    '1.5 - 2.0',
    '> 2.0'
]

def create_jeffreys_colorbar():
    """Create minimal vertical colorbar using legend patches."""
    
    from matplotlib.patches import Rectangle
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # Create colored rectangles and labels
    colors = list(reversed(list(jeffreys_colors.values())))
    reversed_labels = list(reversed(evidence_labels))
    
    # Create legend patches
    patches = []
    for color, label in zip(colors, reversed_labels):
        patch = Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=1)
        patches.append(patch)
    
    # Create legend with proper sizing
    legend = ax.legend(patches, reversed_labels, 
                      loc='center left',
                      bbox_to_anchor=(0.1, 0.5),
                      frameon=False,
                      fontsize=16,
                      handlelength=3,
                      handletextpad=1.0,
                      labelspacing=1.5)
    
    # Remove the main plot axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

if __name__ == "__main__":
    fig = create_jeffreys_colorbar()
    
    # Save the colorbar
    output_file = "jeffreys_colorbar.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Colorbar saved to: {output_file}")
    
    # Also save as PNG
    output_png = "jeffreys_colorbar.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Colorbar saved to: {output_png}")
    
    plt.show()