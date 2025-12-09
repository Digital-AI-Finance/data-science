"""
Create Open Graph image for social sharing
- Size: 1200x630 (standard OG dimensions)
- Shows course title, stats, and branding
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Configure for better text rendering
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
})

def create_og_image():
    """Create Open Graph image for course."""
    fig, ax = plt.subplots(figsize=(12, 6.3), dpi=100)

    # Background gradient effect using patches
    gradient = patches.FancyBboxPatch(
        (0, 0), 1, 1, transform=ax.transAxes,
        boxstyle="square,pad=0",
        facecolor='#1a1a2e', edgecolor='none'
    )
    ax.add_patch(gradient)

    # Add subtle accent bar at top
    accent_bar = patches.Rectangle(
        (0, 0.92), 1, 0.08, transform=ax.transAxes,
        facecolor='#3498db', edgecolor='none', alpha=0.8
    )
    ax.add_patch(accent_bar)

    # Main title
    ax.text(0.5, 0.70, 'Data Science with Python',
            fontsize=48, fontweight='bold', color='white',
            ha='center', va='center', transform=ax.transAxes)

    # Subtitle
    ax.text(0.5, 0.55, 'Complete BSc Course Curriculum',
            fontsize=24, color='#a0a0a0',
            ha='center', va='center', transform=ax.transAxes)

    # Stats boxes
    stats = [
        ('48', 'Lessons'),
        ('534', 'Slides'),
        ('384', 'Charts'),
        ('10', 'Modules')
    ]

    box_width = 0.18
    start_x = 0.14
    spacing = 0.19

    for i, (number, label) in enumerate(stats):
        x = start_x + i * spacing

        # Box background
        box = patches.FancyBboxPatch(
            (x - box_width/2, 0.20), box_width, 0.22,
            transform=ax.transAxes,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor='#16213e', edgecolor='#3498db',
            linewidth=2, alpha=0.9
        )
        ax.add_patch(box)

        # Number
        ax.text(x, 0.36, number,
                fontsize=32, fontweight='bold', color='#3498db',
                ha='center', va='center', transform=ax.transAxes)

        # Label
        ax.text(x, 0.26, label,
                fontsize=14, color='#a0a0a0',
                ha='center', va='center', transform=ax.transAxes)

    # Footer branding
    ax.text(0.5, 0.06, 'Digital-AI-Finance',
            fontsize=14, color='#606060',
            ha='center', va='center', transform=ax.transAxes)

    # Remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Save
    output_path = Path(__file__).parent / 'og-image.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0,
                facecolor='#1a1a2e')
    plt.close()

    print(f"Created: {output_path}")
    print(f"Size: 1200x630 pixels")

if __name__ == '__main__':
    create_og_image()
