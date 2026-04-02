import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Canvas: 6 inches wide, 300 DPI → 1800 px wide ──────────────────────
fig, ax = plt.subplots(figsize=(6, 8.5), dpi=300)
ax.set_xlim(0, 6)
ax.set_ylim(0, 8.5)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Color palette ────────────────────────────────────────────────────────
C_GRAY   = '#F1EFE8'   # neutral boxes
C_BLUE   = '#E6F1FB'   # feature / classifier
C_PURPLE = '#EEEDFE'   # GBM
C_AMBER  = '#FAEEDA'   # rule engine
C_CORAL  = '#FAECE7'   # min()
C_TEAL   = '#E1F5EE'   # recommended set / GBM direct
C_GREEN  = '#EAF3DE'   # final output
BORDER   = '#888780'

def box(ax, x, y, w, h, text, color, fontsize=6.5, bold=False, sub=None):
    """Draw a rounded rect with centered text (and optional subtitle)."""
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.04", linewidth=0.5,
                          edgecolor=BORDER, facecolor=color, zorder=2)
    ax.add_patch(rect)
    if sub:
        ax.text(x, y + h*0.12, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold' if bold else 'normal',
                color='#2C2C2A', zorder=3)
        ax.text(x, y - h*0.18, sub, ha='center', va='center',
                fontsize=fontsize - 1.0, color='#5F5E5A', zorder=3)
    else:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold' if bold else 'normal',
                color='#2C2C2A', zorder=3)

def diamond(ax, x, y, w, h, line1, line2):
    """Draw a diamond decision shape."""
    diamond_pts = [(x, y+h/2), (x+w/2, y), (x, y-h/2), (x-w/2, y)]
    poly = plt.Polygon(diamond_pts, closed=True, linewidth=0.5,
                       edgecolor=BORDER, facecolor='#FFFFFF', zorder=2)
    ax.add_patch(poly)
    ax.text(x, y + h*0.1, line1, ha='center', va='center',
            fontsize=6.5, fontweight='bold', color='#2C2C2A', zorder=3)
    ax.text(x, y - h*0.18, line2, ha='center', va='center',
            fontsize=5.8, color='#5F5E5A', zorder=3)

def arrow(ax, x1, y1, x2, y2, color='#444441', lw=0.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=7),
                zorder=3)

def larrow(ax, points, color='#444441', lw=0.8):
    """L-shaped or custom path arrow — draw segments then arrowhead at end."""
    for i in range(len(points)-1):
        x1,y1 = points[i]; x2,y2 = points[i+1]
        if i < len(points)-2:
            ax.plot([x1,x2],[y1,y2], color=color, lw=lw, zorder=3, solid_capstyle='round')
        else:
            ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=lw, mutation_scale=7), zorder=3)

# ─── Y positions (top → bottom) ─────────────────────────────────────────
Y_IN   = 8.05   # Patient input
Y_FE   = 7.35   # Feature extraction
Y_GBM  = 6.55   # GBM classifier
Y_DIA  = 5.60   # Confidence diamond
Y_RULE = 5.60   # Rule engine (same row as diamond)
Y_GBMD = 5.60   # GBM direct (same row)
Y_MIN  = 4.65   # min() box
Y_REC  = 4.65   # Recommended set (same row as min)
Y_CLI  = 3.70   # Clinician review
Y_OUT  = 2.90   # Final output

BH = 0.42       # standard box height
BH2 = 0.50      # taller box height

# ─── Row 1: patient input ────────────────────────────────────────────────
box(ax, 3.0, Y_IN,  3.2, BH,  'Patient questionnaire responses', C_GRAY)
arrow(ax, 3.0, Y_IN-BH/2, 3.0, Y_FE+BH/2)

# ─── Row 2: feature extraction ───────────────────────────────────────────
box(ax, 3.0, Y_FE, 3.2, BH, 'Feature extraction',
    C_BLUE, sub='42 items + age, gender, TIPI')
arrow(ax, 3.0, Y_FE-BH/2, 3.0, Y_GBM+BH2/2)

# ─── Row 3: GBM classifier ───────────────────────────────────────────────
box(ax, 3.0, Y_GBM, 4.0, BH2, 'LightGBM multi-label classifier',
    C_PURPLE, bold=True, sub='Predict [dep, anx, stress] + confidence score')
arrow(ax, 3.0, Y_GBM-BH2/2, 3.0, Y_DIA+0.47)

# ─── Row 4: confidence diamond ───────────────────────────────────────────
diamond(ax, 3.0, Y_DIA, 1.20, 0.88, 'Confidence', '\u2265 \u03b8 = 0.75?')

# YES → right
arrow(ax, 3.0+0.60, Y_DIA, 4.55, Y_GBMD)
ax.text(4.0, Y_DIA+0.16, 'Yes (~51%)', ha='center', va='bottom',
        fontsize=5.5, color='#0F6E56')

# NO → left
arrow(ax, 3.0-0.60, Y_DIA, 1.45, Y_RULE)
ax.text(2.0, Y_DIA+0.16, 'No (~49%)', ha='center', va='bottom',
        fontsize=5.5, color='#854F0B')

# ─── Row 4 left: rule engine ─────────────────────────────────────────────
box(ax, 0.82, Y_RULE, 1.40, BH2, 'Rule engine',
    C_AMBER, bold=True, sub='Signal-item threshold \u03c4 = 1.2')

# ─── Row 4 right: GBM direct ─────────────────────────────────────────────
box(ax, 5.18, Y_GBMD, 1.40, BH2, 'GBM direct',
    C_TEAL, bold=True, sub='High-confidence output')

# ─── Arrows from rule engine and GBM direct → merge ─────────────────────
# Rule engine down → min()
arrow(ax, 0.82, Y_RULE-BH2/2, 0.82, Y_MIN+BH/2, color='#854F0B')

# GBM direct → curves down then left to recommended set right edge
larrow(ax, [(5.18, Y_GBMD-BH2/2), (5.18, Y_REC), (4.52, Y_REC)],
       color='#0F6E56')

# ─── Row 5 left: min() box ───────────────────────────────────────────────
box(ax, 0.82, Y_MIN, 1.40, BH, 'min(GBM, rule)',
    C_CORAL, sub='Explanation display')

# min() → recommended set left edge
larrow(ax, [(1.52, Y_MIN), (2.48, Y_MIN)], color='#854F0B')

# ─── Row 5: Recommended dimension set ────────────────────────────────────
box(ax, 3.50, Y_REC, 2.00, BH, 'Recommended dimension set',
    C_TEAL, bold=True, sub='avg 1.15 dims \u00b7 Recall@K = 0.989')

arrow(ax, 3.50, Y_REC-BH/2, 3.50, Y_CLI+BH/2)

# ─── Row 6: Clinician review ─────────────────────────────────────────────
box(ax, 3.0, Y_CLI, 4.6, BH, 'Clinician review',
    C_GRAY, sub='Override or confirm \u00b7 Suicidality safeguard')
arrow(ax, 3.0, Y_CLI-BH/2, 3.0, Y_OUT+BH/2)

# ─── Row 7: Final output ─────────────────────────────────────────────────
box(ax, 3.0, Y_OUT, 3.4, BH, 'Questionnaire administered', C_GREEN, bold=True)

# ─── Legend ──────────────────────────────────────────────────────────────
lx, ly = 0.18, 2.22
ax.plot([lx, lx+0.40], [ly, ly], color='#854F0B', lw=1.0)
ax.text(lx+0.48, ly, 'Low-confidence path (rule engine + explanation)',
        va='center', fontsize=5.2, color='#2C2C2A')
ax.plot([lx, lx+0.40], [ly-0.25, ly-0.25], color='#0F6E56', lw=1.0)
ax.text(lx+0.48, ly-0.25, 'High-confidence path (GBM direct output)',
        va='center', fontsize=5.2, color='#2C2C2A')

# ─── Title ───────────────────────────────────────────────────────────────
ax.text(3.0, 8.42, 'Figure 1. RAAS Framework Overview',
        ha='center', va='center', fontsize=7.5, fontweight='bold',
        color='#2C2C2A')

plt.tight_layout(pad=0.2)
plt.savefig('/mnt/user-data/outputs/Figure1_RAAS_300dpi.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print('Done.')

import os
size = os.path.getsize('/mnt/user-data/outputs/Figure1_RAAS_300dpi.png')
print(f'File size: {size/1024:.0f} KB')

from PIL import Image
img = Image.open('/mnt/user-data/outputs/Figure1_RAAS_300dpi.png')
print(f'Dimensions: {img.size[0]} x {img.size[1]} px')
print(f'At 6-inch width: {img.size[0]/6:.0f} DPI effective')
