"""
CRAZY GALAXY-STYLE VISUALIZATIONS 🌌
More insane 3D and dark-themed plots for temporal analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from datetime import datetime
import seaborn as sns
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

print("🌌 Launching Galaxy Visualization Engine...")

# Load data
df = pd.read_csv('cross_college_PES_cleaned.csv')
df['Date_OA'] = pd.to_datetime(df['Date_OA'], format='%d-%m-%Y', errors='coerce')
df_valid = df.dropna(subset=['Date_OA']).copy()
df_valid['Days'] = (df_valid['Date_OA'] - df_valid['Date_OA'].min()).dt.days
print(f"✓ {len(df_valid)} companies loaded\n")

# ============================================================================
# PLOT 1: THE NEBULA - 3D Density Cloud
# ============================================================================
print("🌟 Creating PLOT 1: The Nebula...")

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('black')
ax = fig.add_subplot(111, projection='3d', facecolor='#000000')

df_3d = df_valid[df_valid['CTC_LPA'].notna() & df_valid['CGPA_Cutoff'].notna()].copy()

# Create 3D scatter with varying sizes
x = df_3d['Days']
y = df_3d['CGPA_Cutoff']
z = df_3d['CTC_LPA']

# Size based on total offers
sizes = df_3d['Total_Offers'].fillna(1) * 50

# Color gradient based on CTC
colors = cm.plasma(z / z.max())

# Main scatter
scatter = ax.scatter(x, y, z, c=z, s=sizes, alpha=0.8, 
                    cmap='plasma', edgecolors='white', linewidth=1,
                    depthshade=True)

# Add connecting lines to create nebula effect
for tier in df_3d['Tier'].unique():
    tier_data = df_3d[df_3d['Tier'] == tier].sort_values('Days')
    if len(tier_data) > 1:
        ax.plot(tier_data['Days'], tier_data['CGPA_Cutoff'], tier_data['CTC_LPA'],
               color='cyan', alpha=0.2, linewidth=1, linestyle='-')

# Add glowing effect with duplicate transparent layers
ax.scatter(x, y, z, c=z, s=sizes*2, alpha=0.1, cmap='plasma')
ax.scatter(x, y, z, c=z, s=sizes*3, alpha=0.05, cmap='plasma')

# Styling
ax.set_xlabel('\n\n⏰ Days Into Season', fontsize=14, fontweight='bold', color='cyan')
ax.set_ylabel('\n\n🎓 CGPA Barrier', fontsize=14, fontweight='bold', color='cyan')
ax.set_zlabel('\n\n💰 CTC (Lakhs)', fontsize=14, fontweight='bold', color='cyan')

ax.tick_params(colors='white', labelsize=10)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#1a1a1a')
ax.yaxis.pane.set_edgecolor('#1a1a1a')
ax.zaxis.pane.set_edgecolor('#1a1a1a')
ax.grid(True, alpha=0.1, color='white')

# Colorbar
cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label('CTC (LPA)', color='white', fontsize=12, fontweight='bold')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

fig.text(0.5, 0.95, '🌟 THE NEBULA: 3D Density Cloud of Opportunities',
        ha='center', fontsize=22, fontweight='bold', color='white')
fig.text(0.5, 0.92, 'Size = Total Offers | Color = CTC | Connected paths show tier clusters',
        ha='center', fontsize=12, color='cyan', style='italic')

plt.savefig('GALAXY1_Nebula.png', dpi=400, bbox_inches='tight', facecolor='black')
print("✓ Saved: GALAXY1_Nebula.png\n")
plt.close()

# ============================================================================
# PLOT 2: THE SUPERNOVA - Exploding Timeline 
# ============================================================================
print("💥 Creating PLOT 2: The Supernova...")

fig = plt.figure(figsize=(20, 20))
fig.patch.set_facecolor('black')
ax = fig.add_subplot(111, projection='polar', facecolor='#000000')

# Convert timeline to polar coordinates
df_polar = df_valid[df_valid['CTC_LPA'].notna()].copy()
total_days = df_polar['Days'].max()
df_polar['Theta'] = (df_polar['Days'] / total_days) * 2 * np.pi
df_polar['R'] = df_polar['CTC_LPA']

# Tier colors
tier_colors = {'Tier 1': '#FFD700', 'Tier 2': '#00FFFF', 'Tier 3': '#FF1493'}

# Plot each tier with different effects
for tier, color in tier_colors.items():
    tier_df = df_polar[df_polar['Tier'] == tier]
    
    # Main points
    ax.scatter(tier_df['Theta'], tier_df['R'], s=300, c=color,
              alpha=0.8, edgecolors='white', linewidth=2, zorder=3)
    
    # Glow layers
    ax.scatter(tier_df['Theta'], tier_df['R'], s=600, c=color,
              alpha=0.3, edgecolors='none', zorder=2)
    ax.scatter(tier_df['Theta'], tier_df['R'], s=900, c=color,
              alpha=0.1, edgecolors='none', zorder=1)
    
    # Connect points with spiral arms
    if len(tier_df) > 1:
        tier_sorted = tier_df.sort_values('Theta')
        ax.plot(tier_sorted['Theta'], tier_sorted['R'], 
               color=color, alpha=0.4, linewidth=3, linestyle='--')

# Add radial grid lines for months
months = df_polar.groupby(df_polar['Date_OA'].dt.to_period('M'))['Days'].min()
for month_days in months:
    theta = (month_days / total_days) * 2 * np.pi
    ax.plot([theta, theta], [0, ax.get_ylim()[1]], 
           color='gray', alpha=0.3, linewidth=1.5, linestyle=':')

# Styling
ax.set_ylim(0, 60)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_facecolor('#000000')
ax.grid(True, color='white', alpha=0.2, linewidth=1)
ax.set_xticklabels([])
ax.tick_params(colors='white')

# Make radial labels white
for label in ax.get_yticklabels():
    label.set_color('white')
    label.set_fontweight('bold')

# Legend
legend_elements = [plt.scatter([], [], s=200, c=color, alpha=0.8, 
                              edgecolors='white', linewidth=2, label=tier)
                  for tier, color in tier_colors.items()]
ax.legend(handles=legend_elements, loc='upper right', 
         bbox_to_anchor=(1.2, 1.1), fontsize=13, 
         facecolor='#1a1a1a', edgecolor='white', 
         labelcolor='white', framealpha=0.9)

plt.title('💥 THE SUPERNOVA: Exploding Timeline View\nRadial Energy Distribution of Placements',
         fontsize=22, fontweight='bold', color='white', pad=40)

plt.savefig('GALAXY2_Supernova.png', dpi=400, bbox_inches='tight', facecolor='black')
print("✓ Saved: GALAXY2_Supernova.png\n")
plt.close()

# ============================================================================
# PLOT 3: THE BLACK HOLE - Gravity Well Visualization
# ============================================================================
print("🕳️ Creating PLOT 3: The Black Hole...")

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('black')
ax = fig.add_subplot(111, projection='3d', facecolor='#000000')

# Create a surface representing CGPA difficulty over time and tier
tier_map = {'Tier 1': 3, 'Tier 2': 2, 'Tier 3': 1}
df_surface = df_valid[df_valid['CGPA_Cutoff'].notna()].copy()
df_surface['TierNum'] = df_surface['Tier'].map(tier_map)
# Drop any rows where TierNum is NaN
df_surface = df_surface.dropna(subset=['TierNum', 'Days'])

# Create grid
x_grid = np.linspace(df_surface['Days'].min(), df_surface['Days'].max(), 50)
y_grid = np.linspace(1, 3, 50)
X, Y = np.meshgrid(x_grid, y_grid)

# Interpolate CGPA values (double-checked for no NaN)
points = df_surface[['Days', 'TierNum']].values
values = df_surface['CGPA_Cutoff'].values

# Extra safety check
mask = ~(np.isnan(points).any(axis=1) | np.isnan(values))
points = points[mask]
values = values[mask]

Z = griddata(points, values, (X, Y), method='linear')
Z = np.nan_to_num(Z, nan=7.0)

# Create the "gravity well" effect by inverting
Z_well = 10 - Z  # Flip it

# Plot surface with cool gradient
surf = ax.plot_surface(X, Y, Z_well, cmap='twilight', alpha=0.8,
                       linewidth=0, antialiased=True, edgecolors='none')

# Add companies as stars
for idx, row in df_valid[df_valid['CGPA_Cutoff'].notna()].iterrows():
    tier_num = tier_map.get(row['Tier'], 2)
    z_val = 10 - row['CGPA_Cutoff']
    
    color = '#FFD700' if row['Tier'] == 'Tier 1' else '#00FFFF' if row['Tier'] == 'Tier 2' else '#FF1493'
    ax.scatter(row['Days'], tier_num, z_val, s=100, c=color, 
              alpha=0.9, edgecolors='white', linewidth=1, zorder=10)

# Styling
ax.set_xlabel('\n\n⏰ Timeline (Days)', fontsize=14, fontweight='bold', color='cyan')
ax.set_ylabel('\n\nTier Level', fontsize=14, fontweight='bold', color='cyan')
ax.set_zlabel('\n\nAccessibility\n(CGPA)', fontsize=14, fontweight='bold', color='cyan')

ax.set_yticks([1, 2, 3])
ax.set_yticklabels(['Tier 3', 'Tier 2', 'Tier 1'], color='white')

ax.tick_params(colors='white')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#1a1a1a')
ax.yaxis.pane.set_edgecolor('#1a1a1a')
ax.zaxis.pane.set_edgecolor('#1a1a1a')
ax.grid(True, alpha=0.15, color='white')

# Colorbar
cbar = plt.colorbar(surf, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label('Accessibility', color='white', fontsize=12, fontweight='bold')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

fig.text(0.5, 0.95, '🕳️ THE BLACK HOLE: Accessibility Gravity Well',
        ha='center', fontsize=22, fontweight='bold', color='white')
fig.text(0.5, 0.92, 'Surface depth = CGPA difficulty | Companies shown as stars in the well',
        ha='center', fontsize=12, color='cyan', style='italic')

plt.savefig('GALAXY3_BlackHole.png', dpi=400, bbox_inches='tight', facecolor='black')
print("✓ Saved: GALAXY3_BlackHole.png\n")
plt.close()

# ============================================================================
# PLOT 4: THE CONSTELLATION - Network Connections
# ============================================================================
print("⭐ Creating PLOT 4: The Constellation...")

fig = plt.figure(figsize=(24, 16))
fig.patch.set_facecolor('black')
ax = fig.add_subplot(111, projection='3d', facecolor='#000000')

# Use all data
df_const = df_valid[df_valid['CTC_LPA'].notna()].copy()

x = df_const['Days']
y = df_const['CGPA_Cutoff'].fillna(df_const['CGPA_Cutoff'].mean())
z = df_const['CTC_LPA']

# Plot stars
tier_colors = {'Tier 1': '#FFD700', 'Tier 2': '#00CED1', 'Tier 3': '#FF69B4'}
for tier, color in tier_colors.items():
    tier_df = df_const[df_const['Tier'] == tier]
    ax.scatter(tier_df['Days'], 
              tier_df['CGPA_Cutoff'].fillna(tier_df['CGPA_Cutoff'].mean()),
              tier_df['CTC_LPA'],
              s=250, c=color, alpha=0.9, edgecolors='white', linewidth=2, zorder=5)
    
    # Glow effect
    ax.scatter(tier_df['Days'], 
              tier_df['CGPA_Cutoff'].fillna(tier_df['CGPA_Cutoff'].mean()),
              tier_df['CTC_LPA'],
              s=500, c=color, alpha=0.2, edgecolors='none', zorder=4)

# Connect nearby companies (within same week) to create constellations
df_const_sorted = df_const.sort_values('Days')
for i in range(len(df_const_sorted) - 1):
    row1 = df_const_sorted.iloc[i]
    row2 = df_const_sorted.iloc[i + 1]
    
    # Connect if within 7 days
    if abs(row1['Days'] - row2['Days']) <= 7:
        x_line = [row1['Days'], row2['Days']]
        y_line = [row1['CGPA_Cutoff'] if pd.notna(row1['CGPA_Cutoff']) else y.mean(), 
                 row2['CGPA_Cutoff'] if pd.notna(row2['CGPA_Cutoff']) else y.mean()]
        z_line = [row1['CTC_LPA'], row2['CTC_LPA']]
        
        ax.plot(x_line, y_line, z_line, color='cyan', alpha=0.15, linewidth=1)

# Add "shooting stars" - highlight premium companies
premium = df_const[df_const['CTC_LPA'] >= 30]
for idx, row in premium.iterrows():
    y_val = row['CGPA_Cutoff'] if pd.notna(row['CGPA_Cutoff']) else y.mean()
    # Draw a trail
    trail_days = [row['Days'] - 10, row['Days']]
    trail_y = [y_val, y_val]
    trail_z = [row['CTC_LPA'] - 5, row['CTC_LPA']]
    ax.plot(trail_days, trail_y, trail_z, color='yellow', alpha=0.6, linewidth=3)

# Styling  
ax.set_xlabel('\n\n⏰ Temporal Dimension', fontsize=14, fontweight='bold', color='white')
ax.set_ylabel('\n\n🎓 CGPA Dimension', fontsize=14, fontweight='bold', color='white')
ax.set_zlabel('\n\n💰 Salary Dimension', fontsize=14, fontweight='bold', color='white')

ax.tick_params(colors='white')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('none')
ax.yaxis.pane.set_edgecolor('none')
ax.zaxis.pane.set_edgecolor('none')
ax.grid(True, alpha=0.1, color='white')

# Set viewing angle
ax.view_init(elev=20, azim=45)

fig.text(0.5, 0.95, '⭐ THE CONSTELLATION: Connected Opportunities Network',
        ha='center', fontsize=22, fontweight='bold', color='white')
fig.text(0.5, 0.92, 'Stars = Companies | Lines = Temporal clusters | Trails = Premium packages',
        ha='center', fontsize=12, color='cyan', style='italic')

plt.savefig('GALAXY4_Constellation.png', dpi=400, bbox_inches='tight', facecolor='black')
print("✓ Saved: GALAXY4_Constellation.png\n")
plt.close()

# ============================================================================
# PLOT 5: THE AURORA - Northern Lights Effect
# ============================================================================
print("🌈 Creating PLOT 5: The Aurora...")

from scipy.interpolate import make_interp_spline

fig = plt.figure(figsize=(24, 14))
fig.patch.set_facecolor('black')
ax = fig.add_subplot(111, facecolor='#000814')

# Create flowing ribbons for each tier
for tier_idx, (tier, color) in enumerate([('Tier 1', '#00FF88'), 
                                           ('Tier 2', '#00DDFF'), 
                                           ('Tier 3', '#FF00FF')]):
    tier_data = df_valid[df_valid['Tier'] == tier].sort_values('Days')
    
    if len(tier_data) > 2:
        # Aggregate by Days to handle duplicates - take mean CTC
        tier_agg = tier_data.groupby('Days')['CTC_LPA'].mean().reset_index()
        tier_agg['CTC_LPA'] = tier_agg['CTC_LPA'].fillna(tier_data['CTC_LPA'].mean())
        
        x = tier_agg['Days'].values
        y = tier_agg['CTC_LPA'].values
        
        # Create smooth curve
        if len(x) > 3:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spl = make_interp_spline(x, y, k=min(3, len(x)-1))
            y_smooth = spl(x_smooth)
            
            # Plot multiple alpha layers for aurora effect
            for alpha_mult in [1.0, 0.6, 0.3, 0.15]:
                ax.plot(x_smooth, y_smooth + tier_idx * 2, 
                       color=color, alpha=0.8 * alpha_mult, linewidth=8/alpha_mult)
            
            # Add glow underneath
            ax.fill_between(x_smooth, tier_idx * 2, y_smooth + tier_idx * 2,
                          alpha=0.2, color=color)

# Add stars (companies) on top
for tier, color in [('Tier 1', '#00FF88'), ('Tier 2', '#00DDFF'), ('Tier 3', '#FF00FF')]:
    tier_data = df_valid[df_valid['Tier'] == tier]
    ax.scatter(tier_data['Days'], tier_data['CTC_LPA'].fillna(tier_data['CTC_LPA'].mean()),
              s=100, c=color, alpha=0.9, edgecolors='white', linewidth=1.5, zorder=10)

# Styling
ax.set_xlabel('Timeline (Days into Season)', fontsize=16, fontweight='bold', color='white')
ax.set_ylabel('CTC (Lakhs Per Annum)', fontsize=16, fontweight='bold', color='white')
ax.set_title('🌈 THE AURORA: Northern Lights of Opportunities\nFlowing Tier-wise Salary Streams',
            fontsize=22, fontweight='bold', color='white', pad=20)

ax.tick_params(colors='white', labelsize=12)
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.15, color='white', linestyle='--')

# Legend
legend_elements = [plt.Line2D([0], [0], color=c, linewidth=6, label=t, alpha=0.8)
                  for t, c in [('Tier 1', '#00FF88'), ('Tier 2', '#00DDFF'), ('Tier 3', '#FF00FF')]]
ax.legend(handles=legend_elements, loc='upper left', fontsize=14,
         facecolor='#001428', edgecolor='white', labelcolor='white', framealpha=0.9)

plt.tight_layout()
plt.savefig('GALAXY5_Aurora.png', dpi=400, bbox_inches='tight', facecolor='black')
print("✓ Saved: GALAXY5_Aurora.png\n")
plt.close()

# ============================================================================
print("\n" + "="*80)
print("🌌 GALAXY VISUALIZATION COMPLETE! 🌌")
print("="*80)
print("\n🎨 Generated Files:")
print("  1. GALAXY1_Nebula.png - 3D density cloud with glowing particles")
print("  2. GALAXY2_Supernova.png - Exploding radial timeline")
print("  3. GALAXY3_BlackHole.png - Gravity well surface plot")
print("  4. GALAXY4_Constellation.png - Network of connected opportunities")
print("  5. GALAXY5_Aurora.png - Northern lights flowing streams")
print("\n✨ All plots are 400 DPI with BLACK/SPACE backgrounds!")
print("🚀 Ready to amaze everyone!")
print("="*80)
