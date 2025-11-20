"""
PUBLICATION-QUALITY TEMPORAL ANALYSIS VISUALIZATIONS
Creates stunning, journal-ready plots with advanced aesthetics
Author: Advanced Data Analytics Team
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, ConnectionPatch
import matplotlib.gridspec as gridspec
from matplotlib import patheffects
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import make_interp_spline
import warnings
warnings.filterwarnings('ignore')

print("🎨 Initializing Publication-Quality Visualization Engine...")

# Set publication-quality defaults
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 16,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'lines.linewidth': 2.5,
})

# Custom color palettes
PALETTE_OCEAN = ['#0A1F44', '#1F487E', '#3D70B2', '#6497D4', '#9DC3E8']
PALETTE_FIRE = ['#8B0000', '#DC143C', '#FF6347', '#FFA07A', '#FFE4B5']
PALETTE_FOREST = ['#1A3A1A', '#2D5F2D', '#4A7C4A', '#6B9B6B', '#A8D5A8']
PALETTE_SUNSET = ['#FF6B35', '#F7931E', '#FDC830', '#F37335', '#C73E1D']

# Load data
print("📊 Loading data...")
df = pd.read_csv('cross_college_PES_cleaned.csv')
df['Date_OA'] = pd.to_datetime(df['Date_OA'], format='%d-%m-%Y', errors='coerce')
df_valid = df.dropna(subset=['Date_OA']).copy()
df_valid = df_valid.sort_values('Date_OA')
print(f"✓ Loaded {len(df_valid)} valid records\n")

# ============================================================================
# PLOT 1: THE MASTERPIECE - Ridgeline Plot with Timeline
# ============================================================================
print("🎨 Creating PLOT 1: The Temporal Masterpiece...")

fig = plt.figure(figsize=(24, 14))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 0.3], hspace=0.05)

# Prepare data - group by week
df_valid['Week'] = df_valid['Date_OA'].dt.to_period('W').dt.start_time
weekly_data = []
week_labels = []

for week, group in df_valid.groupby('Week'):
    if len(group) > 0:
        weekly_data.append(group)
        week_labels.append(week)

# Plot ridgeline for each tier
tiers = ['Tier 1', 'Tier 2', 'Tier 3']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
alphas = [0.8, 0.7, 0.6]

for idx, (tier, color, alpha) in enumerate(zip(tiers, colors, alphas)):
    ax = fig.add_subplot(gs[idx])
    
    # Get CTC distribution for this tier over time
    tier_data = df_valid[df_valid['Tier'] == tier].copy()
    
    if len(tier_data) > 0:
        # Create smooth density curves for each month
        for month, group in tier_data.groupby(tier_data['Date_OA'].dt.to_period('M')):
            if len(group) > 3:
                month_start = group['Date_OA'].min()
                ctc_values = group['CTC_LPA'].dropna()
                
                if len(ctc_values) > 0:
                    # Create histogram
                    counts, bins = np.histogram(ctc_values, bins=20)
                    
                    # Smooth the curve
                    if len(counts) > 1:
                        x_smooth = np.linspace(bins[0], bins[-1], 300)
                        spl = make_interp_spline(bins[:-1], counts, k=2)
                        y_smooth = spl(x_smooth)
                        y_smooth = np.maximum(y_smooth, 0)  # No negative values
                        
                        # Fill under curve
                        ax.fill_between(x_smooth, idx * 5, idx * 5 + y_smooth * 0.3,
                                       color=color, alpha=alpha, edgecolor='white', linewidth=2)
    
    # Styling
    ax.set_xlim(0, 60)
    ax.set_ylim(idx * 4.5, (idx + 1) * 5.5)
    ax.set_ylabel(tier, fontsize=14, fontweight='bold', rotation=0, ha='right', va='center')
    
    if idx < 2:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('CTC (Lakhs Per Annum)', fontsize=14, fontweight='bold')
    
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    
    # Add tier stats
    tier_stats = df_valid[df_valid['Tier'] == tier]
    avg_ctc = tier_stats['CTC_LPA'].mean()
    count = len(tier_stats)
    
    ax.text(45, idx * 5 + 2.5, f'n={count}\nμ={avg_ctc:.1f}L',
           fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                    edgecolor=color, linewidth=2, alpha=0.9))

# Bottom timeline
ax_timeline = fig.add_subplot(gs[3])
dates = df_valid['Date_OA']
tier_colors_map = {'Tier 1': colors[0], 'Tier 2': colors[1], 'Tier 3': colors[2]}

for idx, row in df_valid.iterrows():
    color = tier_colors_map.get(row['Tier'], 'gray')
    ax_timeline.scatter(row['Date_OA'], 0, s=100, c=color, alpha=0.6,
                       edgecolors='white', linewidth=1, zorder=3)

ax_timeline.set_ylim(-0.5, 0.5)
ax_timeline.set_yticks([])
ax_timeline.set_xlabel('Timeline (Jul 2024 - Jan 2025)', fontsize=14, fontweight='bold')
ax_timeline.spines['left'].set_visible(False)
ax_timeline.spines['right'].set_visible(False)
ax_timeline.spines['top'].set_visible(False)

plt.suptitle('🎭 THE TEMPORAL MASTERPIECE: Tier-wise CTC Distribution Ridgeline\nJourney from Premium to Accessibility',
            fontsize=20, fontweight='bold', y=0.98)
plt.savefig('VIZ1_Temporal_Masterpiece.png', dpi=400, bbox_inches='tight', facecolor='white')
print("✓ Saved: VIZ1_Temporal_Masterpiece.png\n")
plt.close()

# ============================================================================
# PLOT 2: THE GALAXY PLOT - 3D Scatter with Orbital Paths  
# ============================================================================
print("🌌 Creating PLOT 2: The Galaxy Plot...")

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('black')
ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a0a')

# Prepare 3D data
df_3d = df_valid[df_valid['CTC_LPA'].notna() & df_valid['CGPA_Cutoff'].notna()].copy()
df_3d['Days'] = (df_3d['Date_OA'] - df_3d['Date_OA'].min()).dt.days

# Create the galaxy
x = df_3d['Days']
y = df_3d['CGPA_Cutoff']
z = df_3d['CTC_LPA']

# Color by tier
tier_color_3d = df_3d['Tier'].map({'Tier 1': '#FFD700', 'Tier 2': '#87CEEB', 'Tier 3': '#FF69B4'})

scatter = ax.scatter(x, y, z, c=tier_color_3d, s=200, alpha=0.8,
                    edgecolors='white', linewidth=1.5, depthshade=True)

# Add orbital paths (connect companies by tier)
for tier in ['Tier 1', 'Tier 2', 'Tier 3']:
    tier_data = df_3d[df_3d['Tier'] == tier].sort_values('Days')
    if len(tier_data) > 1:
        ax.plot(tier_data['Days'], tier_data['CGPA_Cutoff'], tier_data['CTC_LPA'],
               color=tier_color_3d.iloc[0] if tier == 'Tier 1' else 
                     ('#87CEEB' if tier == 'Tier 2' else '#FF69B4'),
               alpha=0.3, linewidth=2, linestyle='--')

# Labels and styling
ax.set_xlabel('\n\nDays Since Start', fontsize=14, fontweight='bold', color='white')
ax.set_ylabel('\n\nCGPA Cutoff', fontsize=14, fontweight='bold', color='white')
ax.set_zlabel('\n\nCTC (LPA)', fontsize=14, fontweight='bold', color='white')

ax.tick_params(colors='white')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('gray')
ax.yaxis.pane.set_edgecolor('gray')
ax.zaxis.pane.set_edgecolor('gray')
ax.grid(True, alpha=0.2, color='white')

# Title
fig.text(0.5, 0.95, '🌌 THE GALAXY PLOT: Multi-Dimensional Placement Universe',
        ha='center', fontsize=22, fontweight='bold', color='white')
fig.text(0.5, 0.92, 'X: Timeline | Y: CGPA Requirements | Z: Salary | Color: Company Tier',
        ha='center', fontsize=13, color='lightgray', style='italic')

# Add legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='#0a0a0a', label='Tier 1',
              markerfacecolor='#FFD700', markersize=12, markeredgecolor='white', markeredgewidth=1.5),
    plt.Line2D([0], [0], marker='o', color='#0a0a0a', label='Tier 2',
              markerfacecolor='#87CEEB', markersize=12, markeredgecolor='white', markeredgewidth=1.5),
    plt.Line2D([0], [0], marker='o', color='#0a0a0a', label='Tier 3',
              markerfacecolor='#FF69B4', markersize=12, markeredgecolor='white', markeredgewidth=1.5),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=12, 
         facecolor='#1a1a1a', edgecolor='white', labelcolor='white')

plt.savefig('VIZ2_Galaxy_Plot.png', dpi=400, bbox_inches='tight', facecolor='black')
print("✓ Saved: VIZ2_Galaxy_Plot.png\n")
plt.close()

# ============================================================================
# PLOT 3: THE HEATMAP CHRONICLES - Advanced Temporal Heatmap
# ============================================================================
print("🔥 Creating PLOT 3: The Heatmap Chronicles...")

fig, axes = plt.subplots(3, 1, figsize=(24, 16))
fig.patch.set_facecolor('white')

# Create pivot tables for different metrics
df_valid['Week_Num'] = df_valid['Date_OA'].dt.isocalendar().week
df_valid['DayOfWeek'] = df_valid['Date_OA'].dt.dayofweek
df_valid['Month'] = df_valid['Date_OA'].dt.month

# Heatmap 1: Companies per day-week combination
day_week = df_valid.groupby(['Week_Num', 'DayOfWeek']).size().reset_index(name='count')
pivot1 = day_week.pivot(index='DayOfWeek', columns='Week_Num', values='count').fillna(0)
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
pivot1.index = [day_names[int(i)] for i in pivot1.index]

sns.heatmap(pivot1, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[0],
           cbar_kws={'label': 'Companies'}, linewidths=0.5, linecolor='white')
axes[0].set_title('📅 HIRING INTENSITY: Day × Week Heatmap', 
                 fontsize=16, fontweight='bold', pad=15)
axes[0].set_xlabel('Week Number', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Day of Week', fontsize=12, fontweight='bold')

# Heatmap 2: Average CTC by tier and month
tier_month_ctc = df_valid.groupby(['Tier', 'Month'])['CTC_LPA'].mean().reset_index()
pivot2 = tier_month_ctc.pivot(index='Tier', columns='Month', values='CTC_LPA')
month_names = {7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec', 1: 'Jan'}
pivot2.columns = [month_names.get(m, str(m)) for m in pivot2.columns]

sns.heatmap(pivot2, annot=True, fmt='.1f', cmap='RdYlGn', ax=axes[1],
           cbar_kws={'label': 'Avg CTC (LPA)'}, linewidths=2, linecolor='white',
           vmin=0, vmax=30)
axes[1].set_title('💰 SALARY EVOLUTION: Average CTC by Tier × Month',
                 fontsize=16, fontweight='bold', pad=15)
axes[1].set_xlabel('Month', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Tier', fontsize=12, fontweight='bold')

# Heatmap 3: CGPA requirements by month and tier
tier_month_cgpa = df_valid.groupby(['Tier', 'Month'])['CGPA_Cutoff'].mean().reset_index()
pivot3 = tier_month_cgpa.pivot(index='Tier', columns='Month', values='CGPA_Cutoff')
pivot3.columns = [month_names.get(m, str(m)) for m in pivot3.columns]

sns.heatmap(pivot3, annot=True, fmt='.2f', cmap='coolwarm_r', ax=axes[2],
           cbar_kws={'label': 'Avg CGPA Cutoff'}, linewidths=2, linecolor='white',
           vmin=6, vmax=9)
axes[2].set_title('🎓 ACCESSIBILITY MATRIX: CGPA Requirements by Tier × Month',
                 fontsize=16, fontweight='bold', pad=15)
axes[2].set_xlabel('Month', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Tier', fontsize=12, fontweight='bold')

plt.suptitle('🔥 THE HEATMAP CHRONICLES: Multi-Dimensional Temporal Patterns',
            fontsize=20, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('VIZ3_Heatmap_Chronicles.png', dpi=400, bbox_inches='tight', facecolor='white')
print("✓ Saved: VIZ3_Heatmap_Chronicles.png\n")
plt.close()

# ============================================================================
# PLOT 4: THE INFOGRAPHIC - Dashboard Style
# ============================================================================
print("📊 Creating PLOT 4: The Ultimate Infographic...")

fig = plt.figure(figsize=(24, 18))
fig.patch.set_facecolor('#F0F0F0')
gs = gridspec.GridSpec(4, 3, hspace=0.4, wspace=0.3)

# Top banner
ax_banner = fig.add_subplot(gs[0, :])
ax_banner.axis('off')
ax_banner.text(0.5, 0.7, '📊 PES CAMPUS PLACEMENTS 2025', ha='center', va='center',
              fontsize=32, fontweight='bold', family='sans-serif',
              bbox=dict(boxstyle='round,pad=1', facecolor='#667eea', 
                       edgecolor='#764ba2', linewidth=5, alpha=0.9))
ax_banner.text(0.5, 0.3, 'Comprehensive Temporal Analysis | July 2024 - January 2025',
              ha='center', va='center', fontsize=14, style='italic', color='#333')

# Big numbers
stats = [
    (gs[1, 0], len(df_valid), 'COMPANIES', '#FF6B6B'),
    (gs[1, 1], f"{df_valid['CTC_LPA'].max():.0f}L", 'TOP PACKAGE', '#4ECDC4'),
    (gs[1, 2], f"{(df_valid['Date_OA'].max() - df_valid['Date_OA'].min()).days}", 'DAYS', '#45B7D1'),
]

for pos, value, label, color in stats:
    ax = fig.add_subplot(pos)
    ax.axis('off')
    
    # Circle background
    circle = Circle((0.5, 0.5), 0.35, transform=ax.transAxes, 
                   color=color, alpha=0.3, zorder=1)
    ax.add_patch(circle)
    
    ax.text(0.5, 0.6, str(value), ha='center', va='center',
           fontsize=42, fontweight='bold', color=color,
           transform=ax.transAxes, zorder=2)
    ax.text(0.5, 0.35, label, ha='center', va='center',
           fontsize=14, fontweight='bold', color='#333',
           transform=ax.transAxes, zorder=2)

# Monthly progression
ax2 = fig.add_subplot(gs[2, :])
monthly = df_valid.groupby(df_valid['Date_OA'].dt.to_period('M')).size()
months = [str(m) for m in monthly.index]
values = monthly.values

bars = ax2.barh(range(len(months)), values, height=0.6,
               color=plt.cm.viridis(np.linspace(0.3, 0.9, len(months))),
               edgecolor='white', linewidth=2)

for i, (bar, val) in enumerate(zip(bars, values)):
    ax2.text(val + 1, i, f'{val}', va='center', fontweight='bold', fontsize=11)

ax2.set_yticks(range(len(months)))
ax2.set_yticklabels(months, fontsize=12, fontweight='bold')
ax2.set_xlabel('Number of Companies', fontsize=13, fontweight='bold')
ax2.set_title('📅 MONTHLY HIRING PROGRESSION', fontsize=16, fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3)

# Tier composition pie
ax3 = fig.add_subplot(gs[3, 0])
tier_counts = df_valid['Tier'].value_counts()
colors_pie = ['#00b894', '#fdcb6e', '#e17055']
explode = (0.05, 0.05, 0.05)
wedges, texts, autotexts = ax3.pie(tier_counts.values, labels=tier_counts.index,
                                    autopct='%1.1f%%', colors=colors_pie,
                                    explode=explode, shadow=True, startangle=90,
                                    textprops={'fontsize': 12, 'fontweight': 'bold'})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(13)
    autotext.set_fontweight('bold')
ax3.set_title('🏆 TIER DISTRIBUTION', fontsize=14, fontweight='bold', pad=10)

# Top companies
ax4 = fig.add_subplot(gs[3, 1:])
ax4.axis('off')
top10 = df_valid.nlargest(10, 'CTC_LPA')[['Company', 'CTC_LPA', 'Tier']]

table_data = [[f"#{i+1}", row['Company'][:20], f"₹{row['CTC_LPA']}L", row['Tier']]
             for i, (idx, row) in enumerate(top10.iterrows())]

table = ax4.table(cellText=table_data,
                 colLabels=['Rank', 'Company', 'CTC', 'Tier'],
                 cellLoc='left', loc='center',
                 colWidths=[0.1, 0.5, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

for i in range(4):
    cell = table[(0, i)]
    cell.set_facecolor('#667eea')
    cell.set_text_props(weight='bold', color='white', size=12)

for i in range(1, 11):
    for j in range(4):
        cell = table[(i, j)]
        cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
        cell.set_text_props(size=10)

ax4.text(0.5, 0.95, '🥇 TOP 10 PACKAGES', transform=ax4.transAxes,
        ha='center', fontsize=15, fontweight='bold')

plt.savefig('VIZ4_Ultimate_Infographic.png', dpi=400, bbox_inches='tight', facecolor='#F0F0F0')
print("✓ Saved: VIZ4_Ultimate_Infographic.png\n")
plt.close()

# ============================================================================
# PLOT 5: THE STREAM GRAPH - Beautiful Flow Visualization
# ============================================================================
print("🌊 Creating PLOT 5: The Stream Graph...")

fig, ax = plt.subplots(figsize=(24, 12))
fig.patch.set_facecolor('white')

# Prepare data for stream graph
weekly_tier = df_valid.groupby([df_valid['Date_OA'].dt.to_period('W'), 'Tier']).size().unstack(fill_value=0)
weekly_tier.index = weekly_tier.index.to_timestamp()

# Create smooth curves
x = np.arange(len(weekly_tier))
x_smooth = np.linspace(0, len(weekly_tier)-1, 500)

tier_data = {}
for tier in ['Tier 1', 'Tier 2', 'Tier 3']:
    if tier in weekly_tier.columns:
        y = weekly_tier[tier].values
        # Smooth
        spl = make_interp_spline(x, y, k=3)
        y_smooth = spl(x_smooth)
        y_smooth = np.maximum(y_smooth, 0)
        tier_data[tier] = y_smooth

# Create streamgraph (centered)
baseline = np.zeros(len(x_smooth))
colors_stream = {'Tier 1': '#FF6B6B', 'Tier 2': '#4ECDC4', 'Tier 3': '#95E1D3'}

for tier in ['Tier 3', 'Tier 2', 'Tier 1']:
    if tier in tier_data:
        y = tier_data[tier]
        ax.fill_between(x_smooth, baseline - y/2, baseline + y/2,
                       color=colors_stream[tier], alpha=0.7, label=tier,
                       edgecolor='white', linewidth=2)

# Styling
dates = weekly_tier.index
date_positions = np.linspace(0, len(x_smooth)-1, len(dates))
ax.set_xticks(date_positions[::4])
ax.set_xticklabels([d.strftime('%b %d') for d in dates[::4]], rotation=45, ha='right')

ax.set_ylabel('Weekly Company Flow', fontsize=14, fontweight='bold')
ax.set_xlabel('Timeline (Weeks)', fontsize=14, fontweight='bold')
ax.set_title('🌊 THE STREAM GRAPH: Flowing Through Placement Season\nTier Distribution Stream Over Time',
            fontsize=20, fontweight='bold', pad=20)

ax.legend(loc='upper left', fontsize=13, framealpha=0.95, edgecolor='black', fancybox=True)
ax.grid(True, alpha=0.2, axis='x')
ax.set_xlim(0, len(x_smooth)-1)

# Add phase annotations
phase_colors = ['#FFE5E5', '#E5F5FF', '#FFFBE5']
phase_labels = ['Premium Phase', 'Peak Season', 'Accessibility Phase']
phase_bounds = [0, len(x_smooth)//3, 2*len(x_smooth)//3, len(x_smooth)]

for i in range(3):
    ax.axvspan(phase_bounds[i], phase_bounds[i+1], alpha=0.1, color=phase_colors[i], zorder=0)
    mid = (phase_bounds[i] + phase_bounds[i+1]) / 2
    ax.text(mid, ax.get_ylim()[1] * 0.9, phase_labels[i],
           ha='center', fontsize=12, fontweight='bold', style='italic',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=phase_colors[i], 
                    alpha=0.7, edgecolor='gray'))

plt.tight_layout()
plt.savefig('VIZ5_Stream_Graph.png', dpi=400, bbox_inches='tight', facecolor='white')
print("✓ Saved: VIZ5_Stream_Graph.png\n")
plt.close()

# ============================================================================
# PLOT 6: THE RADIAL TIMELINE - Circular Visualization
# ============================================================================
print("⭕ Creating PLOT 6: The Radial Timeline...")

fig = plt.figure(figsize=(20, 20))
fig.patch.set_facecolor('white')
ax = fig.add_subplot(111, projection='polar')

# Map days to angles
df_plot = df_valid.copy()
df_plot['Days'] = (df_plot['Date_OA'] - df_plot['Date_OA'].min()).dt.days
total_days = df_plot['Days'].max()
df_plot['Theta'] = df_plot['Days'] / total_days * 2 * np.pi

# Color by tier
tier_colors_radial = {'Tier 1': '#FFD700', 'Tier 2': '#87CEEB', 'Tier 3': '#FF69B4'}

for tier in ['Tier 1', 'Tier 2', 'Tier 3']:
    tier_df = df_plot[df_plot['Tier'] == tier]
    
    theta = tier_df['Theta']
    r = tier_df['CTC_LPA'].fillna(tier_df['CTC_LPA'].mean())
    
    ax.scatter(theta, r, s=200, c=tier_colors_radial[tier], 
              alpha=0.7, edgecolors='white', linewidth=2,
              label=tier, zorder=3)

# Add month markers
months = df_plot.groupby(df_plot['Date_OA'].dt.to_period('M')).agg({
    'Days': 'min',
    'Company': 'count'
}).reset_index()
months['Theta'] = months['Days'] / total_days * 2 * np.pi

for _, row in months.iterrows():
    ax.axvline(row['Theta'], color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    month_name = str(row['Date_OA'])[:7]
    ax.text(row['Theta'], ax.get_ylim()[1] * 1.15, month_name,
           ha='center', fontsize=11, fontweight='bold', rotation=0)

# Styling
ax.set_ylim(0, 60)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_xlabel('')
ax.set_ylabel('CTC (LPA)', fontsize=14, fontweight='bold', labelpad=30)
ax.grid(True, alpha=0.3)

# Remove degree labels, add custom
ax.set_xticklabels([])

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=13, 
         framealpha=0.95, edgecolor='black', fancybox=True)

plt.title('⭕ THE RADIAL TIMELINE: 360° View of Placement Season\nCTC Distribution Around the Calendar',
         fontsize=22, fontweight='bold', pad=40)

plt.savefig('VIZ6_Radial_Timeline.png', dpi=400, bbox_inches='tight', facecolor='white')
print("✓ Saved: VIZ6_Radial_Timeline.png\n")
plt.close()

# ============================================================================
print("\n" + "="*80)
print("🎊 ALL PUBLICATION-QUALITY VISUALIZATIONS GENERATED! 🎊")
print("="*80)
print("\n📁 Generated Files:")
print("  1. VIZ1_Temporal_Masterpiece.png - Ridgeline plot with timeline")
print("  2. VIZ2_Galaxy_Plot.png - 3D scatter plot (space theme)")
print("  3. VIZ3_Heatmap_Chronicles.png - Advanced heatmaps")
print("  4. VIZ4_Ultimate_Infographic.png - Dashboard infographic")
print("  5. VIZ5_Stream_Graph.png - Beautiful flow visualization")
print("  6. VIZ6_Radial_Timeline.png - Circular timeline view")
print("\n✨ All plots are 400 DPI publication-ready PNG files!")
print("🎨 Ready for journals, presentations, or publications!")
print("="*80)
