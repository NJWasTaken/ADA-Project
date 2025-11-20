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
df = pd.read_csv('processed_data/2025_PES_withTemporalData.csv')
df['Date_OA'] = pd.to_datetime(df['Date_OA'], format='%d-%m-%Y', errors='coerce')
df_valid = df.dropna(subset=['Date_OA']).copy()
df_valid = df_valid.sort_values('Date_OA')
print(f"✓ Loaded {len(df_valid)} valid records\n")

# ============================================================================
# PLOT 9: Advanced Temporal Heatmap
# ============================================================================
print("🔥 Creating PLOT 9: Advanced Temporal Heatmap")

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

plt.suptitle('Multi-Dimensional Temporal Patterns',
            fontsize=20, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('PLOT9_Temporal_Patterns.png', dpi=400, bbox_inches='tight', facecolor='white')
print("✓ Saved: PLOT9_Temporal_Patterns.png\n")
plt.close()
