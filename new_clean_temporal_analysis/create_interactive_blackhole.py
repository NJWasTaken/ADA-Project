"""
INTERACTIVE BLACK HOLE VISUALIZATION 🕳️
3D gravity well with interactive surface and stars
"""

import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

print("🕳️ Creating Interactive Black Hole Visualization...")

# Load data
df = pd.read_csv('cross_college_PES_cleaned.csv')
df['Date_OA'] = pd.to_datetime(df['Date_OA'], format='%d-%m-%Y', errors='coerce')
df_valid = df.dropna(subset=['Date_OA']).copy()
df_valid['Days'] = (df_valid['Date_OA'] - df_valid['Date_OA'].min()).dt.days

# Prepare surface data
tier_map = {'Tier 1': 3, 'Tier 2': 2, 'Tier 3': 1}
df_surface = df_valid[df_valid['CGPA_Cutoff'].notna()].copy()
df_surface['TierNum'] = df_surface['Tier'].map(tier_map)
df_surface = df_surface.dropna(subset=['TierNum', 'Days'])

# Create grid
x_grid = np.linspace(df_surface['Days'].min(), df_surface['Days'].max(), 60)
y_grid = np.linspace(1, 3, 60)
X, Y = np.meshgrid(x_grid, y_grid)

# Interpolate CGPA values
points = df_surface[['Days', 'TierNum']].values
values = df_surface['CGPA_Cutoff'].values

# Remove NaN
mask = ~(np.isnan(points).any(axis=1) | np.isnan(values))
points = points[mask]
values = values[mask]

Z = griddata(points, values, (X, Y), method='linear')
Z = np.nan_to_num(Z, nan=7.0)

# Create gravity well (invert)
Z_well = 10 - Z

# Create figure
fig = go.Figure()

# Add the gravity well surface
fig.add_trace(go.Surface(
    x=X, y=Y, z=Z_well,
    colorscale='Twilight',
    opacity=0.9,
    name='Accessibility Well',
    showscale=True,
    colorbar=dict(
        title=dict(text='Depth<br>(Accessibility)', side='right'),
        tickmode='linear',
        tick0=0,
        dtick=1
    ),
    hovertemplate='Days: %{x}<br>Tier: %{y}<br>Accessibility: %{z:.2f}<extra></extra>'
))

# Add companies as stars (only premium/notable ones to keep it clean)
tier_colors = {'Tier 1': '#FFD700', 'Tier 2': '#00CED1', 'Tier 3': '#FF69B4'}

# Only show companies with high CTC or interesting CGPA cutoffs
df_notable = df_valid[
    (df_valid['CGPA_Cutoff'].notna()) & 
    ((df_valid['CTC_LPA'] >= 20) | (df_valid['CGPA_Cutoff'] >= 8.5) | (df_valid['CGPA_Cutoff'] <= 6.5))
].copy()

for tier, color in tier_colors.items():
    tier_df = df_notable[df_notable['Tier'] == tier].copy()
    if len(tier_df) == 0:
        continue
        
    tier_num = tier_map[tier]
    z_vals = 10 - tier_df['CGPA_Cutoff']
    
    hover_text = [
        f"<b>{row['Company']}</b><br>" +
        f"CGPA: {row['CGPA_Cutoff']}<br>" +
        f"CTC: ₹{row['CTC_LPA'] if pd.notna(row['CTC_LPA']) else 'N/A'}L<br>" +
        f"Day: {row['Days']}<br>" +
        f"Date: {row['Date_OA'].strftime('%d %b %Y')}<br>" +
        f"Tier: {tier}"
        for _, row in tier_df.iterrows()
    ]
    
    fig.add_trace(go.Scatter3d(
        x=tier_df['Days'],
        y=[tier_num] * len(tier_df),
        z=z_vals,
        mode='markers',
        name=f'{tier} (Notable)',
        marker=dict(
            size=10,
            color=color,
            opacity=0.9,
            line=dict(color='white', width=2),
            symbol='diamond'
        ),
        hovertext=hover_text,
        hoverinfo='text'
    ))

# Add phase separators
phase_dates = [
    (0, df_surface['Days'].max() // 3, 'Premium Phase'),
    (df_surface['Days'].max() // 3, 2 * df_surface['Days'].max() // 3, 'Peak Season'),
    (2 * df_surface['Days'].max() // 3, df_surface['Days'].max(), 'Accessibility Phase')
]

for start, end, label in phase_dates:
    mid = (start + end) / 2
    # Add vertical plane markers
    fig.add_trace(go.Scatter3d(
        x=[start, start, start, start],
        y=[1, 3, 3, 1],
        z=[0, 0, 10, 10],
        mode='lines',
        line=dict(color='cyan', width=1, dash='dash'),
        opacity=0.3,
        showlegend=False,
        hoverinfo='skip'
    ))

# Update layout
fig.update_layout(
    title=dict(
        text='🕳️ THE BLACK HOLE: Interactive Accessibility Gravity Well<br>' +
             '<sub>Surface depth shows CGPA difficulty | Stars are companies falling into the well</sub>',
        font=dict(size=24, color='white')
    ),
    scene=dict(
        xaxis=dict(
            title='Timeline (Days)',
            backgroundcolor='#0a0a0a',
            gridcolor='#333',
            showbackground=True,
            range=[df_surface['Days'].min(), df_surface['Days'].max()]
        ),
        yaxis=dict(
            title='Tier Level',
            backgroundcolor='#0a0a0a',
            gridcolor='#333',
            showbackground=True,
            tickmode='array',
            tickvals=[1, 2, 3],
            ticktext=['Tier 3', 'Tier 2', 'Tier 1'],
            range=[0.5, 3.5]
        ),
        zaxis=dict(
            title='Accessibility<br>(Inverted CGPA)',
            backgroundcolor='#0a0a0a',
            gridcolor='#333',
            showbackground=True,
            range=[0, 10]
        ),
        bgcolor='#000000',
        camera=dict(
            eye=dict(x=1.8, y=1.8, z=1.3),
            center=dict(x=0, y=0, z=0)
        )
    ),
    paper_bgcolor='#000000',
    plot_bgcolor='#000000',
    font=dict(color='white', size=12),
    showlegend=True,
    legend=dict(
        bgcolor='rgba(0,0,0,0.8)',
        bordercolor='cyan',
        borderwidth=2,
        font=dict(size=11)
    ),
    height=900,
    margin=dict(l=0, r=0, t=100, b=0)
)

# Add annotations
fig.add_annotation(
    text='💡 Insight: Deeper wells = Higher CGPA requirements<br>' +
         'Companies sink into the well based on their accessibility barrier',
    xref='paper', yref='paper',
    x=0.5, y=0.02,
    showarrow=False,
    font=dict(size=11, color='cyan'),
    bgcolor='rgba(0,0,0,0.7)',
    bordercolor='cyan',
    borderwidth=1
)

fig.write_html('INTERACTIVE_BlackHole.html')
print("✓ Saved: INTERACTIVE_BlackHole.html")
print("\n🌌 Interactive Black Hole created!")
print("📂 File location: INTERACTIVE_BlackHole.html")
print("\n✨ Features:")
print("  🖱️  Drag to rotate and explore the gravity well")
print("  🔍 Zoom to see details")
print("  ⭐ Hover over stars (companies) for info")
print("  🎨 Color-coded by tier with glow effects")
print("\n🚀 Double-click the file to open in your browser!")
