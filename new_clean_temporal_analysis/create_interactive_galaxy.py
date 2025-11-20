"""
INTERACTIVE 3D GALAXY PLOTS 🌌
Using Plotly for fully interactive, rotatable visualizations
Open the HTML files in your browser to explore!
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🚀 Launching Interactive Galaxy Engine...")
print("Creating rotatable 3D visualizations with Plotly\n")

# Load data
df = pd.read_csv('cross_college_PES_cleaned.csv')
df['Date_OA'] = pd.to_datetime(df['Date_OA'], format='%d-%m-%Y', errors='coerce')
df_valid = df.dropna(subset=['Date_OA']).copy()
df_valid['Days'] = (df_valid['Date_OA'] - df_valid['Date_OA'].min()).dt.days
print(f"✓ {len(df_valid)} companies loaded\n")

# ============================================================================
# INTERACTIVE 1: The 3D Galaxy - Main View
# ============================================================================
print("🌌 Creating INTERACTIVE 1: The 3D Galaxy...")

df_3d = df_valid[df_valid['CTC_LPA'].notna() & df_valid['CGPA_Cutoff'].notna()].copy()

fig = go.Figure()

# Color mapping for tiers
tier_colors = {'Tier 1': '#FFD700', 'Tier 2': '#00CED1', 'Tier 3': '#FF69B4'}

# Add scatter for each tier
for tier, color in tier_colors.items():
    tier_df = df_3d[df_3d['Tier'] == tier]
    
    # Create hover text
    hover_text = [
        f"<b>{row['Company']}</b><br>" +
        f"CTC: ₹{row['CTC_LPA']}L<br>" +
        f"CGPA: {row['CGPA_Cutoff']}<br>" +
        f"Day: {row['Days']}<br>" +
        f"Date: {row['Date_OA'].strftime('%d %b %Y')}<br>" +
        f"Role: {row['Role']}<br>" +
        f"Offers: {row['Total_Offers']}"
        for _, row in tier_df.iterrows()
    ]
    
    fig.add_trace(go.Scatter3d(
        x=tier_df['Days'],
        y=tier_df['CGPA_Cutoff'],
        z=tier_df['CTC_LPA'],
        mode='markers',
        name=tier,
        marker=dict(
            size=tier_df['Total_Offers'].fillna(1) * 3,
            color=color,
            opacity=0.8,
            line=dict(color='white', width=1)
        ),
        hovertext=hover_text,
        hoverinfo='text'
    ))

# Update layout
fig.update_layout(
    title=dict(
        text='🌌 THE INTERACTIVE GALAXY: 3D Placement Universe<br><sub>Drag to rotate | Scroll to zoom | Hover for details</sub>',
        font=dict(size=24, color='white')
    ),
    scene=dict(
        xaxis=dict(title='Days into Season', backgroundcolor='#0a0a0a', gridcolor='#333', showbackground=True),
        yaxis=dict(title='CGPA Cutoff', backgroundcolor='#0a0a0a', gridcolor='#333', showbackground=True),
        zaxis=dict(title='CTC (LPA)', backgroundcolor='#0a0a0a', gridcolor='#333', showbackground=True),
        bgcolor='#000000',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.3)
        )
    ),
    paper_bgcolor='#000000',
    plot_bgcolor='#000000',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        bgcolor='rgba(0,0,0,0.7)',
        bordercolor='white',
        borderwidth=1
    ),
    height=800
)

fig.write_html('INTERACTIVE1_Galaxy_3D.html')
print("✓ Saved: INTERACTIVE1_Galaxy_3D.html\n")

# ============================================================================
# INTERACTIVE 2: The Temporal Spiral
# ============================================================================
print("🌀 Creating INTERACTIVE 2: The Temporal Spiral...")

df_spiral = df_valid[df_valid['CTC_LPA'].notna()].copy()

# Convert timeline to spiral (cylindrical coords)
total_days = df_spiral['Days'].max()
df_spiral['Theta'] = (df_spiral['Days'] / total_days) * 4 * np.pi  # 2 full rotations
df_spiral['R'] = df_spiral['CTC_LPA']
df_spiral['X'] = df_spiral['R'] * np.cos(df_spiral['Theta'])
df_spiral['Y'] = df_spiral['R'] * np.sin(df_spiral['Theta'])
df_spiral['Z'] = df_spiral['Days']

fig2 = go.Figure()

for tier, color in tier_colors.items():
    tier_df = df_spiral[df_spiral['Tier'] == tier]
    
    hover_text = [
        f"<b>{row['Company']}</b><br>" +
        f"CTC: ₹{row['CTC_LPA']}L<br>" +
        f"Day: {row['Days']}<br>" +
        f"Date: {row['Date_OA'].strftime('%d %b %Y')}"
        for _, row in tier_df.iterrows()
    ]
    
    fig2.add_trace(go.Scatter3d(
        x=tier_df['X'],
        y=tier_df['Y'],
        z=tier_df['Z'],
        mode='markers',
        name=tier,
        marker=dict(
            size=8,
            color=color,
            opacity=0.8,
            line=dict(color='white', width=1)
        ),
        hovertext=hover_text,
        hoverinfo='text'
    ))
    
    # Add connecting line to create spiral
    tier_sorted = tier_df.sort_values('Days')
    if len(tier_sorted) > 1:
        fig2.add_trace(go.Scatter3d(
            x=tier_sorted['X'],
            y=tier_sorted['Y'],
            z=tier_sorted['Z'],
            mode='lines',
            line=dict(color=color, width=2),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))

fig2.update_layout(
    title=dict(
        text='🌀 THE TEMPORAL SPIRAL: Timeline Helix<br><sub>Watch the placement season unfold in 3D space</sub>',
        font=dict(size=24, color='white')
    ),
    scene=dict(
        xaxis=dict(title='X (CTC × cos(θ))', backgroundcolor='#0a0a0a', gridcolor='#333', showbackground=True),
        yaxis=dict(title='Y (CTC × sin(θ))', backgroundcolor='#0a0a0a', gridcolor='#333', showbackground=True),
        zaxis=dict(title='Timeline (Days)', backgroundcolor='#0a0a0a', gridcolor='#333', showbackground=True),
        bgcolor='#000000'
    ),
    paper_bgcolor='#000000',
    plot_bgcolor='#000000',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(bgcolor='rgba(0,0,0,0.7)', bordercolor='white', borderwidth=1),
    height=800
)

fig2.write_html('INTERACTIVE2_Spiral.html')
print("✓ Saved: INTERACTIVE2_Spiral.html\n")

# ============================================================================
# INTERACTIVE 3: The Clustering Sphere
# ============================================================================
print("🔮 Creating INTERACTIVE 3: The Clustering Sphere...")

# Map to spherical coordinates
df_sphere = df_valid[df_valid['CTC_LPA'].notna() & df_valid['CGPA_Cutoff'].notna()].copy()

# Normalize values to [0, 1]
df_sphere['Days_norm'] = (df_sphere['Days'] - df_sphere['Days'].min()) / (df_sphere['Days'].max() - df_sphere['Days'].min())
df_sphere['CGPA_norm'] = (df_sphere['CGPA_Cutoff'] - df_sphere['CGPA_Cutoff'].min()) / (df_sphere['CGPA_Cutoff'].max() - df_sphere['CGPA_Cutoff'].min())
df_sphere['CTC_norm'] = df_sphere['CTC_LPA'] / df_sphere['CTC_LPA'].max()

# Spherical coordinates
df_sphere['Phi'] = df_sphere['Days_norm'] * np.pi  # Latitude
df_sphere['Theta2'] = df_sphere['CGPA_norm'] * 2 * np.pi  # Longitude
df_sphere['R2'] = 10 + df_sphere['CTC_LPA']  # Radius = 10 + CTC

df_sphere['X2'] = df_sphere['R2'] * np.sin(df_sphere['Phi']) * np.cos(df_sphere['Theta2'])
df_sphere['Y2'] = df_sphere['R2'] * np.sin(df_sphere['Phi']) * np.sin(df_sphere['Theta2'])
df_sphere['Z2'] = df_sphere['R2'] * np.cos(df_sphere['Phi'])

fig3 = go.Figure()

for tier, color in tier_colors.items():
    tier_df = df_sphere[df_sphere['Tier'] == tier]
    
    hover_text = [
        f"<b>{row['Company']}</b><br>" +
        f"CTC: ₹{row['CTC_LPA']}L<br>" +
        f"CGPA: {row['CGPA_Cutoff']}<br>" +
        f"Day: {row['Days']}"
        for _, row in tier_df.iterrows()
    ]
    
    fig3.add_trace(go.Scatter3d(
        x=tier_df['X2'],
        y=tier_df['Y2'],
        z=tier_df['Z2'],
        mode='markers',
        name=tier,
        marker=dict(
            size=10,
            color=color,
            opacity=0.9,
            line=dict(color='white', width=2)
        ),
        hovertext=hover_text,
        hoverinfo='text'
    ))

# Add reference sphere
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x_sphere = 10 * np.outer(np.cos(u), np.sin(v))
y_sphere = 10 * np.outer(np.sin(u), np.sin(v))
z_sphere = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

fig3.add_trace(go.Surface(
    x=x_sphere, y=y_sphere, z=z_sphere,
    opacity=0.1,
    colorscale=[[0, '#1a1a1a'], [1, '#1a1a1a']],
    showscale=False,
    hoverinfo='skip'
))

fig3.update_layout(
    title=dict(
        text='🔮 THE CLUSTERING SPHERE: Spherical Placement Space<br><sub>Distance from center = Total compensation value</sub>',
        font=dict(size=24, color='white')
    ),
    scene=dict(
        xaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
        zaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
        bgcolor='#000000'
    ),
    paper_bgcolor='#000000',
    plot_bgcolor='#000000',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(bgcolor='rgba(0,0,0,0.7)', bordercolor='white', borderwidth=1),
    height=800
)

fig3.write_html('INTERACTIVE3_Sphere.html')
print("✓ Saved: INTERACTIVE3_Sphere.html\n")

# ============================================================================
# INTERACTIVE 4: The Animated Timeline
# ============================================================================
print("📽️ Creating INTERACTIVE 4: The Animated Timeline...")

df_anim = df_valid[df_valid['CTC_LPA'].notna() & df_valid['CGPA_Cutoff'].notna()].copy()
df_anim = df_anim.sort_values('Date_OA')

# Create frames for animation
df_anim['Week'] = df_anim['Date_OA'].dt.to_period('W').dt.to_timestamp()
weeks = sorted(df_anim['Week'].unique())

fig4 = go.Figure()

# Initial frame (first week)
first_week = df_anim[df_anim['Week'] == weeks[0]]

for tier, color in tier_colors.items():
    tier_df = first_week[first_week['Tier'] == tier]
    fig4.add_trace(go.Scatter3d(
        x=tier_df['Days'],
        y=tier_df['CGPA_Cutoff'],
        z=tier_df['CTC_LPA'],
        mode='markers',
        name=tier,
        marker=dict(size=10, color=color, opacity=0.8, line=dict(color='white', width=1))
    ))

# Create frames
frames = []
for week in weeks:
    frame_data = []
    week_data = df_anim[df_anim['Week'] <= week]  # Cumulative
    
    for tier, color in tier_colors.items():
        tier_df = week_data[week_data['Tier'] == tier]
        frame_data.append(go.Scatter3d(
            x=tier_df['Days'],
            y=tier_df['CGPA_Cutoff'],
            z=tier_df['CTC_LPA'],
            mode='markers',
            name=tier,
            marker=dict(size=10, color=color, opacity=0.8, line=dict(color='white', width=1))
        ))
    
    frames.append(go.Frame(data=frame_data, name=week.strftime('%Y-%m-%d')))

fig4.frames = frames

# Add play/pause buttons
fig4.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'buttons': [
            {'label': '▶ Play', 'method': 'animate', 'args': [None, {
                'frame': {'duration': 500, 'redraw': True},
                'fromcurrent': True,
                'mode': 'immediate'
            }]},
            {'label': '⏸ Pause', 'method': 'animate', 'args': [[None], {
                'frame': {'duration': 0, 'redraw': False},
                'mode': 'immediate'
            }]}
        ],
        'x': 0.1,
        'y': 0,
        'xanchor': 'left',
        'yanchor': 'bottom'
    }],
    sliders=[{
        'active': 0,
        'steps': [{'args': [[f.name], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                  'label': f.name, 'method': 'animate'} for f in frames],
        'x': 0.1,
        'y': 0,
        'currentvalue': {'prefix': 'Week: ', 'visible': True, 'xanchor': 'right'},
        'len': 0.9
    }],
    title=dict(
        text='📽️ THE ANIMATED TIMELINE: Watch Placements Unfold<br><sub>Press Play to see companies arrive week by week</sub>',
        font=dict(size=24, color='white')
    ),
    scene=dict(
        xaxis=dict(title='Days', backgroundcolor='#0a0a0a', gridcolor='#333', showbackground=True),
        yaxis=dict(title='CGPA', backgroundcolor='#0a0a0a', gridcolor='#333', showbackground=True),
        zaxis=dict(title='CTC (LPA)', backgroundcolor='#0a0a0a', gridcolor='#333', showbackground=True),
        bgcolor='#000000'
    ),
    paper_bgcolor='#000000',
    plot_bgcolor='#000000',
    font=dict(color='white'),
    height=800
)

fig4.write_html('INTERACTIVE4_Animated.html')
print("✓ Saved: INTERACTIVE4_Animated.html\n")

# ============================================================================
print("\n" + "="*80)
print("🎮 INTERACTIVE GALAXY VISUALIZATIONS COMPLETE! 🎮")
print("="*80)
print("\n🌟 Generated Files (Open in Browser):")
print("  1. INTERACTIVE1_Galaxy_3D.html - Full 3D galaxy with hover info")
print("  2. INTERACTIVE2_Spiral.html - Timeline spiral helix")
print("  3. INTERACTIVE3_Sphere.html - Spherical clustering view")
print("  4. INTERACTIVE4_Animated.html - Animated timeline (with play button!)")
print("\n✨ Features:")
print("  🖱️  Drag to rotate view")
print("  🔍 Scroll to zoom in/out")
print("  👆 Hover over points for company details")
print("  📸 Click camera icon to save as PNG")
print("  ▶️  Animation controls in INTERACTIVE4")
print("\n🚀 Double-click any HTML file to open in browser!")
print("="*80)
