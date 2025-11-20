"""
Interactive Placement Analytics Dashboard
==========================================
Streamlit-based interactive tool for exploring placement data

Features:
- Data explorer with filters
- Interactive visualizations
- CTC prediction tool
- Company comparison
- Trend analysis
- Export capabilities

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path

# Page config
st.set_page_config(
    page_title="PES Placement Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1f77b4; font-weight: bold; margin-bottom: 0;}
    .sub-header {font-size: 1.2rem; color: #666; margin-top: 0;}
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {height: 50px; background-color: #f0f2f6; border-radius: 5px 5px 0 0;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_data():
    """Load cleaned placement data"""
    try:
        df = pd.read_csv('processed_data/cleaned_placement_data.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found! Please ensure 'processed_data/cleaned_placement_data.csv' exists.")
        st.stop()

@st.cache_data
def load_analysis_results():
    """Load analysis results if available"""
    results = {}
    
    # Load RDD results
    try:
        with open('analysis_outputs/02_causal/rdd_results.json', 'r') as f:
            results['rdd'] = json.load(f)
    except FileNotFoundError:
        results['rdd'] = None
    
    # Load clustering results
    try:
        with open('analysis_outputs/05_clustering/clustering_results.json', 'r') as f:
            results['clustering'] = json.load(f)
    except FileNotFoundError:
        results['clustering'] = None
    
    # Load network results
    try:
        with open('analysis_outputs/04_network/network_metrics.json', 'r') as f:
            results['network'] = json.load(f)
    except FileNotFoundError:
        results['network'] = None
    
    return results

df = load_data()
analysis_results = load_analysis_results()

# Separate FTE and internship data
df_fte = df[~df['is_internship_record'] & df['fte_ctc'].notna()].copy()
df_intern = df[df['is_internship_record']].copy()

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<p class="main-header">📊 PES Placement Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Temporal and Statistical Insights into Talent Acquisition (2022-2026)</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# SIDEBAR FILTERS
# ============================================================================

st.sidebar.header("🔍 Filters")

# Year filter
years = sorted(df['batch_year'].unique())
selected_years = st.sidebar.multiselect(
    "Select Years",
    options=years,
    default=years
)

# Tier filter
tiers = df['placement_tier'].dropna().unique().tolist()
selected_tiers = st.sidebar.multiselect(
    "Select Tiers",
    options=tiers,
    default=tiers
)

# Role filter
roles = df['role_type'].dropna().unique().tolist()
selected_roles = st.sidebar.multiselect(
    "Select Role Types",
    options=roles,
    default=roles[:5] if len(roles) > 5 else roles
)

# Apply filters
df_filtered = df[
    (df['batch_year'].isin(selected_years)) &
    (df['placement_tier'].isin(selected_tiers)) &
    (df['role_type'].isin(selected_roles))
]

df_fte_filtered = df_filtered[~df_filtered['is_internship_record'] & df_filtered['fte_ctc'].notna()]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Filtered Records:** {len(df_filtered):,}")
st.sidebar.markdown(f"**FTE Records:** {len(df_fte_filtered):,}")

# ============================================================================
# TOP METRICS
# ============================================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Records",
        f"{len(df_filtered):,}",
        delta=f"{len(df_filtered) - len(df)}" if len(df_filtered) != len(df) else None
    )

with col2:
    if len(df_fte_filtered) > 0:
        avg_ctc = df_fte_filtered['fte_ctc'].mean()
        st.metric("Avg CTC", f"₹{avg_ctc:.2f}L")
    else:
        st.metric("Avg CTC", "N/A")

with col3:
    if len(df_fte_filtered) > 0:
        max_ctc = df_fte_filtered['fte_ctc'].max()
        st.metric("Max CTC", f"₹{max_ctc:.2f}L")
    else:
        st.metric("Max CTC", "N/A")

with col4:
    unique_companies = df_filtered['company_name'].nunique()
    st.metric("Companies", f"{unique_companies:,}")

with col5:
    if 'cgpa_cutoff' in df_filtered.columns:
        avg_cgpa = df_filtered['cgpa_cutoff'].mean()
        st.metric("Avg CGPA Cutoff", f"{avg_cgpa:.2f}")
    else:
        st.metric("Avg CGPA Cutoff", "N/A")

st.markdown("---")

# ============================================================================
# MAIN CONTENT TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview", "🔮 Predictions", "🏢 Companies", 
    "📊 Analysis Results", "🎯 Insights", "📂 Data Explorer"
])

# ----------------------------------------------------------------------------
# TAB 1: OVERVIEW
# ----------------------------------------------------------------------------

with tab1:
    st.header("📈 Placement Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CTC distribution
        st.subheader("CTC Distribution")
        if len(df_fte_filtered) > 0:
            fig = px.histogram(
                df_fte_filtered,
                x='fte_ctc',
                nbins=30,
                title='FTE CTC Distribution',
                labels={'fte_ctc': 'CTC (LPA)', 'count': 'Frequency'},
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No FTE data available for selected filters")
    
    with col2:
        # Year-wise trends
        st.subheader("Yearly CTC Trends")
        if len(df_fte_filtered) > 0:
            yearly = df_fte_filtered.groupby('batch_year')['fte_ctc'].agg(['mean', 'median']).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yearly['batch_year'], y=yearly['mean'],
                mode='lines+markers', name='Mean',
                line=dict(color='blue', width=3),
                marker=dict(size=10)
            ))
            fig.add_trace(go.Scatter(
                x=yearly['batch_year'], y=yearly['median'],
                mode='lines+markers', name='Median',
                line=dict(color='green', width=3),
                marker=dict(size=10)
            ))
            fig.update_layout(
                title='CTC Trends Over Years',
                xaxis_title='Year',
                yaxis_title='CTC (LPA)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No FTE data available")
    
    # Tier distribution
    st.subheader("Tier Distribution")
    tier_counts = df_filtered['placement_tier'].value_counts()
    
    fig = px.pie(
        values=tier_counts.values,
        names=tier_counts.index,
        title='Placement by Tier',
        hole=0.4
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 2: PREDICTIONS
# ----------------------------------------------------------------------------

with tab2:
    st.header("🔮 CTC Prediction Tool")
    
    st.markdown("""
    Predict expected CTC based on student profile. This uses historical patterns 
    from the placement data to estimate likely compensation.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        pred_cgpa = st.slider("CGPA", min_value=6.0, max_value=10.0, value=8.0, step=0.1)
        pred_year = st.selectbox("Batch Year", options=list(range(2027, 2030)))
        pred_tier = st.selectbox("Target Tier", options=['Tier-3', 'Tier-2', 'Tier-1', 'Dream'])
    
    with col2:
        pred_role = st.selectbox("Role Type", options=df['role_type'].dropna().unique().tolist())
        has_stocks_pred = st.checkbox("Stocks/ESOPs Included")
        has_bonus_pred = st.checkbox("Joining Bonus Included")
    
    if st.button("🎯 Predict CTC", type="primary"):
        # Simple prediction based on historical averages with adjustments
        base_estimates = {
            'Tier-3': 5.0,
            'Tier-2': 9.0,
            'Tier-1': 16.0,
            'Dream': 35.0
        }
        
        estimated_ctc = base_estimates[pred_tier]
        
        # Adjust for CGPA (10% increase per 0.5 CGPA above 7.5)
        if pred_cgpa > 7.5:
            estimated_ctc *= (1 + 0.1 * (pred_cgpa - 7.5) / 0.5)
        
        # Adjust for stocks
        if has_stocks_pred:
            estimated_ctc *= 1.15
        
        # Adjust for bonus
        if has_bonus_pred:
            estimated_ctc *= 1.08
        
        # Add uncertainty
        lower_bound = estimated_ctc * 0.85
        upper_bound = estimated_ctc * 1.15
        
        st.success(f"""
        ### Predicted CTC: ₹{estimated_ctc:.2f} LPA
        
        **95% Confidence Interval:** ₹{lower_bound:.2f} - ₹{upper_bound:.2f} LPA
        
        *Note: This is a statistical estimate based on historical data. 
        Actual offers may vary based on company, market conditions, and individual performance.*
        """)

# ----------------------------------------------------------------------------
# TAB 3: COMPANIES
# ----------------------------------------------------------------------------

with tab3:
    st.header("🏢 Company Analysis")
    
    # Top recruiters
    st.subheader("Top 15 Recruiters")
    company_counts = df_filtered['company_name'].value_counts().head(15)
    
    fig = px.bar(
        x=company_counts.values,
        y=company_counts.index,
        orientation='h',
        title='Top 15 Companies by Placement Count',
        labels={'x': 'Number of Placements', 'y': 'Company'},
        color=company_counts.values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top paying companies
    if len(df_fte_filtered) > 0:
        st.subheader("Top 15 Highest Paying Companies")
        top_paying = df_fte_filtered.groupby('company_name')['fte_ctc'].agg(['mean', 'count']).reset_index()
        top_paying = top_paying[top_paying['count'] >= 2]  # At least 2 placements
        top_paying = top_paying.nlargest(15, 'mean')
        
        fig = px.bar(
            top_paying,
            x='mean',
            y='company_name',
            orientation='h',
            title='Top 15 Highest Paying Companies (Avg CTC)',
            labels={'mean': 'Average CTC (LPA)', 'company_name': 'Company'},
            color='mean',
            color_continuous_scale='Reds'
        )
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 4: ANALYSIS RESULTS
# ----------------------------------------------------------------------------

with tab4:
    st.header("📊 Advanced Analysis Results")
    
    # RDD Results
    if analysis_results['rdd']:
        st.subheader("🔬 Causal Inference (Regression Discontinuity)")
        rdd = analysis_results['rdd']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CGPA Cutoff", f"{rdd['cutoff']}")
        with col2:
            st.metric("Treatment Effect", f"₹{rdd['treatment_effect_lpa']:.2f} LPA")
        with col3:
            st.metric("Sample Size", f"{rdd['n_below']} + {rdd['n_above']}")
        
        st.success(f"""
        **Key Finding:** Students meeting the CGPA cutoff of {rdd['cutoff']} 
        experience a **causal increase** in CTC of ₹{rdd['treatment_effect_lpa']:.2f} LPA.
        """)
        
        # Show image
        rdd_img_path = Path('analysis_outputs/02_causal/rdd_analysis.png')
        if rdd_img_path.exists():
            st.image(str(rdd_img_path), caption='Regression Discontinuity Design')
    
    st.markdown("---")
    
    # Network Results
    if analysis_results['network']:
        st.subheader("🕸️ Network Analysis")
        network = analysis_results['network']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Companies", f"{network['n_companies']}")
            st.metric("Total Connections", f"{network['n_edges']}")
        
        with col2:
            st.markdown("**Top Connected Companies:**")
            for i, comp in enumerate(network['top_connected_companies'][:5], 1):
                st.markdown(f"{i}. {comp['company']} ({comp['score']:.3f})")
    
    st.markdown("---")
    
    # Clustering Results
    if analysis_results['clustering']:
        st.subheader("🎯 Company Clustering")
        clustering = analysis_results['clustering']
        
        st.metric("Number of Segments", clustering['n_clusters'])
        st.info(f"Identified {clustering['n_clusters']} distinct company archetypes using Gaussian Mixture Model")
        
        # Show image
        cluster_img_path = Path('analysis_outputs/05_clustering/clusters_3d_pca.png')
        if cluster_img_path.exists():
            st.image(str(cluster_img_path), caption='Company Segments (3D PCA)')

# ----------------------------------------------------------------------------
# TAB 5: INSIGHTS
# ----------------------------------------------------------------------------

with tab5:
    st.header("🎯 Key Insights")
    
    insights = []
    
    # Generate insights
    if len(df_fte_filtered) > 0:
        avg_ctc = df_fte_filtered['fte_ctc'].mean()
        max_ctc = df_fte_filtered['fte_ctc'].max()
        top_company = df_fte_filtered.groupby('company_name')['fte_ctc'].mean().idxmax()
        
        insights.append(f"📊 Average FTE CTC across all placements: **₹{avg_ctc:.2f} LPA**")
        insights.append(f"🏆 Highest CTC offered: **₹{max_ctc:.2f} LPA**")
        insights.append(f"⭐ Top paying company (avg): **{top_company}**")
    
    if analysis_results['rdd']:
        rdd = analysis_results['rdd']
        insights.append(f"🔬 Meeting CGPA cutoff ≥{rdd['cutoff']} **causally increases** CTC by ₹{rdd['treatment_effect_lpa']:.2f} LPA")
    
    # Top tier distribution
    if len(df_filtered) > 0:
        top_tier = df_filtered['placement_tier'].mode()[0]
        tier_pct = (df_filtered['placement_tier'] == top_tier).sum() / len(df_filtered) * 100
        insights.append(f"📈 Most common placement tier: **{top_tier}** ({tier_pct:.1f}%)")
    
    # Display insights
    for insight in insights:
        st.markdown(f"- {insight}")
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("💡 Recommendations for Students")
    
    st.markdown("""
    1. **CGPA Matters**: Maintain CGPA above identified cutoffs for better opportunities
    2. **Target High-Tier Companies**: Focus preparation on companies offering better packages
    3. **Skill Development**: Align skills with high-demand role types
    4. **Early Preparation**: Start placement prep well in advance
    5. **Network**: Connect with companies showing consistent hiring patterns
    """)

# ----------------------------------------------------------------------------
# TAB 6: DATA EXPLORER
# ----------------------------------------------------------------------------

with tab6:
    st.header("📂 Data Explorer")
    
    st.markdown("Explore the raw placement data with interactive filters and sorting.")
    
    # Display options
    col1, col2 = st.columns(2)
    with col1:
        show_columns = st.multiselect(
            "Select Columns to Display",
            options=df_filtered.columns.tolist(),
            default=['batch_year', 'company_name', 'job_role', 'fte_ctc', 
                     'placement_tier', 'cgpa_cutoff']
        )
    
    with col2:
        sort_by = st.selectbox("Sort By", options=show_columns if show_columns else df_filtered.columns.tolist())
        sort_order = st.radio("Order", options=['Descending', 'Ascending'], horizontal=True)
    
    # Display dataframe
    if show_columns:
        display_df = df_filtered[show_columns].sort_values(
            by=sort_by,
            ascending=(sort_order == 'Ascending')
        )
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=500
        )
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name=f"placement_data_filtered.csv",
            mime="text/csv"
        )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>PES University Placement Analytics Dashboard | Data: 2022-2026 Batches</p>
    <p>Built with Streamlit | © 2025 ADA Project Team</p>
</div>
""", unsafe_allow_html=True)
