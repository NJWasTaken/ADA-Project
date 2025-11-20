"""
PES Placement Analytics Dashboard
==================================
Interactive Streamlit dashboard for exploring placement data (2022-2026)

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
from temporal_analysis import TemporalAnalyzer

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="PES Placement Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 0;
        text-align: center;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-top: 0;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #0f0f0f;
        border-radius: 5px 5px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_data():
    """Load placement data"""
    try:
        df = pd.read_csv('processed_data/placement_data.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found! Run `python3 consolidate_placement_data.py` first.")
        st.stop()

@st.cache_data
def load_summary():
    """Load summary statistics"""
    try:
        with open('processed_data/summary_statistics.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_temporal_analysis():
    """Load temporal analysis results"""
    try:
        with open('processed_data/temporal_analysis.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_resource
def get_temporal_analyzer():
    """Get temporal analyzer instance"""
    return TemporalAnalyzer()

df = load_data()
summary = load_summary()
temporal_data = load_temporal_analysis()
analyzer = get_temporal_analyzer()

# Separate FTE and internship data
df_fte = df[~df['is_internship'] & df['has_ctc_data']].copy()
df_intern = df[df['is_internship']].copy()

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<p class="main-header">📊 PES Placement Analytics</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Data-Driven Insights | 2022-2026 Batches</p>', unsafe_allow_html=True)
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
tiers = sorted(df['tier'].dropna().unique().tolist())
selected_tiers = st.sidebar.multiselect(
    "Select Tiers",
    options=tiers,
    default=tiers
)

# Company search
company_search = st.sidebar.text_input("Search Company (optional)", "")

# Apply filters
df_filtered = df[
    (df['batch_year'].isin(selected_years)) &
    (df['tier'].isin(selected_tiers))
]

if company_search:
    df_filtered = df_filtered[
        df_filtered['company_name'].str.contains(company_search, case=False, na=False)
    ]

df_fte_filtered = df_filtered[~df_filtered['is_internship'] & df_filtered['has_ctc_data']]
df_intern_filtered = df_filtered[df_filtered['is_internship']]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Filtered Records:** {len(df_filtered):,}")
st.sidebar.markdown(f"**FTE:** {len(df_fte_filtered):,} | **Internships:** {len(df_intern_filtered):,}")

# ============================================================================
# TOP METRICS
# ============================================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Records", f"{len(df_filtered):,}")

with col2:
    if len(df_fte_filtered) > 0:
        avg_ctc = df_fte_filtered['total_ctc'].mean()
        st.metric("Avg FTE CTC", f"₹{avg_ctc:.2f}L")
    else:
        st.metric("Avg FTE CTC", "N/A")

with col3:
    if len(df_fte_filtered) > 0:
        max_ctc = df_fte_filtered['total_ctc'].max()
        st.metric("Max FTE CTC", f"₹{max_ctc:.2f}L")
    else:
        st.metric("Max FTE CTC", "N/A")

with col4:
    unique_companies = df_filtered['company_name'].nunique()
    st.metric("Unique Companies", f"{unique_companies:,}")

with col5:
    if df_filtered['has_cgpa_data'].sum() > 0:
        avg_cgpa = df_filtered[df_filtered['has_cgpa_data']]['cgpa_cutoff'].mean()
        st.metric("Avg CGPA Cutoff", f"{avg_cgpa:.2f}")
    else:
        st.metric("Avg CGPA Cutoff", "N/A")

st.markdown("---")

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview", "🏢 Companies", "💰 Salary Analysis", "🎯 Insights", "📂 Data Explorer", "🔮 Temporal Analysis & Predictions"
])

# ----------------------------------------------------------------------------
# TAB 1: OVERVIEW
# ----------------------------------------------------------------------------

with tab1:
    st.header("📈 Placement Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("FTE CTC Distribution")
        if len(df_fte_filtered) > 0:
            fig = px.histogram(
                df_fte_filtered,
                x='total_ctc',
                nbins=40,
                title='FTE CTC Distribution',
                labels={'total_ctc': 'CTC (LPA)', 'count': 'Frequency'},
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No FTE data available for selected filters")

    with col2:
        st.subheader("Yearly CTC Trends")
        if len(df_fte_filtered) > 0:
            yearly = df_fte_filtered.groupby('batch_year')['total_ctc'].agg(['mean', 'median', 'count']).reset_index()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yearly['batch_year'], y=yearly['mean'],
                mode='lines+markers', name='Mean CTC',
                line=dict(color='blue', width=3),
                marker=dict(size=10)
            ))
            fig.add_trace(go.Scatter(
                x=yearly['batch_year'], y=yearly['median'],
                mode='lines+markers', name='Median CTC',
                line=dict(color='green', width=3),
                marker=dict(size=10)
            ))
            fig.update_layout(
                title='CTC Trends Over Years',
                xaxis_title='Batch Year',
                yaxis_title='CTC (LPA)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No FTE data available")

    # Tier distribution
    st.subheader("Placement Distribution by Tier")

    col1, col2 = st.columns(2)

    with col1:
        tier_counts = df_filtered['tier'].value_counts()
        fig = px.pie(
            values=tier_counts.values,
            names=tier_counts.index,
            title='Records by Tier',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Year-wise breakdown
        year_tier = df_filtered.groupby(['batch_year', 'tier']).size().reset_index(name='count')
        fig = px.bar(
            year_tier,
            x='batch_year',
            y='count',
            color='tier',
            title='Year-wise Tier Distribution',
            labels={'batch_year': 'Batch Year', 'count': 'Number of Records'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=400, barmode='stack')
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 2: COMPANIES
# ----------------------------------------------------------------------------

with tab2:
    st.header("🏢 Company Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 15 Recruiters by Volume")
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

    with col2:
        st.subheader("Top 15 Highest Paying Companies")
        if len(df_fte_filtered) > 0:
            top_paying = df_fte_filtered.groupby('company_name')['total_ctc'].agg(['mean', 'count']).reset_index()
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
        else:
            st.info("No FTE data available")

    # Company details table
    st.subheader("Company Details")
    if len(df_fte_filtered) > 0:
        company_stats = df_fte_filtered.groupby('company_name').agg({
            'total_ctc': ['mean', 'median', 'max', 'count'],
            'base_salary': 'mean',
            'cgpa_cutoff': 'mean'
        }).round(2)

        company_stats.columns = ['Avg CTC', 'Median CTC', 'Max CTC', 'Placements', 'Avg Base', 'Avg CGPA Cutoff']
        company_stats = company_stats.sort_values('Avg CTC', ascending=False)

        st.dataframe(
            company_stats.head(20),
            use_container_width=True,
            height=400
        )

# ----------------------------------------------------------------------------
# TAB 3: SALARY ANALYSIS
# ----------------------------------------------------------------------------

with tab3:
    st.header("💰 Salary Analysis")

    if len(df_fte_filtered) > 0:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("CTC Distribution by Tier")
            fig = px.box(
                df_fte_filtered,
                x='tier',
                y='total_ctc',
                title='CTC Distribution by Tier',
                labels={'tier': 'Tier', 'total_ctc': 'CTC (LPA)'},
                color='tier',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("CTC Distribution by Year")
            fig = px.box(
                df_fte_filtered,
                x='batch_year',
                y='total_ctc',
                title='CTC Distribution by Batch Year',
                labels={'batch_year': 'Batch Year', 'total_ctc': 'CTC (LPA)'},
                color='batch_year',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Percentiles
        st.subheader("Salary Percentiles")
        percentiles = df_fte_filtered['total_ctc'].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).round(2)

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("10th %ile", f"₹{percentiles[0.1]:.2f}L")
        col2.metric("25th %ile", f"₹{percentiles[0.25]:.2f}L")
        col3.metric("Median", f"₹{percentiles[0.5]:.2f}L")
        col4.metric("75th %ile", f"₹{percentiles[0.75]:.2f}L")
        col5.metric("90th %ile", f"₹{percentiles[0.9]:.2f}L")
        col6.metric("95th %ile", f"₹{percentiles[0.95]:.2f}L")

        # CGPA vs CTC scatter
        if df_fte_filtered['has_cgpa_data'].sum() > 10:
            st.subheader("CGPA Cutoff vs CTC")
            cgpa_data = df_fte_filtered[df_fte_filtered['has_cgpa_data']]

            fig = px.scatter(
                cgpa_data,
                x='cgpa_cutoff',
                y='total_ctc',
                title='CGPA Cutoff vs CTC',
                labels={'cgpa_cutoff': 'CGPA Cutoff', 'total_ctc': 'CTC (LPA)'},
                color='tier',
                hover_data=['company_name'],
                opacity=0.6
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No FTE data available for selected filters")

# ----------------------------------------------------------------------------
# TAB 4: INSIGHTS
# ----------------------------------------------------------------------------

with tab4:
    st.header("🎯 Key Insights")

    if summary:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Overall Statistics")
            st.markdown(f"""
            - **Total Records:** {summary['total_records']:,}
            - **FTE Records:** {summary['fte_records']:,}
            - **Internship Records:** {summary['internship_records']:,}
            - **Unique Companies:** {summary['unique_companies']:,}
            - **Years Covered:** {', '.join(map(str, summary['years_covered']))}
            """)

            st.subheader("💰 FTE CTC Statistics")
            st.markdown(f"""
            - **Mean CTC:** ₹{summary['fte_statistics']['mean_ctc']:.2f} LPA
            - **Median CTC:** ₹{summary['fte_statistics']['median_ctc']:.2f} LPA
            - **Min CTC:** ₹{summary['fte_statistics']['min_ctc']:.2f} LPA
            - **Max CTC:** ₹{summary['fte_statistics']['max_ctc']:.2f} LPA
            - **Std Dev:** ₹{summary['fte_statistics']['std_ctc']:.2f} LPA
            """)

        with col2:
            st.subheader("📈 Data Completeness")
            st.markdown(f"""
            - **CTC Data:** {summary['data_completeness']['ctc_completeness']:.1f}%
            - **Base Salary Data:** {summary['data_completeness']['base_completeness']:.1f}%
            - **CGPA Data:** {summary['data_completeness']['cgpa_completeness']:.1f}%
            """)

            st.subheader("🏆 Top 10 Recruiters")
            for i, (company, count) in enumerate(summary['top_10_companies'].items(), 1):
                st.markdown(f"{i}. **{company}** - {count} placements")

    # Generate insights from filtered data
    st.markdown("---")
    st.subheader("📌 Insights from Filtered Data")

    insights = []

    if len(df_fte_filtered) > 0:
        avg_ctc = df_fte_filtered['total_ctc'].mean()
        max_ctc = df_fte_filtered['total_ctc'].max()
        top_company = df_fte_filtered.groupby('company_name')['total_ctc'].mean().idxmax()

        insights.append(f"📊 Average FTE CTC: **₹{avg_ctc:.2f} LPA**")
        insights.append(f"🏆 Highest CTC offered: **₹{max_ctc:.2f} LPA**")
        insights.append(f"⭐ Top paying company (avg): **{top_company}**")

        # Top tier analysis
        if 'tier' in df_fte_filtered.columns:
            tier_avg = df_fte_filtered.groupby('tier')['total_ctc'].mean().sort_values(ascending=False)
            if len(tier_avg) > 0:
                top_tier = tier_avg.index[0]
                insights.append(f"📈 Highest paying tier: **{top_tier}** (avg ₹{tier_avg.iloc[0]:.2f} LPA)")

    for insight in insights:
        st.markdown(f"- {insight}")

# ----------------------------------------------------------------------------
# TAB 5: DATA EXPLORER
# ----------------------------------------------------------------------------

with tab5:
    st.header("📂 Data Explorer")

    st.markdown("Explore the raw placement data with interactive filters and sorting.")

    # Display options
    col1, col2 = st.columns(2)

    available_cols = ['batch_year', 'company_name', 'job_role', 'tier',
                     'total_ctc', 'base_salary', 'num_fte', 'num_intern',
                     'cgpa_cutoff', 'is_internship']

    with col1:
        show_columns = st.multiselect(
            "Select Columns to Display",
            options=available_cols,
            default=['batch_year', 'company_name', 'job_role', 'total_ctc', 'tier', 'cgpa_cutoff']
        )

    with col2:
        sort_by = st.selectbox("Sort By", options=show_columns if show_columns else available_cols)
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
            file_name=f"placement_data_filtered_{len(display_df)}_records.csv",
            mime="text/csv"
        )

# ----------------------------------------------------------------------------
# TAB 6: TEMPORAL ANALYSIS & PREDICTIONS
# ----------------------------------------------------------------------------

with tab6:
    st.header("🔮 Temporal Analysis & Predictions")

    if temporal_data is None:
        st.warning("⚠️ Temporal analysis not found. Run `python temporal_analysis.py` to generate insights.")
        if st.button("Generate Temporal Analysis Now"):
            with st.spinner("Running temporal analysis..."):
                from temporal_analysis import save_temporal_analysis
                temporal_data = save_temporal_analysis()
                st.success("✅ Analysis complete! Refresh the page to see results.")
                st.experimental_rerun()
    else:
        # Key Insights Summary
        st.subheader("🎯 Key Insights")
        if 'key_insights' in temporal_data and temporal_data['key_insights']:
            for insight in temporal_data['key_insights']:
                st.info(f"💡 {insight}")

        st.markdown("---")

        # Yearly Statistics
        st.subheader("📊 Yearly Statistics & Trends")

        col1, col2 = st.columns(2)

        with col1:
            # CTC Trends
            if 'yearly_statistics' in temporal_data:
                yearly_df = pd.DataFrame(temporal_data['yearly_statistics'])

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=yearly_df['year'],
                    y=yearly_df['mean_ctc'],
                    mode='lines+markers',
                    name='Mean CTC',
                    line=dict(color='blue', width=3),
                    marker=dict(size=10)
                ))
                fig.add_trace(go.Scatter(
                    x=yearly_df['year'],
                    y=yearly_df['median_ctc'],
                    mode='lines+markers',
                    name='Median CTC',
                    line=dict(color='green', width=3),
                    marker=dict(size=10)
                ))

                # Add forecast if available
                if 'ctc_forecast' in temporal_data and 'predictions' in temporal_data['ctc_forecast']:
                    forecast = temporal_data['ctc_forecast']
                    pred_years = [p['year'] for p in forecast['predictions']]
                    pred_mean = [p['predicted_mean_ctc'] for p in forecast['predictions']]
                    pred_median = [p['predicted_median_ctc'] for p in forecast['predictions']]

                    # Combine historical with forecast
                    all_years = list(yearly_df['year']) + pred_years
                    hist_mean = list(yearly_df['mean_ctc']) + [None] * len(pred_years)
                    hist_median = list(yearly_df['median_ctc']) + [None] * len(pred_years)
                    forecast_mean = [None] * len(yearly_df) + pred_mean
                    forecast_median = [None] * len(yearly_df) + pred_median

                    fig.add_trace(go.Scatter(
                        x=all_years,
                        y=forecast_mean,
                        mode='lines+markers',
                        name='Predicted Mean',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=8, symbol='diamond')
                    ))
                    fig.add_trace(go.Scatter(
                        x=all_years,
                        y=forecast_median,
                        mode='lines+markers',
                        name='Predicted Median',
                        line=dict(color='orange', width=2, dash='dash'),
                        marker=dict(size=8, symbol='diamond')
                    ))

                    # Add confidence interval if available
                    if 'confidence_interval_mean' in forecast['predictions'][0]:
                        lower_bounds = [None] * len(yearly_df) + [p['confidence_interval_mean']['lower'] for p in forecast['predictions']]
                        upper_bounds = [None] * len(yearly_df) + [p['confidence_interval_mean']['upper'] for p in forecast['predictions']]

                        fig.add_trace(go.Scatter(
                            x=all_years + all_years[::-1],
                            y=upper_bounds + lower_bounds[::-1],
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.1)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='95% Confidence',
                            showlegend=True
                        ))

                fig.update_layout(
                    title='CTC Trends & Forecast',
                    xaxis_title='Year',
                    yaxis_title='CTC (LPA)',
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Placement Volume Trends
            if 'yearly_statistics' in temporal_data:
                yearly_df = pd.DataFrame(temporal_data['yearly_statistics'])

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=yearly_df['year'],
                    y=yearly_df['total_placements'],
                    name='Historical',
                    marker_color='steelblue'
                ))

                # Add forecast
                if 'placement_volume_forecast' in temporal_data and 'predictions' in temporal_data['placement_volume_forecast']:
                    forecast = temporal_data['placement_volume_forecast']
                    pred_years = [p['year'] for p in forecast['predictions']]
                    pred_placements = [p['predicted_placements'] for p in forecast['predictions']]

                    fig.add_trace(go.Bar(
                        x=pred_years,
                        y=pred_placements,
                        name='Predicted',
                        marker_color='lightcoral',
                        marker_pattern_shape="/"
                    ))

                fig.update_layout(
                    title='Placement Volume Trends & Forecast',
                    xaxis_title='Year',
                    yaxis_title='Number of Placements',
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Year-over-Year Growth
        st.subheader("📈 Year-over-Year Growth Analysis")

        col1, col2 = st.columns(2)

        with col1:
            if 'year_over_year_growth' in temporal_data:
                growth_df = pd.DataFrame(temporal_data['year_over_year_growth'])

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=growth_df['year'],
                    y=growth_df['mean_ctc_growth_%'],
                    name='Mean CTC Growth',
                    marker_color=['green' if x > 0 else 'red' for x in growth_df['mean_ctc_growth_%']]
                ))

                fig.update_layout(
                    title='Mean CTC Year-over-Year Growth (%)',
                    xaxis_title='Year',
                    yaxis_title='Growth (%)',
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'year_over_year_growth' in temporal_data:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=growth_df['year'],
                    y=growth_df['placement_growth_%'],
                    name='Placement Volume Growth',
                    marker_color=['green' if x > 0 else 'red' for x in growth_df['placement_growth_%']]
                ))

                fig.update_layout(
                    title='Placement Volume Year-over-Year Growth (%)',
                    xaxis_title='Year',
                    yaxis_title='Growth (%)',
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Tier Distribution Trends
        st.subheader("🎯 Tier Distribution Trends Over Time")

        if 'tier_distribution_trends' in temporal_data:
            tier_trends = temporal_data['tier_distribution_trends']

            # Prepare data for stacked area chart
            years = sorted([int(y) for y in tier_trends.keys()])
            tiers = ['Dream', 'Tier-1', 'Tier-2', 'Tier-3']

            fig = go.Figure()

            for tier in tiers:
                tier_values = [tier_trends[str(year)].get(tier, 0) for year in years]
                fig.add_trace(go.Scatter(
                    x=years,
                    y=tier_values,
                    mode='lines',
                    name=tier,
                    stackgroup='one',
                    fillcolor='rgba(0,0,0,0.1)'
                ))

            fig.update_layout(
                title='Tier Distribution Trends (Percentage)',
                xaxis_title='Year',
                yaxis_title='Percentage (%)',
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Company Hiring Patterns
        st.subheader("🏢 Top Company Hiring Patterns")

        if 'top_company_hiring_patterns' in temporal_data:
            company_patterns = pd.DataFrame(temporal_data['top_company_hiring_patterns'])

            # Interactive table
            st.dataframe(
                company_patterns[[c for c in company_patterns.columns if not c.startswith('hires_')]].head(15),
                use_container_width=True,
                height=400
            )

            # Heatmap of hiring patterns
            st.subheader("📊 Hiring Heatmap: Top 15 Companies Over Years")

            hire_cols = [c for c in company_patterns.columns if c.startswith('hires_')]
            if hire_cols:
                heatmap_data = company_patterns.head(15)[['company'] + hire_cols]
                heatmap_data = heatmap_data.set_index('company')
                heatmap_data.columns = [c.replace('hires_', '') for c in heatmap_data.columns]

                fig = px.imshow(
                    heatmap_data,
                    labels=dict(x="Year", y="Company", color="Hires"),
                    aspect="auto",
                    color_continuous_scale="Blues"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Emerging Companies
        st.subheader("🚀 Emerging Companies (Recently Started Hiring)")

        if 'emerging_companies' in temporal_data and temporal_data['emerging_companies']:
            emerging_df = pd.DataFrame(temporal_data['emerging_companies'])

            col1, col2 = st.columns([2, 1])

            with col1:
                st.dataframe(emerging_df, use_container_width=True, height=400)

            with col2:
                st.metric("Total Emerging Companies", len(temporal_data['emerging_companies']))

                if 'avg_ctc' in emerging_df.columns:
                    avg_emerging_ctc = emerging_df['avg_ctc'].mean()
                    st.metric("Avg CTC (Emerging)", f"₹{avg_emerging_ctc:.2f}L")

                tier_dist = emerging_df['tier'].value_counts()
                st.write("**Tier Distribution:**")
                for tier, count in tier_dist.items():
                    st.write(f"- {tier}: {count}")
        else:
            st.info("No emerging companies identified in the recent period.")

        st.markdown("---")

        # Volatility Analysis
        st.subheader("📉 CTC Volatility Analysis")

        if 'volatility_analysis' in temporal_data:
            vol = temporal_data['volatility_analysis']

            col1, col2, col3 = st.columns(3)

            with col1:
                if vol.get('overall_trend_stability'):
                    st.metric("Overall Market", vol['overall_trend_stability'].title())

            with col2:
                if vol.get('most_stable_year'):
                    st.metric("Most Stable Year", vol['most_stable_year'])

            with col3:
                if vol.get('most_volatile_year'):
                    st.metric("Most Volatile Year", vol['most_volatile_year'])

            # CV by year chart
            if 'coefficient_of_variation_by_year' in vol:
                cv_data = {k: v for k, v in vol['coefficient_of_variation_by_year'].items() if v is not None}

                if cv_data:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=list(cv_data.keys()),
                        y=list(cv_data.values()),
                        marker_color='purple'
                    ))

                    fig.update_layout(
                        title='Coefficient of Variation by Year (Lower = More Stable)',
                        xaxis_title='Year',
                        yaxis_title='CV (%)',
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Forecast Details
        st.subheader("🔮 Detailed Forecast Information")

        col1, col2 = st.columns(2)

        with col1:
            if 'ctc_forecast' in temporal_data:
                forecast = temporal_data['ctc_forecast']
                st.write("**CTC Forecast Model:**")
                st.write(f"- Model Type: {forecast.get('model_type', 'N/A')}")
                st.write(f"- R² Score (Mean): {forecast.get('r2_score_mean', 0):.4f}")
                st.write(f"- R² Score (Median): {forecast.get('r2_score_median', 0):.4f}")
                st.write(f"- Slope (Mean): ₹{forecast.get('slope_mean', 0):.2f} LPA/year")

                if 'predictions' in forecast:
                    st.write("\n**Predictions:**")
                    for pred in forecast['predictions']:
                        st.write(f"- **{pred['year']}:** ₹{pred['predicted_mean_ctc']:.2f}L (mean), ₹{pred['predicted_median_ctc']:.2f}L (median)")

        with col2:
            if 'placement_volume_forecast' in temporal_data:
                forecast = temporal_data['placement_volume_forecast']
                st.write("**Placement Volume Forecast:**")
                st.write(f"- Model Type: {forecast.get('model_type', 'N/A')}")
                st.write(f"- R² Score: {forecast.get('r2_score', 0):.4f}")
                st.write(f"- Trend: {forecast.get('trend', 'N/A').title()}")
                st.write(f"- Avg Yearly Change: {forecast.get('avg_yearly_change', 0):.1f} placements/year")

                if 'predictions' in forecast:
                    st.write("\n**Predictions:**")
                    for pred in forecast['predictions']:
                        st.write(f"- **{pred['year']}:** {pred['predicted_placements']} placements")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><b>PES University Placement Analytics</b> | Data: 2022-2026 Batches | {total_records} Records</p>
    <p>Built with Streamlit & Plotly | Powered by clean data pipeline</p>
</div>
""".format(total_records=len(df)), unsafe_allow_html=True)
