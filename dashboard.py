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
        background-color: #f0f2f6;
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

df = load_data()
summary = load_summary()

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

# College filter
colleges = sorted(df['college'].dropna().unique().tolist())
selected_colleges = st.sidebar.multiselect(
    "Select Colleges",
    options=colleges,
    default=colleges
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
    (df['college'].isin(selected_colleges)) &
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
    "📈 Overview", "🏢 Companies", "💰 Salary Analysis", "🎓 Cross-College", "🎯 Insights", "📂 Data Explorer"
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
# TAB 4: CROSS-COLLEGE COMPARISON
# ----------------------------------------------------------------------------

with tab4:
    st.header("🎓 Cross-College Comparison")

    # Only show if multiple colleges are in the data
    if df['college'].nunique() > 1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Average FTE CTC by College")
            if len(df_fte_filtered) > 0:
                college_avg = df_fte_filtered.groupby('college')['total_ctc'].agg(['mean', 'median', 'count']).reset_index()
                college_avg = college_avg.sort_values('mean', ascending=False)

                fig = px.bar(
                    college_avg,
                    x='college',
                    y='mean',
                    title='Average CTC by College',
                    labels={'college': 'College', 'mean': 'Average CTC (LPA)'},
                    color='mean',
                    color_continuous_scale='Viridis',
                    text='mean'
                )
                fig.update_traces(texttemplate='₹%{text:.2f}L', textposition='outside')
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                # Show stats table
                st.dataframe(
                    college_avg.rename(columns={
                        'college': 'College',
                        'mean': 'Avg CTC',
                        'median': 'Median CTC',
                        'count': 'Placements'
                    }).round(2),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No FTE data available")

        with col2:
            st.subheader("Placement Distribution by College")
            college_counts = df_filtered['college'].value_counts()

            fig = px.pie(
                values=college_counts.values,
                names=college_counts.index,
                title='Records by College',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # CTC Distribution Comparison
        st.subheader("CTC Distribution Comparison")
        if len(df_fte_filtered) > 0:
            fig = px.box(
                df_fte_filtered,
                x='college',
                y='total_ctc',
                title='CTC Distribution by College',
                labels={'college': 'College', 'total_ctc': 'CTC (LPA)'},
                color='college',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Top companies by college
        st.subheader("Top Recruiters by College")

        for college in sorted(df_filtered['college'].unique()):
            with st.expander(f"📚 {college}"):
                college_data = df_filtered[df_filtered['college'] == college]
                top_companies = college_data['company_name'].value_counts().head(10)

                st.markdown(f"**Total Records:** {len(college_data):,}")
                st.markdown(f"**Unique Companies:** {college_data['company_name'].nunique():,}")

                st.markdown("**Top 10 Recruiters:**")
                for i, (company, count) in enumerate(top_companies.items(), 1):
                    st.markdown(f"{i}. {company} - {count} placements")
    else:
        st.info("Cross-college comparison requires data from multiple colleges. Currently showing data from PES only.")

# ----------------------------------------------------------------------------
# TAB 5: INSIGHTS
# ----------------------------------------------------------------------------

with tab5:
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
# TAB 6: DATA EXPLORER
# ----------------------------------------------------------------------------

with tab6:
    st.header("📂 Data Explorer")

    st.markdown("Explore the raw placement data with interactive filters and sorting.")

    # Display options
    col1, col2 = st.columns(2)

    available_cols = ['batch_year', 'college', 'company_name', 'job_role', 'tier',
                     'total_ctc', 'base_salary', 'num_fte', 'num_intern',
                     'cgpa_cutoff', 'is_internship']

    with col1:
        show_columns = st.multiselect(
            "Select Columns to Display",
            options=available_cols,
            default=['batch_year', 'college', 'company_name', 'job_role', 'total_ctc', 'tier']
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
