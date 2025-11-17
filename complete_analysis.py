"""
Complete Analysis Pipeline for ADA Project
Temporal and Statistical Data Driven Insights into Talent Acquisition

This script performs comprehensive analysis including:
1. Comprehensive EDA
2. Temporal Trend Analysis
3. Cross-College Comparative Analysis
4. Statistical Testing
5. Predictive Modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
import os
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report, confusion_matrix, precision_recall_fscore_support
import datetime

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)

# Create output directories
Path('analysis_outputs').mkdir(exist_ok=True)
Path('analysis_outputs/eda').mkdir(exist_ok=True)
Path('analysis_outputs/temporal').mkdir(exist_ok=True)
Path('analysis_outputs/cross_college').mkdir(exist_ok=True)
Path('analysis_outputs/statistical').mkdir(exist_ok=True)
Path('analysis_outputs/predictive').mkdir(exist_ok=True)
Path('analysis_outputs/advanced').mkdir(exist_ok=True)

print("="*100)
print(" "*30 + "COMPLETE ANALYSIS PIPELINE")
print(" "*20 + "Temporal and Statistical Data Driven Insights")
print("="*100)

# ============================================================================
# PART 1: DATA LOADING AND PREPARATION
# ============================================================================
print("\n" + "="*100)
print("PART 1: DATA LOADING AND PREPARATION")
print("="*100)

df = pd.read_csv('processed_data/consolidated_placement_data.csv')
print(f"✓ Loaded {len(df):,} records")
print(f"✓ Years covered: {sorted(df['batch_year'].unique())}")
print(f"✓ Unique companies: {df['company_name'].nunique():,}")
print(f"✓ Unique colleges: {df['college'].nunique()}")

# Normalize placement tier labels to reduce fragmentation
tier_map = {
    'Tier 1': 'Tier-1', 'Tier1': 'Tier-1', 'Tier-1': 'Tier-1',
    'Tier 2': 'Tier-2', 'Tier2': 'Tier-2', 'Tier-2': 'Tier-2',
    'Tier 3': 'Tier-3', 'Tier3': 'Tier-3', 'Tier-3': 'Tier-3',
    'Dream Tier': 'Dream', 'Dream': 'Dream',
    'Super Dream': 'Super-Dream', 'Super-Dream': 'Super-Dream', 'Super Dream Tier': 'Super-Dream'
}
if 'placement_tier' in df.columns:
    df['placement_tier'] = df['placement_tier'].replace(tier_map)

# Derive internship flag & FTE CTC (if produced by consolidator this will already exist)
if 'is_internship_record' not in df.columns:
    df['is_internship_record'] = (
        df['placement_tier'].str.contains('Internship', case=False, na=False) |
        df.get('placement_type', pd.Series(index=df.index, dtype=str)).str.contains('Intern', case=False, na=False)
    )
if 'fte_ctc' not in df.columns:
    df['fte_ctc'] = np.where(~df['is_internship_record'], df['total_ctc'], np.nan)

# Fallback estimation for missing FTE CTC using base_salary heuristic (base * 1.25)
missing_fte_mask = (~df['is_internship_record']) & df['fte_ctc'].isna() & df['base_salary'].notna()
if missing_fte_mask.any():
    df.loc[missing_fte_mask, 'fte_ctc'] = df.loc[missing_fte_mask, 'base_salary'] * 1.25
    print(f"✓ Estimated FTE CTC for {missing_fte_mask.sum()} records using base_salary heuristic")

# Filter valid data (FTE only for compensation statistics)
df_ctc = df[(df['fte_ctc'].notna()) & (df['fte_ctc'] > 0) & (~df['is_internship_record'])].copy()
df_cgpa = df[df['cgpa_cutoff'].notna()].copy()
df_offers = df[df['num_offers_total'].notna() & (df['num_offers_total'] > 0)].copy()

print(f"✓ Records with CTC: {len(df_ctc):,} ({len(df_ctc)/len(df)*100:.1f}%)")
print(f"✓ Records with CGPA: {len(df_cgpa):,} ({len(df_cgpa)/len(df)*100:.1f}%)")
print(f"✓ Records with Offers: {len(df_offers):,} ({len(df_offers)/len(df)*100:.1f}%)")

# ============================================================================
# PART 2: COMPREHENSIVE EDA
# ============================================================================
print("\n" + "="*100)
print("PART 2: COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
print("="*100)

# Basic statistics
print("\n📊 Dataset Overview:")
print(f"  • Shape: {df.shape}")
print(f"  • Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"  • Date Range: {df['batch_year'].min()} - {df['batch_year'].max()}")

# Compensation statistics (FTE only)
if len(df_ctc) > 0:
    print("\n💰 Compensation Statistics (FTE CTC, LPA):")
    print(f"  • Mean FTE CTC: ₹{df_ctc['fte_ctc'].mean():.2f}")
    print(f"  • Median FTE CTC: ₹{df_ctc['fte_ctc'].median():.2f}")
    print(f"  • Std Dev: ₹{df_ctc['fte_ctc'].std():.2f}")
    print(f"  • Range: ₹{df_ctc['fte_ctc'].min():.2f} - ₹{df_ctc['fte_ctc'].max():.2f}")
    print(f"  • 90th Percentile: ₹{df_ctc['fte_ctc'].quantile(0.90):.2f}")

# Top companies
top_recruiters = df_offers.groupby('company_name')['num_offers_total'].sum().nlargest(10)
print("\n🏆 Top 10 Recruiters by Offers:")
for idx, (company, offers) in enumerate(top_recruiters.items(), 1):
    print(f"  {idx:2d}. {company:35s}: {offers:5.0f} offers")

# Visualizations
print("\n📈 Generating EDA visualizations...")

# 1. Year-wise distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
year_dist = df['batch_year'].value_counts().sort_index()
axes[0].bar(year_dist.index, year_dist.values, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Batch Year', fontsize=12)
axes[0].set_ylabel('Number of Records', fontsize=12)
axes[0].set_title('Placement Records by Year', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
for x, y in zip(year_dist.index, year_dist.values):
    axes[0].text(x, y+20, str(y), ha='center', fontweight='bold')

tier_dist = df['placement_tier'].value_counts()
axes[1].pie(tier_dist.values, labels=tier_dist.index, autopct='%1.1f%%', startangle=45)
axes[1].set_title('Placement Distribution by Tier', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('analysis_outputs/eda/year_tier_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. CTC distribution
if len(df_ctc) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].hist(df_ctc['fte_ctc'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('FTE CTC (LPA)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('FTE CTC Distribution', fontsize=13, fontweight='bold')
    axes[0, 0].axvline(df_ctc['fte_ctc'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ₹{df_ctc["fte_ctc"].mean():.2f}')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    yearly_ctc = df_ctc.groupby('batch_year')['fte_ctc'].agg(['mean', 'median'])
    axes[0, 1].plot(yearly_ctc.index, yearly_ctc['mean'], marker='o', linewidth=2, markersize=8, label='Mean', color='blue')
    axes[0, 1].plot(yearly_ctc.index, yearly_ctc['median'], marker='s', linewidth=2, markersize=8, label='Median', color='green')
    axes[0, 1].set_xlabel('Batch Year', fontsize=11)
    axes[0, 1].set_ylabel('FTE CTC (LPA)', fontsize=11)
    axes[0, 1].set_title('FTE CTC Trends Over Years', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    tier_ctc = df_ctc.groupby('placement_tier')['fte_ctc'].mean().sort_values()
    axes[1, 0].barh(tier_ctc.index, tier_ctc.values, color='purple', edgecolor='black')
    axes[1, 0].set_xlabel('Average FTE CTC (LPA)', fontsize=11)
    axes[1, 0].set_ylabel('Placement Tier', fontsize=11)
    axes[1, 0].set_title('Average FTE CTC by Tier', fontsize=13, fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)

    role_ctc = df_ctc.groupby('role_type')['fte_ctc'].mean().sort_values(ascending=False).head(10)
    axes[1, 1].barh(range(len(role_ctc)), role_ctc.values, color='orange', edgecolor='black')
    axes[1, 1].set_yticks(range(len(role_ctc)))
    axes[1, 1].set_yticklabels(role_ctc.index)
    axes[1, 1].set_xlabel('Average FTE CTC (LPA)', fontsize=11)
    axes[1, 1].set_ylabel('Role Type', fontsize=11)
    axes[1, 1].set_title('Top 10 Roles by Average FTE CTC', fontsize=13, fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('analysis_outputs/eda/ctc_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Top companies visualization
company_ctc = df_ctc.groupby('company_name').agg({'fte_ctc': ['mean', 'count']}).reset_index()
company_ctc.columns = ['company_name', 'avg_fte_ctc', 'count']
# Prefer companies with at least 2 records, but relax if fewer than 15
filtered_ctc = company_ctc[company_ctc['count'] >= 2]
if len(filtered_ctc) < 15:
    filtered_ctc = company_ctc[company_ctc['count'] >= 1]
top_paying = filtered_ctc.nlargest(15, 'avg_fte_ctc').sort_values('avg_fte_ctc')

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
top_recruiters.sort_values().plot(kind='barh', ax=axes[0], color='teal', edgecolor='black')
axes[0].set_xlabel('Number of Offers', fontsize=12)
axes[0].set_ylabel('Company Name', fontsize=12)
axes[0].set_title('Top 10 Recruiters by Offer Count', fontsize=14, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

axes[1].barh(range(len(top_paying)), top_paying['avg_fte_ctc'], color='gold', edgecolor='black')
axes[1].set_yticks(range(len(top_paying)))
axes[1].set_yticklabels(top_paying['company_name'])
axes[1].set_xlabel('Average FTE CTC (LPA)', fontsize=12)
axes[1].set_ylabel('Company Name', fontsize=12)
axes[1].set_title('Top 15 Highest Paying Companies', fontsize=14, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('analysis_outputs/eda/top_companies.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Saved: year_tier_distribution.png")
print("  ✓ Saved: ctc_comprehensive_analysis.png")
print("  ✓ Saved: top_companies.png")

# --------------------------------------------------------------------------
# ADVANCED / ADDITIONAL VISUALIZATIONS (EYE-CATCHING)
# --------------------------------------------------------------------------
print("\n📊 Generating advanced visualizations...")

# 1. Compensation breakdown donut (aggregate proportions among FTE rows)
if len(df_ctc) > 0:
    comp_components = {
        'Base Salary': df_ctc['base_salary'].fillna(0),
        'Joining Bonus': df_ctc['joining_bonus'].fillna(0),
        'Stocks/ESOPs': df_ctc['stocks_esops'].fillna(0),
        'Other Components': df_ctc['fte_ctc'] - (
            df_ctc['base_salary'].fillna(0) +
            df_ctc['joining_bonus'].fillna(0) +
            df_ctc['stocks_esops'].fillna(0)
        )
    }
    comp_sums = {k: v[v>0].sum() for k, v in comp_components.items()}
    labels = list(comp_sums.keys())
    sizes = list(comp_sums.values())
    if sum(sizes) > 0:
        plt.figure(figsize=(8,8))
        wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=40,
                                           textprops={'fontsize':11})
        plt.title('Aggregate Compensation Component Proportions (FTE)', fontsize=15, fontweight='bold')
        centre_circle = plt.Circle((0,0), 0.55, fc='white')
        fig = plt.gcf(); fig.gca().add_artist(centre_circle)
        plt.tight_layout()
        plt.savefig('analysis_outputs/advanced/comp_breakdown_donut.png', dpi=300)
        plt.close()
        print("  ✓ Saved: comp_breakdown_donut.png")

# 2. Offer vs Avg FTE CTC bubble chart
if len(df_ctc) > 0 and 'num_offers_total' in df.columns:
    company_offer = df.groupby('company_name')['num_offers_total'].sum().reset_index()
    merged = company_offer.merge(company_ctc[['company_name','avg_fte_ctc','count']], on='company_name', how='left')
    merged = merged[merged['avg_fte_ctc'].notna()]
    top_subset = merged.nlargest(30, 'num_offers_total')
    if len(top_subset) > 5:
        plt.figure(figsize=(12,7))
        scatter = plt.scatter(top_subset['num_offers_total'], top_subset['avg_fte_ctc'],
                              s=top_subset['num_offers_total']*4, alpha=0.6, c=top_subset['avg_fte_ctc'], cmap='viridis', edgecolor='black')
        plt.colorbar(scatter, label='Avg FTE CTC (LPA)')
        plt.xlabel('Total Offers', fontsize=12); plt.ylabel('Average FTE CTC (LPA)', fontsize=12)
        plt.title('Offers vs Average FTE CTC (Bubble Size = Offers)', fontsize=15, fontweight='bold')
        # Annotate top 8 by offers
        for _, row in top_subset.nlargest(8, 'num_offers_total').iterrows():
            plt.annotate(row['company_name'][:18], (row['num_offers_total'], row['avg_fte_ctc']),
                         textcoords='offset points', xytext=(5,5), fontsize=9, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('analysis_outputs/advanced/offers_vs_ctc_bubble.png', dpi=300)
        plt.close()
        print("  ✓ Saved: offers_vs_ctc_bubble.png")

# 3. Role vs Tier heatmap (average FTE CTC)
if len(df_ctc) > 0:
    pivot_rt = df_ctc.pivot_table(values='fte_ctc', index='role_type', columns='placement_tier', aggfunc='mean')
    if pivot_rt.shape[1] > 0 and pivot_rt.shape[0] > 0:
        plt.figure(figsize=(14,8))
        sns.heatmap(pivot_rt, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=0.5, cbar_kws={'label':'Avg FTE CTC'})
        plt.title('Average FTE CTC by Role and Tier', fontsize=16, fontweight='bold')
        plt.xlabel('Placement Tier'); plt.ylabel('Role Type')
        plt.tight_layout()
        plt.savefig('analysis_outputs/advanced/role_tier_heatmap.png', dpi=300)
        plt.close()
        print("  ✓ Saved: role_tier_heatmap.png")

# 4. Tier stacked area over years (normalized major tiers)
major_tiers = ['Tier-1','Tier-2','Tier-3','Dream','Super-Dream']
tier_yearly_full = pd.crosstab(df['batch_year'], df['placement_tier'], normalize='index') * 100
available_major = [t for t in major_tiers if t in tier_yearly_full.columns]
if len(available_major) >= 2:
    plt.figure(figsize=(12,6))
    tier_yearly_full[available_major].plot.area(ax=plt.gca(), alpha=0.6)
    plt.ylabel('Percentage (%)'); plt.xlabel('Batch Year')
    plt.title('Tier Mix Evolution (Normalized %)', fontsize=15, fontweight='bold')
    plt.legend(title='Tier', loc='upper right')
    plt.tight_layout()
    plt.savefig('analysis_outputs/advanced/tier_stacked_area.png', dpi=300)
    plt.close()
    print("  ✓ Saved: tier_stacked_area.png")

# 5. Pareto chart of offers distribution
if 'num_offers_total' in df.columns and len(df_offers) > 0:
    offers_agg = df_offers.groupby('company_name')['num_offers_total'].sum().sort_values(ascending=False)
    cum_pct = offers_agg.cumsum() / offers_agg.sum() * 100
    top_n = min(50, len(offers_agg))
    plt.figure(figsize=(14,6))
    offers_agg.head(top_n).plot(kind='bar', color='steelblue', edgecolor='black')
    plt.twinx()
    plt.plot(range(top_n), cum_pct.head(top_n), color='red', marker='o')
    plt.ylabel('Cumulative % of Offers', color='red')
    plt.title('Pareto Chart - Offer Concentration (Top Companies)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('analysis_outputs/advanced/offer_pareto.png', dpi=300)
    plt.close()
    print("  ✓ Saved: offer_pareto.png")

# 6. CGPA vs Offers scatter (if enough data)
if 'cgpa_cutoff' in df.columns and 'num_offers_total' in df.columns:
    df_cgpa_offers = df[(df['cgpa_cutoff'].notna()) & (df['num_offers_total'].notna())]
    if len(df_cgpa_offers) > 40:
        plt.figure(figsize=(10,6))
        sns.scatterplot(data=df_cgpa_offers, x='cgpa_cutoff', y='num_offers_total', alpha=0.6)
        sns.regplot(data=df_cgpa_offers, x='cgpa_cutoff', y='num_offers_total', scatter=False, color='darkred')
        plt.title('CGPA Cutoff vs Number of Offers', fontsize=15, fontweight='bold')
        plt.xlabel('CGPA Cutoff'); plt.ylabel('Number of Offers')
        plt.tight_layout()
        plt.savefig('analysis_outputs/advanced/cgpa_vs_offers.png', dpi=300)
        plt.close()
        print("  ✓ Saved: cgpa_vs_offers.png")


# ============================================================================
# PART 3: TEMPORAL TREND ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("PART 3: TEMPORAL TREND ANALYSIS")
print("="*100)

# Year-over-year growth analysis
yearly_stats = df.groupby('batch_year').agg({
    'company_name': ['count', lambda x: x.nunique()],
    'is_internship_record': ['sum']
}).reset_index()
yearly_stats.columns = ['batch_year', 'total_records', 'unique_companies', 'internship_records']
yearly_stats = yearly_stats.set_index('batch_year')

# Add FTE CTC statistics (exclude years with fewer than 5 FTE entries to reduce noise)
ctc_stats_raw = df_ctc.groupby('batch_year')['fte_ctc']
ctc_stats = ctc_stats_raw.agg(['mean', 'median', 'std', 'count'])
# Lower threshold to 3 to surface more years with estimated FTE data
ctc_stats = ctc_stats[ctc_stats['count'] >= 3].drop(columns=['count'])
yearly_stats = yearly_stats.join(ctc_stats, how='left')

# Calculate growth rates (using only years with valid mean values)
yearly_stats['record_growth_%'] = yearly_stats['total_records'].pct_change() * 100
valid_mean = yearly_stats['mean']
yearly_stats['ctc_growth_%'] = valid_mean.pct_change() * 100

print("\n📈 Year-over-Year Growth Analysis:")
print(yearly_stats.round(2))

# Filtered growth excluding incomplete or low-density years (<50 total records or <5 valid FTE CTC entries)
filtered_years = yearly_stats[(yearly_stats['total_records'] >= 50) & (yearly_stats['mean'].notna())]
filtered_growth = None
if len(filtered_years) >= 2:
    filtered_growth = ((filtered_years.iloc[-1]['mean'] - filtered_years.iloc[0]['mean']) / filtered_years.iloc[0]['mean']) * 100
    print(f"\n🔍 Filtered FTE CTC Growth (quality years {filtered_years.index[0]}→{filtered_years.index[-1]}): {filtered_growth:+.2f}%")

# Overall growth
if len(yearly_stats) >= 2:
    total_growth = ((yearly_stats.iloc[-1]['total_records'] - yearly_stats.iloc[0]['total_records']) /
                    yearly_stats.iloc[0]['total_records'] * 100)
    valid_ctc_years = yearly_stats['mean'].dropna()
    if len(valid_ctc_years) >= 2:
        ctc_growth = ((valid_ctc_years.iloc[-1] - valid_ctc_years.iloc[0]) / valid_ctc_years.iloc[0] * 100)
    else:
        ctc_growth = float('nan')
    
    print(f"\n🎯 Overall Growth ({yearly_stats.index[0]} to {yearly_stats.index[-1]}):")
    print(f"  • Total Records: {total_growth:+.1f}%")
    print(f"  • Average FTE CTC: {ctc_growth:+.1f}%" if not np.isnan(ctc_growth) else "  • Average FTE CTC: N/A (insufficient data)")
    if filtered_growth is not None:
        print(f"  • Filtered FTE CTC Growth: {filtered_growth:+.2f}%")

# Visualizations
print("\n📊 Generating temporal visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Total records trend
axes[0, 0].plot(yearly_stats.index, yearly_stats['total_records'], 
                marker='o', linewidth=2.5, markersize=10, color='blue')
axes[0, 0].fill_between(yearly_stats.index, yearly_stats['total_records'], alpha=0.3, color='blue')
axes[0, 0].set_xlabel('Year', fontsize=11)
axes[0, 0].set_ylabel('Number of Records', fontsize=11)
axes[0, 0].set_title('Total Placement Records Over Years', fontsize=13, fontweight='bold')
axes[0, 0].grid(alpha=0.3)

# CTC trends
axes[0, 1].plot(yearly_stats.index, yearly_stats['mean'], 
                marker='D', linewidth=2.5, markersize=10, color='red', label='Mean')
axes[0, 1].plot(yearly_stats.index, yearly_stats['median'], 
                marker='o', linewidth=2.5, markersize=10, color='orange', label='Median')
axes[0, 1].set_xlabel('Year', fontsize=11)
axes[0, 1].set_ylabel('CTC (LPA)', fontsize=11)
axes[0, 1].set_title('CTC Trends Over Years', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Tier evolution
tier_yearly = pd.crosstab(df['batch_year'], df['placement_tier'], normalize='index') * 100
major_tiers = ['Tier-1', 'Tier-2', 'Tier-3', 'Dream']
for tier in major_tiers:
    if tier in tier_yearly.columns:
        axes[1, 0].plot(tier_yearly.index, tier_yearly[tier], 
                       marker='o', linewidth=2, markersize=8, label=tier)
axes[1, 0].set_xlabel('Year', fontsize=11)
axes[1, 0].set_ylabel('Percentage (%)', fontsize=11)
axes[1, 0].set_title('Tier Distribution Evolution', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Role evolution
top_roles = df['role_type'].value_counts().head(5).index
role_yearly = df[df['role_type'].isin(top_roles)].groupby(['batch_year', 'role_type']).size().unstack(fill_value=0)
for role in role_yearly.columns:
    axes[1, 1].plot(role_yearly.index, role_yearly[role], 
                   marker='o', linewidth=2, markersize=7, label=role)
axes[1, 1].set_xlabel('Year', fontsize=11)
axes[1, 1].set_ylabel('Number of Positions', fontsize=11)
axes[1, 1].set_title('Top 5 Role Types Evolution', fontsize=13, fontweight='bold')
axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('analysis_outputs/temporal/comprehensive_temporal_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Saved: comprehensive_temporal_analysis.png")

# Additional temporal plots directory
Path('analysis_outputs/temporal/advanced').mkdir(exist_ok=True)

print("  ↪ Generating extended temporal plots...")

# 1. Stacked bar: FTE vs Internship counts per year
year_counts = df.groupby('batch_year')['is_internship_record'].agg(['count','sum'])
year_counts['fte_count'] = year_counts['count'] - year_counts['sum']
plt.figure(figsize=(10,6))
plt.bar(year_counts.index, year_counts['fte_count'], label='FTE', color='steelblue')
plt.bar(year_counts.index, year_counts['sum'], bottom=year_counts['fte_count'], label='Internship', color='orange')
plt.xlabel('Batch Year'); plt.ylabel('Record Count')
plt.title('FTE vs Internship Record Composition by Year', fontsize=14, fontweight='bold')
plt.legend(); plt.tight_layout(); plt.savefig('analysis_outputs/temporal/advanced/fte_vs_internship_counts.png', dpi=300); plt.close()

# 2. Average CGPA cutoff trend (for records with cgpa_cutoff)
cgpa_year = df[df['cgpa_cutoff'].notna()].groupby('batch_year')['cgpa_cutoff'].mean()
if len(cgpa_year) > 0:
    plt.figure(figsize=(10,5))
    plt.plot(cgpa_year.index, cgpa_year.values, marker='o', linewidth=2.5, color='purple')
    plt.title('Average CGPA Cutoff Trend', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Year'); plt.ylabel('Average CGPA Cutoff')
    for x,y in zip(cgpa_year.index, cgpa_year.values):
        plt.text(x, y+0.03, f"{y:.2f}", ha='center', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig('analysis_outputs/temporal/advanced/cgpa_cutoff_trend.png', dpi=300); plt.close()

# 3. Total offers trend (combine all offer columns + estimate for years with missing data)
if 'num_offers_total' in df.columns:
    # Create unified offers column combining num_offers_total, num_offers_fte, num_offers_both
    df_offers_calc = df.copy()
    df_offers_calc['offers_unified'] = df_offers_calc[['num_offers_total','num_offers_fte','num_offers_both']].sum(axis=1, min_count=1)
    offers_year = df_offers_calc.groupby('batch_year')['offers_unified'].sum()
    
    # For years with 0 or very low counts, estimate by counting records (as proxy for placement activity)
    # This gives a visual representation even when explicit offer counts are missing
    record_counts = df.groupby('batch_year').size()
    offers_year_filled = offers_year.copy()
    for year in record_counts.index:
        if year not in offers_year_filled.index or offers_year_filled[year] < 10:
            # Use record count as estimated activity level (scaled down for visualization)
            offers_year_filled[year] = record_counts[year]
    
    if offers_year_filled.sum() > 0:
        plt.figure(figsize=(12,6))
        colors = ['steelblue' if y in offers_year.index and offers_year[y] >= 10 else 'lightcoral' 
                  for y in offers_year_filled.index]
        bars = plt.bar(offers_year_filled.index, offers_year_filled.values, color=colors, edgecolor='black', alpha=0.8)
        plt.title('Placement Activity by Year\n(Blue=Reported Offers, Red=Estimated from Records)', fontsize=14, fontweight='bold')
        plt.xlabel('Batch Year'); plt.ylabel('Count')
        for x,y in zip(offers_year_filled.index, offers_year_filled.values):
            if y > 0:
                plt.text(x, y+20, f"{int(y)}", ha='center', fontsize=9, fontweight='bold')
        plt.tight_layout(); plt.savefig('analysis_outputs/temporal/advanced/offers_trend.png', dpi=300); plt.close()

# 4. Role diversity trend (# unique roles per year)
if 'role_type' in df.columns:
    role_div = df.groupby('batch_year')['role_type'].nunique()
    plt.figure(figsize=(10,5))
    plt.plot(role_div.index, role_div.values, marker='s', linewidth=2.5, color='darkgreen')
    plt.title('Role Diversity (Unique Role Types) by Year', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Year'); plt.ylabel('Unique Roles')
    for x,y in zip(role_div.index, role_div.values):
        plt.text(x, y+0.3, f"{int(y)}", ha='center', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig('analysis_outputs/temporal/advanced/role_diversity_trend.png', dpi=300); plt.close()

# 5. FTE CTC boxplot per year (lower threshold to show more years with fallback estimates)
ctc_box = df_ctc.groupby('batch_year')['fte_ctc'].count()
valid_years_box = ctc_box[ctc_box >= 2].index  # Lower from 3 to 2
if len(valid_years_box) > 0:
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df_ctc[df_ctc['batch_year'].isin(valid_years_box)], x='batch_year', y='fte_ctc')
    plt.title('FTE CTC Distribution by Year', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Year'); plt.ylabel('FTE CTC (LPA)')
    plt.tight_layout(); plt.savefig('analysis_outputs/temporal/advanced/fte_ctc_yearly_boxplot.png', dpi=300); plt.close()

# 6. Role presence heatmap (Top 10 roles by frequency across years)
if 'role_type' in df.columns:
    top_roles_global = df['role_type'].value_counts().head(10).index
    role_presence = df[df['role_type'].isin(top_roles_global)].groupby(['batch_year','role_type']).size().unstack(fill_value=0)
    if role_presence.shape[0] > 0:
        role_presence_norm = role_presence.div(role_presence.sum(axis=1), axis=0) * 100
        plt.figure(figsize=(12,7))
        sns.heatmap(role_presence_norm, annot=True, fmt='.1f', cmap='magma', linewidths=0.5, cbar_kws={'label':'Percentage of Top Role Mix'})
        plt.title('Top Role Mix Percentage by Year', fontsize=15, fontweight='bold')
        plt.xlabel('Role Type'); plt.ylabel('Batch Year')
        plt.tight_layout(); plt.savefig('analysis_outputs/temporal/advanced/role_presence_heatmap.png', dpi=300); plt.close()


# ============================================================================
# PART 4: CROSS-COLLEGE COMPARATIVE ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("PART 4: CROSS-COLLEGE COMPARATIVE ANALYSIS")
print("="*100)

colleges = df['college'].unique()
print(f"\n🏫 Colleges in dataset: {colleges}")

if len(colleges) > 1:
    college_stats = df_ctc.groupby('college')['fte_ctc'].agg(['count', 'mean', 'median', 'std']).round(2)
    print("\n📊 College-wise CTC Statistics:")
    print(college_stats)
    
    # Visualizations
    print("\n📈 Generating cross-college visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # CTC comparison
    college_ctc = df_ctc.groupby('college')['fte_ctc'].mean().sort_values()
    college_ctc.plot(kind='barh', ax=axes[0, 0], color='purple', edgecolor='black')
    axes[0, 0].set_xlabel('Average CTC (LPA)', fontsize=12)
    axes[0, 0].set_ylabel('College', fontsize=12)
    axes[0, 0].set_title('Average CTC by College', fontsize=14, fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Box plot comparison
    sns.boxplot(data=df_ctc, x='college', y='fte_ctc', ax=axes[0, 1], palette='Set2')
    axes[0, 1].set_xlabel('College', fontsize=12)
    axes[0, 1].set_ylabel('FTE CTC (LPA)', fontsize=12)
    axes[0, 1].set_title('CTC Distribution by College', fontsize=14, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Placement count comparison
    college_count = df.groupby('college').size().sort_values()
    college_count.plot(kind='barh', ax=axes[1, 0], color='teal', edgecolor='black')
    axes[1, 0].set_xlabel('Number of Records', fontsize=12)
    axes[1, 0].set_ylabel('College', fontsize=12)
    axes[1, 0].set_title('Total Placement Records by College', fontsize=14, fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Tier distribution by college
    tier_college = pd.crosstab(df['college'], df['placement_tier'], normalize='index') * 100
    tier_college.plot(kind='bar', stacked=True, ax=axes[1, 1], colormap='tab10', edgecolor='black')
    axes[1, 1].set_xlabel('College', fontsize=12)
    axes[1, 1].set_ylabel('Percentage (%)', fontsize=12)
    axes[1, 1].set_title('Tier Distribution by College', fontsize=14, fontweight='bold')
    axes[1, 1].legend(title='Tier', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[1, 1].set_xticklabels(tier_college.index, rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_outputs/cross_college/college_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: college_comparison.png")
else:
    print("  ⚠ Only one college in dataset, skipping comparative analysis")

# ============================================================================
# PART 5: STATISTICAL ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("PART 5: STATISTICAL ANALYSIS & HYPOTHESIS TESTING")
print("="*100)

# Correlation analysis
print("\n🔍 Correlation Analysis:")
numeric_cols = ['batch_year', 'fte_ctc', 'cgpa_cutoff', 'num_offers_total', 
                'has_internship', 'has_stocks', 'has_joining_bonus']
available_cols = [col for col in numeric_cols if col in df.columns]
df_corr = df[available_cols].corr()

print(df_corr.round(3))

# Visualize correlation
plt.figure(figsize=(12, 10))
sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Numeric Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('analysis_outputs/statistical/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# CGPA vs CTC relationship
df_both = df[(df['cgpa_cutoff'].notna()) & (df['fte_ctc'].notna()) & (df['fte_ctc'] > 0) & (~df['is_internship_record'])]
if len(df_both) > 20:
    corr_cgpa_ctc = df_both['cgpa_cutoff'].corr(df_both['fte_ctc'])
    print(f"\n📊 CGPA vs CTC Correlation: {corr_cgpa_ctc:.3f}")
    
    plt.figure(figsize=(12, 6))
    plt.scatter(df_both['cgpa_cutoff'], df_both['fte_ctc'], alpha=0.5, color='blue')
    plt.xlabel('CGPA Cutoff', fontsize=12)
    plt.ylabel('FTE CTC (LPA)', fontsize=12)
    plt.title(f'CGPA Cutoff vs CTC (Correlation: {corr_cgpa_ctc:.3f})', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('analysis_outputs/statistical/cgpa_vs_ctc.png', dpi=300, bbox_inches='tight')
    plt.close()

# Statistical tests
if len(colleges) > 1:
    print("\n🧪 Statistical Tests:")
    college_groups = [df_ctc[df_ctc['college'] == college]['fte_ctc'].values 
                     for college in colleges if len(df_ctc[df_ctc['college'] == college]) > 0]
    
    if len(college_groups) >= 2:
        # ANOVA test
        f_stat, p_value = stats.f_oneway(*college_groups)
        print(f"  • ANOVA Test (CTC across colleges):")
        print(f"    F-statistic: {f_stat:.4f}")
        print(f"    P-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"    Result: Significant difference in CTC across colleges (p < 0.05)")
        else:
            print(f"    Result: No significant difference in CTC across colleges (p >= 0.05)")

print("  ✓ Saved: correlation_matrix.png")
if len(df_both) > 20:
    print("  ✓ Saved: cgpa_vs_ctc.png")

# ============================================================================
# PART 6: PREDICTIVE MODELING
# ============================================================================
print("\n" + "="*100)
print("PART 6: PREDICTIVE MODELING")
print("="*100)

# Prepare data for modeling
print("\n🤖 Building CTC Prediction Models...")

# Filter and prepare features
df_model = df_ctc[df_ctc['cgpa_cutoff'].notna()].copy()

if len(df_model) > 50:
    # Encode categorical variables
    le_company = LabelEncoder()
    le_role = LabelEncoder()
    le_tier = LabelEncoder()
    
    df_model['company_encoded'] = le_company.fit_transform(df_model['company_name'])
    df_model['role_type_encoded'] = le_role.fit_transform(df_model['role_type'])
    df_model['tier_encoded'] = le_tier.fit_transform(df_model['placement_tier'])
    
    # Feature engineering enhancements
    df_model['log_fte_ctc'] = np.log1p(df_model['fte_ctc'])
    df_model['is_high_tier'] = (df_model['placement_tier'].isin(['Dream','Super-Dream','Tier-1'])).astype(int)
    df_model['has_comp_breakdown'] = ((df_model['base_salary'].notna()) | (df_model['stocks_esops'].notna())).astype(int)

    features = ['batch_year', 'cgpa_cutoff', 'company_encoded', 'role_type_encoded',
                'tier_encoded', 'has_internship', 'has_stocks', 'has_joining_bonus',
                'is_high_tier', 'has_comp_breakdown']
    target = 'log_fte_ctc'
    
    # Remove rows with missing features
    df_model = df_model.dropna(subset=features + [target])
    
    X = df_model[features]
    y = df_model[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"  • Training set size: {len(X_train)}")
    print(f"  • Test set size: {len(X_test)}")
    
    # Train models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    }
    
    results = {}
    
    print("\n📊 Model Performance:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        # Convert error metrics back to original scale using inverse transform
        test_rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(test_pred)))
        test_mae = mean_absolute_error(np.expm1(y_test), np.expm1(test_pred))
        
        results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rmse': test_rmse,
            'mae': test_mae
        }
        
        print(f"\n  {name}:")
        print(f"    • Train R²: {train_r2:.4f}")
        print(f"    • Test R²: {test_r2:.4f}")
        print(f"    • Test RMSE: ₹{test_rmse:.2f} LPA")
        print(f"    • Test MAE: ₹{test_mae:.2f} LPA")
    
    # Feature importance (using Random Forest)
    rf_model = models['Random Forest']
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🎯 Feature Importance (Random Forest):")
    for idx, row in feature_importance.iterrows():
        print(f"  • {row['feature']:25s}: {row['importance']:.4f}")
    
    # Visualize predictions
    best_model = models['Random Forest']
    y_pred = best_model.predict(X_test)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.5, color='blue')
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual FTE CTC (LPA)', fontsize=12)
    axes[0].set_ylabel('Predicted FTE CTC (LPA)', fontsize=12)
    axes[0].set_title(f'Actual vs Predicted FTE CTC (R² = {results["Random Forest"]["test_r2"]:.3f})', 
                     fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Feature importance
    feature_importance.plot(kind='barh', x='feature', y='importance', ax=axes[1], 
                          color='green', edgecolor='black', legend=False)
    axes[1].set_xlabel('Importance Score', fontsize=12)
    axes[1].set_ylabel('Feature', fontsize=12)
    axes[1].set_title('Feature Importance', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_outputs/predictive/model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Model comparison
    plt.figure(figsize=(12, 6))
    model_names = list(results.keys())
    test_r2_scores = [results[m]['test_r2'] for m in model_names]
    rmse_scores = [results[m]['rmse'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    bars1 = ax1.bar(x - width/2, test_r2_scores, width, label='R² Score', color='skyblue', edgecolor='black')
    bars2 = ax2.bar(x + width/2, rmse_scores, width, label='RMSE (LPA)', color='lightcoral', edgecolor='black')
    
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12, color='blue')
    ax2.set_ylabel('RMSE (LPA)', fontsize=12, color='red')
    ax1.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=15)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    ax1.grid(alpha=0.3)
    
    fig.tight_layout()
    plt.savefig('analysis_outputs/predictive/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n  ✓ Saved: model_performance.png")
    print("  ✓ Saved: model_comparison.png")
    
else:
    print("  ⚠ Insufficient data for regression; attempting salary tier classification...")
    df_class = df[(df['fte_ctc'].notna()) & (~df['is_internship_record'])].copy()
    if 'salary_category' not in df_class.columns or df_class['salary_category'].isna().all():
        df_class['salary_category'] = pd.cut(
            df_class['fte_ctc'],
            bins=[0, 6, 12, 20, 60, float('inf')],
            labels=['Tier-3', 'Tier-2', 'Tier-1', 'Super-Dream', 'Dream']
        )
    df_class = df_class[df_class['salary_category'].notna()]
    if len(df_class) < 30:
        print("  ⚠ Still insufficient data for classification; modeling skipped.")
        modeling_mode = 'skipped'
    else:
        le_role = LabelEncoder(); le_tier = LabelEncoder(); le_company = LabelEncoder()
        df_class['company_encoded'] = le_company.fit_transform(df_class['company_name'])
        df_class['role_type_encoded'] = le_role.fit_transform(df_class['role_type'])
        df_class['tier_encoded'] = le_tier.fit_transform(df_class['placement_tier'].fillna('Unknown'))
        df_class['is_high_tier'] = (df_class['placement_tier'].isin(['Dream','Super-Dream','Tier-1'])).astype(int)
        features_cls = ['batch_year','company_encoded','role_type_encoded','tier_encoded','is_high_tier','has_internship','has_stocks','has_joining_bonus']
        Xc = df_class[features_cls]
        yc = df_class['salary_category']
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.25, random_state=42, stratify=yc)
        cls = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
        cls.fit(Xc_train, yc_train)
        yc_pred = cls.predict(Xc_test)
        print("\n📊 Salary Category Classification Performance:")
        print(classification_report(yc_test, yc_pred, digits=3))
        cm = confusion_matrix(yc_test, yc_pred, labels=cls.classes_)
        cm_df = pd.DataFrame(cm, index=cls.classes_, columns=cls.classes_)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Salary Category Classification')
        plt.xlabel('Predicted'); plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('analysis_outputs/predictive/salary_category_confusion_matrix.png', dpi=300)
        plt.close()
        print("  ✓ Saved: salary_category_confusion_matrix.png")
        # Performance metrics plot
        precisions, recalls, f1s, supports = precision_recall_fscore_support(yc_test, yc_pred, labels=cls.classes_)
        perf_df = pd.DataFrame({
            'Class': cls.classes_,
            'Precision': precisions,
            'Recall': recalls,
            'F1': f1s,
            'Support': supports
        })
        plt.figure(figsize=(10,6))
        bar_width = 0.25
        x = np.arange(len(perf_df))
        plt.bar(x - bar_width, perf_df['Precision'], width=bar_width, label='Precision', color='steelblue')
        plt.bar(x, perf_df['Recall'], width=bar_width, label='Recall', color='orange')
        plt.bar(x + bar_width, perf_df['F1'], width=bar_width, label='F1', color='green')
        plt.xticks(x, perf_df['Class'], rotation=15)
        plt.ylabel('Score')
        plt.title('Classification Performance by Class')
        plt.ylim(0, 1.05)
        plt.legend()
        plt.tight_layout()
        plt.savefig('analysis_outputs/predictive/model_performance.png', dpi=300)
        plt.close()
        print("  ✓ Saved: model_performance.png")
        # Feature importance for classifier
        fi_df = pd.DataFrame({'feature': features_cls, 'importance': cls.feature_importances_}).sort_values('importance', ascending=False)
        plt.figure(figsize=(10,6))
        plt.barh(fi_df['feature'], fi_df['importance'], color='purple', edgecolor='black')
        plt.xlabel('Importance')
        plt.title('Feature Importance - Classification Model')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('analysis_outputs/predictive/model_comparison.png', dpi=300)
        plt.close()
        print("  ✓ Saved: model_comparison.png")
        modeling_mode = 'classification'
        df_classification_rows = len(df_class)

# Set modeling_mode if regression ran
if 'modeling_mode' not in locals():
    modeling_mode = 'regression'
df_regression_rows = len(df_model)

# ============================================================================
# PART 7: KEY INSIGHTS & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*100)
print("PART 7: KEY INSIGHTS & RECOMMENDATIONS")
print("="*100)

insights = {
    'dataset_overview': {
        'total_records': len(df),
        'years_covered': f"{df['batch_year'].min()}-{df['batch_year'].max()}",
        'unique_companies': df['company_name'].nunique(),
        'unique_colleges': df['college'].nunique()
    },
    'compensation_fte': {
        'avg_fte_ctc': float(df_ctc['fte_ctc'].mean()) if len(df_ctc) > 0 else 0,
        'median_fte_ctc': float(df_ctc['fte_ctc'].median()) if len(df_ctc) > 0 else 0,
        'max_fte_ctc': float(df_ctc['fte_ctc'].max()) if len(df_ctc) > 0 else 0,
        'p90_fte_ctc': float(df_ctc['fte_ctc'].quantile(0.90)) if len(df_ctc) > 0 else 0
    },
    'top_performers': {
        'top_recruiter': top_recruiters.index[0] if len(top_recruiters) > 0 else 'N/A',
        'top_recruiter_offers': int(top_recruiters.iloc[0]) if len(top_recruiters) > 0 else 0,
        'highest_paying': top_paying.iloc[-1]['company_name'] if len(top_paying) > 0 else 'N/A',
        'highest_avg_fte_ctc': float(top_paying.iloc[-1]['avg_fte_ctc']) if len(top_paying) > 0 else 0
    },
    'trends': {
        'most_common_role': df['role_type'].value_counts().index[0],
        'avg_cgpa': float(df_cgpa['cgpa_cutoff'].mean()) if len(df_cgpa) > 0 else 0,
        'peak_year': int(year_dist.idxmax()) if len(year_dist) > 0 else 0
    },
    'temporal': {
        'overall_growth_%': float(total_growth) if 'total_growth' in locals() else 0,
        'fte_ctc_growth_%': float(ctc_growth) if 'ctc_growth' in locals() else 0,
        'filtered_fte_ctc_growth_%': float(filtered_growth) if filtered_growth is not None else None
    },
    'data_quality_flags': {
        'missing_fte_ctc_original': int(((~df['is_internship_record']) & df['total_ctc'].isna() & df['base_salary'].isna()).sum()),
        'fte_ctc_estimated_count': int(missing_fte_mask.sum()),
        'regression_rows': int(df_regression_rows),
        'classification_rows': int(df_classification_rows) if 'df_classification_rows' in locals() else 0,
        'modeling_mode': modeling_mode
    }
}

print("\n" + "="*80)
print(" "*25 + "KEY INSIGHTS SUMMARY")
print("="*80)

print(f"\n📊 DATASET OVERVIEW:")
print(f"  • Total Records: {insights['dataset_overview']['total_records']:,}")
print(f"  • Coverage Period: {insights['dataset_overview']['years_covered']}")
print(f"  • Unique Companies: {insights['dataset_overview']['unique_companies']:,}")
print(f"  • Unique Colleges: {insights['dataset_overview']['unique_colleges']}")

print(f"\n💰 FTE COMPENSATION INSIGHTS:")
print(f"  • Average FTE CTC: ₹{insights['compensation_fte']['avg_fte_ctc']:.2f} LPA")
print(f"  • Median FTE CTC: ₹{insights['compensation_fte']['median_fte_ctc']:.2f} LPA")
print(f"  • Highest FTE Package: ₹{insights['compensation_fte']['max_fte_ctc']:.2f} LPA")
print(f"  • 90th Percentile FTE: ₹{insights['compensation_fte']['p90_fte_ctc']:.2f} LPA")

print(f"\n🏆 TOP PERFORMERS:")
print(f"  • Top Recruiter: {insights['top_performers']['top_recruiter']} ({insights['top_performers']['top_recruiter_offers']} offers)")
print(f"  • Highest Paying: {insights['top_performers']['highest_paying']} (₹{insights['top_performers']['highest_avg_fte_ctc']:.2f} LPA)")

print(f"\n📈 TRENDS:")
print(f"  • Most Common Role: {insights['trends']['most_common_role']}")
print(f"  • Average CGPA Cutoff: {insights['trends']['avg_cgpa']:.2f}")
print(f"  • Peak Placement Year: {insights['trends']['peak_year']}")

if 'total_growth' in locals():
    print(f"\n🎯 TEMPORAL ANALYSIS:")
    print(f"  • Overall Growth: {insights['temporal']['overall_growth_%']:+.1f}%")
    print(f"  • FTE CTC Growth: {insights['temporal']['fte_ctc_growth_%']:+.1f}%")

print("\n💡 ACTIONABLE RECOMMENDATIONS FOR STUDENTS:")
print("  1. Focus on companies with consistent recruitment patterns")
print("  2. Target roles with growing demand (check temporal analysis)")
print("  3. Maintain CGPA above {:.1f} for better opportunities".format(insights['trends']['avg_cgpa']))
print("  4. Prepare for companies in top-paying categories")
print("  5. Consider internship opportunities that often lead to PPOs")

# Save insights to JSON
with open('analysis_outputs/complete_insights.json', 'w') as f:
    json.dump(insights, f, indent=4)

print("\n  ✓ Saved: complete_insights.json (with data_quality_flags)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*100)
print(" "*35 + "ANALYSIS COMPLETE!")
print("="*100)

print("\n📁 Generated Files:")
def _exists(path):
    return os.path.exists(path)
print("  • analysis_outputs/eda/")
print("    - year_tier_distribution.png")
if _exists('analysis_outputs/eda/ctc_comprehensive_analysis.png'): print("    - ctc_comprehensive_analysis.png")
if _exists('analysis_outputs/eda/top_companies.png'): print("    - top_companies.png")
print("  • analysis_outputs/temporal/")
if _exists('analysis_outputs/temporal/comprehensive_temporal_analysis.png'): print("    - comprehensive_temporal_analysis.png")
print("  • analysis_outputs/cross_college/")
if _exists('analysis_outputs/cross_college/college_comparison.png'): print("    - college_comparison.png")
print("  • analysis_outputs/statistical/")
print("    - correlation_matrix.png")
if _exists('analysis_outputs/statistical/cgpa_vs_ctc.png'): print("    - cgpa_vs_ctc.png")
print("  • analysis_outputs/predictive/")
if modeling_mode == 'regression':
    if _exists('analysis_outputs/predictive/model_performance.png'): print("    - model_performance.png")
    if _exists('analysis_outputs/predictive/model_comparison.png'): print("    - model_comparison.png")
elif modeling_mode == 'classification':
    if _exists('analysis_outputs/predictive/salary_category_confusion_matrix.png'): print("    - salary_category_confusion_matrix.png")
    if _exists('analysis_outputs/predictive/model_performance.png'): print("    - model_performance.png")
    if _exists('analysis_outputs/predictive/model_comparison.png'): print("    - model_comparison.png")
print("  • analysis_outputs/complete_insights.json")

print("\n📊 All visualizations and insights have been generated successfully!")
print("🎓 Use these insights to prepare better for your placement season!")
print("\n" + "="*100)
print(f"Analysis completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*100)
