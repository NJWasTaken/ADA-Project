"""
EDA Starter Script for PES Placement Data
This script provides initial exploratory analysis and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
print("=" * 80)
print("PES University Placement Data - EDA Starter")
print("=" * 80)

df = pd.read_csv('processed_data/consolidated_placement_data.csv')

print(f"\nDataset loaded: {len(df)} records")
print(f"Date range: {df['batch_year'].min()} - {df['batch_year'].max()}")
print(f"Unique companies: {df['company_name'].nunique()}")

# ============================================================================
# 1. BASIC STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("1. BASIC STATISTICS")
print("=" * 80)

print("\nDataset Info:")
print(f"  Total Records: {len(df)}")
print(f"  Total Columns: {len(df.columns)}")
print(f"  Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nRecords by Year:")
year_counts = df['batch_year'].value_counts().sort_index()
for year, count in year_counts.items():
    print(f"  {year}: {count:4d} records ({count/len(df)*100:.1f}%)")

print("\nRecords by Placement Tier:")
tier_counts = df['placement_tier'].value_counts()
for tier, count in tier_counts.head(10).items():
    print(f"  {tier}: {count:4d} records ({count/len(df)*100:.1f}%)")

# ============================================================================
# 2. COMPENSATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("2. COMPENSATION ANALYSIS")
print("=" * 80)

# Filter records with valid CTC
df_ctc = df[df['total_ctc'].notna() & (df['total_ctc'] > 0)]

print(f"\nRecords with CTC data: {len(df_ctc)} ({len(df_ctc)/len(df)*100:.1f}%)")

if len(df_ctc) > 0:
    print("\nCTC Statistics (LPA):")
    print(f"  Mean:     ₹{df_ctc['total_ctc'].mean():6.2f}")
    print(f"  Median:   ₹{df_ctc['total_ctc'].median():6.2f}")
    print(f"  Std Dev:  ₹{df_ctc['total_ctc'].std():6.2f}")
    print(f"  Min:      ₹{df_ctc['total_ctc'].min():6.2f}")
    print(f"  Max:      ₹{df_ctc['total_ctc'].max():6.2f}")

    print("\nCTC Percentiles (LPA):")
    for p in [25, 50, 75, 90, 95]:
        val = df_ctc['total_ctc'].quantile(p/100)
        print(f"  {p:2d}th: ₹{val:6.2f}")

    # Year-wise compensation trends
    print("\nAverage CTC by Year:")
    yearly_ctc = df_ctc.groupby('batch_year')['total_ctc'].agg(['mean', 'median', 'count'])
    for year, row in yearly_ctc.iterrows():
        print(f"  {year}: Mean=₹{row['mean']:6.2f}, Median=₹{row['median']:6.2f} (n={int(row['count'])})")

    # Tier-wise compensation
    print("\nAverage CTC by Placement Tier:")
    tier_ctc = df_ctc.groupby('placement_tier')['total_ctc'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
    for tier, row in tier_ctc.head(10).iterrows():
        print(f"  {tier:20s}: Mean=₹{row['mean']:6.2f}, Median=₹{row['median']:6.2f} (n={int(row['count'])})")

# ============================================================================
# 3. TOP COMPANIES ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("3. TOP COMPANIES ANALYSIS")
print("=" * 80)

print("\nTop 15 Companies by Number of Offers:")
top_recruiters = df.groupby('company_name')['num_offers_total'].sum().nlargest(15)
for idx, (company, offers) in enumerate(top_recruiters.items(), 1):
    print(f"  {idx:2d}. {company:30s}: {offers:4.0f} offers")

if len(df_ctc) > 0:
    print("\nTop 15 Companies by Average CTC:")
    # Filter companies with at least 2 placements for reliability
    company_ctc = df_ctc.groupby('company_name').agg({
        'total_ctc': ['mean', 'count']
    }).reset_index()
    company_ctc.columns = ['company_name', 'avg_ctc', 'count']
    company_ctc = company_ctc[company_ctc['count'] >= 1]  # At least 1 placement
    company_ctc = company_ctc.nlargest(15, 'avg_ctc')

    for idx, row in company_ctc.iterrows():
        print(f"  {row.name+1:2d}. {row['company_name']:30s}: ₹{row['avg_ctc']:6.2f} LPA (n={int(row['count'])})")

# ============================================================================
# 4. ROLE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("4. JOB ROLE ANALYSIS")
print("=" * 80)

print("\nRole Type Distribution:")
role_dist = df['role_type'].value_counts()
for role, count in role_dist.items():
    print(f"  {role:25s}: {count:4d} records ({count/len(df)*100:.1f}%)")

if len(df_ctc) > 0:
    print("\nAverage CTC by Role Type:")
    role_ctc = df_ctc.groupby('role_type')['total_ctc'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
    for role, row in role_ctc.iterrows():
        print(f"  {role:25s}: Mean=₹{row['mean']:6.2f}, Median=₹{row['median']:6.2f} (n={int(row['count'])})")

# ============================================================================
# 5. CGPA ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("5. CGPA CUTOFF ANALYSIS")
print("=" * 80)

df_cgpa = df[df['cgpa_cutoff'].notna()]
print(f"\nRecords with CGPA cutoff data: {len(df_cgpa)} ({len(df_cgpa)/len(df)*100:.1f}%)")

if len(df_cgpa) > 0:
    print("\nCGPA Cutoff Statistics:")
    print(f"  Mean:   {df_cgpa['cgpa_cutoff'].mean():.2f}")
    print(f"  Median: {df_cgpa['cgpa_cutoff'].median():.2f}")
    print(f"  Min:    {df_cgpa['cgpa_cutoff'].min():.2f}")
    print(f"  Max:    {df_cgpa['cgpa_cutoff'].max():.2f}")

    print("\nCGPA Distribution:")
    cgpa_dist = df_cgpa['cgpa_cutoff'].value_counts().sort_index()
    for cgpa, count in cgpa_dist.items():
        print(f"  {cgpa:.1f}: {'█' * int(count/5)} {count}")

    # CGPA by tier
    print("\nAverage CGPA Cutoff by Tier:")
    tier_cgpa = df_cgpa.groupby('placement_tier')['cgpa_cutoff'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    for tier, row in tier_cgpa.head(10).iterrows():
        print(f"  {tier:20s}: {row['mean']:.2f} (n={int(row['count'])})")

# ============================================================================
# 6. PLACEMENT OFFERS ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("6. PLACEMENT OFFERS ANALYSIS")
print("=" * 80)

df_offers = df[df['num_offers_total'].notna() & (df['num_offers_total'] > 0)]
print(f"\nRecords with offer count data: {len(df_offers)} ({len(df_offers)/len(df)*100:.1f}%)")

if len(df_offers) > 0:
    print(f"\nTotal Offers Tracked: {df_offers['num_offers_total'].sum():.0f}")

    print("\nOffer Type Distribution:")
    fte_offers = df[df['num_offers_fte'].notna()]['num_offers_fte'].sum()
    intern_offers = df[df['num_offers_intern'].notna()]['num_offers_intern'].sum()
    both_offers = df[df['num_offers_both'].notna()]['num_offers_both'].sum()

    print(f"  FTE Only:        {fte_offers:4.0f}")
    print(f"  Internship Only: {intern_offers:4.0f}")
    print(f"  FTE + Intern:    {both_offers:4.0f}")

# ============================================================================
# 7. ADDITIONAL BENEFITS ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("7. ADDITIONAL BENEFITS ANALYSIS")
print("=" * 80)

print("\nCompanies Offering Additional Benefits:")
print(f"  Internship Program:  {df['has_internship'].sum():4d} companies ({df['has_internship'].sum()/len(df)*100:.1f}%)")
print(f"  Stocks/ESOPs:        {df['has_stocks'].sum():4d} companies ({df['has_stocks'].sum()/len(df)*100:.1f}%)")
print(f"  Joining Bonus:       {df['has_joining_bonus'].sum():4d} companies ({df['has_joining_bonus'].sum()/len(df)*100:.1f}%)")

if df['has_stocks'].sum() > 0:
    df_stocks = df[df['stocks_esops'].notna()]
    print(f"\nAverage Stock/ESOP value: ₹{df_stocks['stocks_esops'].mean():.2f} LPA")

if df['has_joining_bonus'].sum() > 0:
    df_jb = df[df['joining_bonus'].notna()]
    print(f"Average Joining Bonus: ₹{df_jb['joining_bonus'].mean():.2f} LPA")

# ============================================================================
# 8. DATA QUALITY SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("8. DATA QUALITY SUMMARY")
print("=" * 80)

print("\nField Completeness:")
key_fields = ['total_ctc', 'base_salary', 'internship_stipend', 'cgpa_cutoff',
              'num_offers_total', 'job_role', 'placement_tier']

for field in key_fields:
    if field in df.columns:
        non_null = df[field].notna().sum()
        pct = non_null / len(df) * 100
        bar = '█' * int(pct / 5)
        print(f"  {field:25s}: {bar:20s} {pct:5.1f}% ({non_null}/{len(df)})")

# ============================================================================
# 9. BASIC VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("9. GENERATING BASIC VISUALIZATIONS")
print("=" * 80)

try:
    # Create output directory
    Path('eda_outputs').mkdir(exist_ok=True)

    # 1. CTC Distribution
    if len(df_ctc) > 0:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(df_ctc[df_ctc['total_ctc'] < 50]['total_ctc'], bins=30, edgecolor='black')
        plt.xlabel('Total CTC (LPA)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Total CTC (<50 LPA)')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        df_ctc.boxplot(column='total_ctc', by='batch_year')
        plt.xlabel('Batch Year')
        plt.ylabel('Total CTC (LPA)')
        plt.title('CTC Distribution by Year')
        plt.suptitle('')
        plt.savefig('eda_outputs/ctc_distribution.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: eda_outputs/ctc_distribution.png")
        plt.close()

    # 2. Year-wise trends
    plt.figure(figsize=(12, 6))
    year_stats = df.groupby('batch_year').size()
    plt.subplot(1, 2, 1)
    year_stats.plot(kind='bar', color='steelblue')
    plt.xlabel('Batch Year')
    plt.ylabel('Number of Records')
    plt.title('Placement Records by Year')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)

    if len(df_ctc) > 0:
        plt.subplot(1, 2, 2)
        yearly_avg = df_ctc.groupby('batch_year')['total_ctc'].mean()
        yearly_avg.plot(kind='line', marker='o', color='green', linewidth=2)
        plt.xlabel('Batch Year')
        plt.ylabel('Average CTC (LPA)')
        plt.title('Average CTC Trend Over Years')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('eda_outputs/yearly_trends.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: eda_outputs/yearly_trends.png")
    plt.close()

    # 3. Role type distribution
    plt.figure(figsize=(12, 6))
    role_dist.head(10).plot(kind='barh', color='coral')
    plt.xlabel('Number of Records')
    plt.ylabel('Role Type')
    plt.title('Top 10 Role Types')
    plt.tight_layout()
    plt.savefig('eda_outputs/role_distribution.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: eda_outputs/role_distribution.png")
    plt.close()

    # 4. Top recruiters
    plt.figure(figsize=(12, 6))
    top_recruiters.head(15).plot(kind='barh', color='teal')
    plt.xlabel('Number of Offers')
    plt.ylabel('Company')
    plt.title('Top 15 Recruiters by Offer Count')
    plt.tight_layout()
    plt.savefig('eda_outputs/top_recruiters.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: eda_outputs/top_recruiters.png")
    plt.close()

    # 5. Tier distribution
    plt.figure(figsize=(10, 6))
    tier_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.ylabel('')
    plt.title('Placement Distribution by Tier')
    plt.tight_layout()
    plt.savefig('eda_outputs/tier_distribution.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: eda_outputs/tier_distribution.png")
    plt.close()

    print("\n  All visualizations saved to 'eda_outputs/' directory")

except Exception as e:
    print(f"  Error creating visualizations: {e}")

# ============================================================================
# 10. KEY INSIGHTS
# ============================================================================
print("\n" + "=" * 80)
print("10. KEY INSIGHTS")
print("=" * 80)

print("\n📊 Data Coverage:")
print(f"  • {len(df)} placement records across {df['batch_year'].nunique()} years")
print(f"  • {df['company_name'].nunique()} unique companies participated")
print(f"  • Coverage period: {df['batch_year'].min()} to {df['batch_year'].max()}")

if len(df_ctc) > 0:
    print("\n💰 Compensation Insights:")
    print(f"  • Average CTC: ₹{df_ctc['total_ctc'].mean():.2f} LPA")
    print(f"  • Highest package: ₹{df_ctc['total_ctc'].max():.2f} LPA")
    print(f"  • 90th percentile: ₹{df_ctc['total_ctc'].quantile(0.9):.2f} LPA")

    if len(yearly_ctc) > 1:
        growth = ((yearly_ctc.iloc[-1]['mean'] - yearly_ctc.iloc[0]['mean']) /
                 yearly_ctc.iloc[0]['mean'] * 100)
        print(f"  • YoY growth in average CTC: {growth:+.1f}%")

print("\n🏆 Top Performers:")
print(f"  • Top recruiter: {top_recruiters.index[0]} ({top_recruiters.iloc[0]:.0f} offers)")
if len(df_ctc) > 0 and len(company_ctc) > 0:
    print(f"  • Highest paying: {company_ctc.iloc[0]['company_name']} (₹{company_ctc.iloc[0]['avg_ctc']:.2f} LPA)")

print("\n📈 Trends:")
print(f"  • Most popular role type: {role_dist.index[0]} ({role_dist.iloc[0]} positions)")
print(f"  • Peak placement year: {year_counts.idxmax()} ({year_counts.max()} records)")

if len(df_cgpa) > 0:
    print(f"  • Average CGPA requirement: {df_cgpa['cgpa_cutoff'].mean():.2f}")

print("\n✅ Data Quality:")
print(f"  • CTC data available: {len(df_ctc)/len(df)*100:.1f}%")
print(f"  • CGPA data available: {len(df_cgpa)/len(df)*100:.1f}%")
print(f"  • Offer count data available: {len(df_offers)/len(df)*100:.1f}%")

# ============================================================================
# RECOMMENDATIONS FOR FURTHER ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("RECOMMENDATIONS FOR FURTHER ANALYSIS")
print("=" * 80)

print("""
1. DEEP DIVE ANALYSES:
   • Company-specific trend analysis over years
   • Role evolution and emerging job categories
   • Compensation benchmarking by industry
   • CGPA vs CTC correlation analysis

2. PREDICTIVE MODELING:
   • CTC prediction based on company, role, tier, year
   • Placement probability estimation
   • Company tier classification
   • Demand forecasting for specific roles

3. COMPARATIVE STUDIES:
   • Cross-college placement comparison (PES vs RVCE vs BMS)
   • Tier-wise performance metrics
   • Internship to FTE conversion rates

4. VISUALIZATION ENHANCEMENTS:
   • Interactive dashboards (Plotly/Dash)
   • Time series analysis
   • Correlation heatmaps
   • Geographic distribution (if location data added)

5. DATA ENRICHMENT:
   • Collect missing compensation breakdowns
   • Add company industry/sector information
   • Include location data for offers
   • Track previous year packages for same companies

For detailed data quality information, refer to: DATA_QUALITY_REPORT.md
For complete field descriptions, refer to: README_DATA_CONSOLIDATION.md
""")

print("=" * 80)
print("EDA Complete! Check 'eda_outputs/' directory for visualizations")
print("=" * 80)
