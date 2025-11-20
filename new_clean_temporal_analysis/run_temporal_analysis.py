"""
Temporal Analysis for PES Cross-College Placement Data
This script creates visualizations and insights about recruitment timeline
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("Starting analysis...")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
print("Loading data...")
df = pd.read_csv('new_clean_data/cross_college_PES_cleaned.csv')
print(f"Loaded {len(df)} rows")

# Parse dates
print("Parsing dates...")
df['Date_OA'] = pd.to_datetime(df['Date_OA'], format='%d-%m-%Y', errors='coerce')

# Extract temporal features
df['Month'] = df['Date_OA'].dt.month
df['MonthName'] = df['Date_OA'].dt.month_name()
df['DayOfWeek'] = df['Date_OA'].dt.day_name()

# Filter valid dates
df_valid = df.dropna(subset=['Date_OA']).copy()
print(f"Valid dates: {len(df_valid)} rows")

print("\n" + "="*80)
print("TEMPORAL ANALYSIS - PES CAMPUS PLACEMENTS 2025")
print("="*80)

print(f"\nDataset Overview:")
print(f"  Total Companies: {len(df)}")
print(f"  Companies with Dates: {len(df_valid)}")
print(f"  Date Range: {df_valid['Date_OA'].min().strftime('%d %B %Y')} to {df_valid['Date_OA'].max().strftime('%d %B %Y')}")
print(f"  Duration: {(df_valid['Date_OA'].max() - df_valid['Date_OA'].min()).days} days")

# Monthly distribution
print("\n" + "-"*80)
print("MONTHLY DISTRIBUTION:")
print("-"*80)
monthly_counts = df_valid['MonthName'].value_counts()
month_order = ['July', 'August', 'September', 'October', 'November', 'December', 'January']
for month in month_order:
    count = monthly_counts.get(month, 0)
    if count > 0:
        bar = '█' * int(count / 2)
        print(f"{month:12s}: {count:3d} {bar}")

# Day of week distribution 
print("\n" +"="*80)
print("DAY OF WEEK DISTRIBUTION:")
print("="*80)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_counts = df_valid['DayOfWeek'].value_counts().reindex(day_order, fill_value=0)
for day in day_order:
    count = day_counts.get(day, 0)
    bar = '█' * int(count / 3)
    print(f"{day:12s}: {count:3d} {bar}")

# Tier distribution over time
print("\n" + "="*80)
print("TIER-WISE MONTHLY BREAKDOWN:")
print("="*80)
tier_monthly = df_valid.groupby(['MonthName', 'Tier']).size().unstack(fill_value=0)
tier_monthly = tier_monthly.reindex(month_order, fill_value=0)
print(tier_monthly.to_string())

# CTC Statistics
print("\n" + "="*80)
print("CTC (LPA) STATISTICS BY MONTH:")
print("="*80)
ctc_stats = df_valid.groupby('MonthName')['CTC_LPA'].agg(['count', 'mean', 'median', 'max'])
ctc_stats = ctc_stats.reindex(month_order)
print(ctc_stats.round(2).to_string())

# Peak hiring days
print("\n" + "="*80)
print("TOP 10 BUSIEST HIRING DAYS:")
print("="*80)
daily_counts = df_valid.groupby('Date_OA').size().nlargest(10)
for i, (date, count) in enumerate(daily_counts.items(), 1):
    print(f"{i:2d}. {date.strftime('%d %B %Y (%A)')}: {count} companies")

# Job type distribution
print("\n" + "="*80)
print("JOB TYPE DISTRIBUTION:")
print("="*80)
type_counts = df_valid['Type'].value_counts()
for job_type, count in type_counts.items():
    pct = (count / len(df_valid)) * 100
    print(f"{job_type:25s}: {count:3d} ({pct:5.1f}%)")

# Online vs Offline tests
print("\n" + "="*80)
print("TEST MODE DISTRIBUTION:")
print("="*80)
test_counts = df_valid['Offline_Test'].value_counts()
print(f"Online Tests:  {test_counts.get(False, 0)} ({(test_counts.get(False, 0)/len(df_valid)*100):.1f}%)")
print(f"Offline Tests: {test_counts.get(True, 0)} ({(test_counts.get(True, 0)/len(df_valid)*100):.1f}%)")

# Top companies by offers
print("\n" + "="*80)
print("TOP 10 COMPANIES BY TOTAL OFFERS:")
print("="*80)
top_offers = df_valid.groupby('Company')['Total_Offers'].sum().nlargest(10)
for i, (company, offers) in enumerate(top_offers.items(), 1):
    print(f"{i:2d}. {company:30s}: {offers:3.0f} offers")

# CGPA cutoff analysis
print("\n" + "="*80)
print("CGPA CUTOFF STATISTICS:")
print("="*80)
cgpa_stats = df_valid['CGPA_Cutoff'].describe()
print(f"Mean CGPA Cutoff:   {cgpa_stats['mean']:.2f}")
print(f"Median CGPA Cutoff: {cgpa_stats['50%']:.2f}")
print(f"Min CGPA Cutoff:    {cgpa_stats['min']:.2f}")
print(f"Max CGPA Cutoff:    {cgpa_stats['max']:.2f}")

# Hiring pace
print("\n" + "="*80)
print("HIRING PACE ANALYSIS:")
print("="*80)
df_sorted = df_valid.sort_values('Date_OA')
time_diffs = df_sorted['Date_OA'].diff().dt.days
print(f"Average days between companies: {time_diffs.mean():.2f}")
print(f"Median days between companies:  {time_diffs.median():.2f}")
print(f"Shortest gap:                   {time_diffs.min():.0f} days")
print(f"Longest gap:                    {time_diffs.max():.0f} days")

print("\n" + "="*80)
print("Creating visualizations...")
print("="*80)

# Create visualizations
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('Temporal Analysis - PES Campus Placements 2025', fontsize=16, fontweight='bold')

# 1. Companies per month
ax = axes[0, 0]
monthly_counts_ordered = df_valid['MonthName'].value_counts().reindex(month_order, fill_value=0)
bars = ax.bar(range(len(monthly_counts_ordered)), monthly_counts_ordered.values, 
              color=plt.cm.viridis(np.linspace(0, 1, len(monthly_counts_ordered))))
ax.set_xticks(range(len(monthly_counts_ordered)))
ax.set_xticklabels(monthly_counts_ordered.index, rotation=45, ha='right')
ax.set_title('Companies per Month', fontweight='bold')
ax.set_ylabel('Count')
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# 2. Day of week
ax = axes[0, 1]
bars = ax.bar(range(len(day_counts)), day_counts.values,
              color=plt.cm.plasma(np.linspace(0, 1, len(day_counts))))
ax.set_xticks(range(len(day_counts)))
ax.set_xticklabels(day_counts.index, rotation=45, ha='right')
ax.set_title('Distribution by Day of Week', fontweight='bold')
ax.set_ylabel('Count')
ax.grid(True, alpha=0.3, axis='y')

# 3. Cumulative timeline
ax = axes[1, 0]
cumulative = range(1, len(df_sorted) + 1)
ax.plot(df_sorted['Date_OA'], cumulative, linewidth=2.5, color='#2E86AB')
ax.fill_between(df_sorted['Date_OA'], cumulative, alpha=0.3, color='#2E86AB')
ax.set_title('Cumulative Companies Over Time', fontweight='bold')
ax.set_ylabel('Cumulative Count')
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 4. CTC distribution
ax = axes[1, 1]
df_ctc = df_valid[df_valid['CTC_LPA'].notna()]
scatter = ax.scatter(df_ctc['Date_OA'], df_ctc['CTC_LPA'],
                    c=df_ctc['CTC_LPA'], cmap='coolwarm',
                    s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax.set_title('CTC Distribution Over Time', fontweight='bold')
ax.set_ylabel('CTC (LPA)')
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.colorbar(scatter, ax=ax, label='CTC (LPA)')

# 5. Tier stacked bar
ax = axes[2, 0]
tier_monthly.plot(kind='bar', stacked=True, ax=ax, 
                 color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_title('Monthly Tier Distribution', fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Count')
ax.legend(title='Tier')
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 6. Test type distribution
ax = axes[2, 1]
test_monthly = df_valid.groupby(['MonthName', 'Offline_Test']).size().unstack(fill_value=0)
test_monthly = test_monthly.reindex(month_order, fill_value=0)
test_monthly.columns = ['Online', 'Offline']
test_monthly.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'])
ax.set_title('Online vs Offline Tests by Month', fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Count')
ax.legend(title='Test Type')
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('temporal_analysis_report.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved: temporal_analysis_report.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
