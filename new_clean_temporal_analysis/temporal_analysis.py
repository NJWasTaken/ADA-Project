import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the data
df = pd.read_csv('new_clean_data/cross_college_PES_cleaned.csv')

# Clean and parse the Date_OA column
df['Date_OA'] = pd.to_datetime(df['Date_OA'], format='%d-%m-%Y', errors='coerce')

# Extract temporal features
df['Month'] = df['Date_OA'].dt.month
df['Week'] = df['Date_OA'].dt.isocalendar().week
df['Day'] = df['Date_OA'].dt.day
df['DayOfWeek'] = df['Date_OA'].dt.day_name()
df['MonthName'] = df['Date_OA'].dt.month_name()

# Remove rows without valid dates (PPO entries and incomplete data)
df_valid = df.dropna(subset=['Date_OA']).copy()

print("=" * 80)
print("TEMPORAL ANALYSIS OF PES CAMPUS PLACEMENTS 2025")
print("=" * 80)
print(f"\nDataset Overview:")
print(f"Total Companies: {len(df)}")
print(f"Companies with Valid Dates: {len(df_valid)}")
print(f"Date Range: {df_valid['Date_OA'].min().strftime('%d %B %Y')} to {df_valid['Date_OA'].max().strftime('%d %B %Y')}")
print(f"Total Duration: {(df_valid['Date_OA'].max() - df_valid['Date_OA'].min()).days} days")

# Create figure for all visualizations
fig = plt.figure(figsize=(20, 24))

# ============================================================================
# VISUALIZATION 1: Timeline of Company Visits (Gantt-style)
# ============================================================================
ax1 = plt.subplot(6, 2, 1)
df_timeline = df_valid.sort_values('Date_OA').reset_index(drop=True)
df_timeline['index'] = range(len(df_timeline))

# Color by tier
tier_colors = {'Tier 1': '#1f77b4', 'Tier 2': '#ff7f0e', 'Tier 3': '#2ca02c'}
colors = df_timeline['Tier'].map(tier_colors).fillna('#d62728')

ax1.scatter(df_timeline['Date_OA'], df_timeline['index'], c=colors, alpha=0.6, s=50)
ax1.set_xlabel('Date', fontsize=10, fontweight='bold')
ax1.set_ylabel('Company Number (Chronological)', fontsize=10, fontweight='bold')
ax1.set_title('Timeline of Company Visits (Color-coded by Tier)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45)
ax1.grid(True, alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, label=tier) for tier, color in tier_colors.items()]
ax1.legend(handles=legend_elements, loc='upper left')

# ============================================================================
# VISUALIZATION 2: Companies per Month with Tier Breakdown
# ============================================================================
ax2 = plt.subplot(6, 2, 2)
monthly_tier = df_valid.groupby(['MonthName', 'Tier']).size().unstack(fill_value=0)
monthly_tier = monthly_tier.reindex(['July', 'August', 'September', 'October', 'November', 'December', 'January'], fill_value=0)
monthly_tier.plot(kind='bar', stacked=True, ax=ax2, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax2.set_title('Companies per Month (Stacked by Tier)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Month', fontsize=10, fontweight='bold')
ax2.set_ylabel('Number of Companies', fontsize=10, fontweight='bold')
ax2.legend(title='Tier')
plt.xticks(rotation=45)
ax2.grid(True, alpha=0.3, axis='y')

# ============================================================================
# VISUALIZATION 3: Day of Week Distribution
# ============================================================================
ax3 = plt.subplot(6, 2, 3)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_counts = df_valid['DayOfWeek'].value_counts().reindex(day_order, fill_value=0)
bars = ax3.bar(range(len(day_counts)), day_counts.values, color=plt.cm.viridis(np.linspace(0, 1, len(day_counts))))
ax3.set_xticks(range(len(day_counts)))
ax3.set_xticklabels(day_counts.index, rotation=45, ha='right')
ax3.set_title('Distribution of Company Visits by Day of Week', fontsize=12, fontweight='bold')
ax3.set_xlabel('Day of Week', fontsize=10, fontweight='bold')
ax3.set_ylabel('Number of Companies', fontsize=10, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, day_counts.values)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             str(int(val)), ha='center', va='bottom', fontweight='bold')

# ============================================================================
# VISUALIZATION 4: CTC Distribution Over Time (Scatter with Trend)
# ============================================================================
ax4 = plt.subplot(6, 2, 4)
df_ctc = df_valid[df_valid['CTC_LPA'].notna()].copy()
scatter = ax4.scatter(df_ctc['Date_OA'], df_ctc['CTC_LPA'], 
                     c=df_ctc['CTC_LPA'], cmap='coolwarm', 
                     s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
ax4.set_title('CTC (LPA) Distribution Over Time', fontsize=12, fontweight='bold')
ax4.set_xlabel('Date', fontsize=10, fontweight='bold')
ax4.set_ylabel('CTC (LPA)', fontsize=10, fontweight='bold')
plt.xticks(rotation=45)
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='CTC (LPA)')

# Add trend line
z = np.polyfit(df_ctc['Date_OA'].astype(np.int64) // 10**9, df_ctc['CTC_LPA'], 1)
p = np.poly1d(z)
ax4.plot(df_ctc['Date_OA'], p(df_ctc['Date_OA'].astype(np.int64) // 10**9), 
         "r--", alpha=0.8, linewidth=2, label='Trend')
ax4.legend()

# ============================================================================
# VISUALIZATION 5: Cumulative Companies Over Time
# ============================================================================
ax5 = plt.subplot(6, 2, 5)
df_sorted = df_valid.sort_values('Date_OA')
cumulative = range(1, len(df_sorted) + 1)
ax5.plot(df_sorted['Date_OA'], cumulative, linewidth=3, color='#2E86AB', alpha=0.8)
ax5.fill_between(df_sorted['Date_OA'], cumulative, alpha=0.3, color='#2E86AB')
ax5.set_title('Cumulative Number of Companies Over Time', fontsize=12, fontweight='bold')
ax5.set_xlabel('Date', fontsize=10, fontweight='bold')
ax5.set_ylabel('Cumulative Count', fontsize=10, fontweight='bold')
plt.xticks(rotation=45)
ax5.grid(True, alpha=0.3)

# Add milestone markers
milestones = [len(cumulative)//4, len(cumulative)//2, 3*len(cumulative)//4]
for milestone in milestones:
    ax5.axhline(y=milestone, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax5.text(df_sorted['Date_OA'].iloc[-1], milestone, f'{milestone}', 
             ha='left', va='bottom', fontweight='bold', color='red')

# ============================================================================
# VISUALIZATION 6: Weekly Hiring Activity Heatmap
# ============================================================================
ax6 = plt.subplot(6, 2, 6)
weekly = df_valid.groupby(['Week', 'DayOfWeek']).size().unstack(fill_value=0)
weekly = weekly.reindex(columns=day_order, fill_value=0)
sns.heatmap(weekly, annot=True, fmt='g', cmap='YlOrRd', ax=ax6, cbar_kws={'label': 'Count'})
ax6.set_title('Weekly Hiring Activity Heatmap', fontsize=12, fontweight='bold')
ax6.set_xlabel('Day of Week', fontsize=10, fontweight='bold')
ax6.set_ylabel('Week Number', fontsize=10, fontweight='bold')

# ============================================================================
# VISUALIZATION 7: Job Type Distribution Over Time
# ============================================================================
ax7 = plt.subplot(6, 2, 7)
monthly_type = df_valid.groupby(['MonthName', 'Type']).size().unstack(fill_value=0)
monthly_type = monthly_type.reindex(['July', 'August', 'September', 'October', 'November', 'December', 'January'], fill_value=0)
monthly_type.plot(kind='area', ax=ax7, alpha=0.7, stacked=True)
ax7.set_title('Job Type Distribution Over Time (Stacked Area)', fontsize=12, fontweight='bold')
ax7.set_xlabel('Month', fontsize=10, fontweight='bold')
ax7.set_ylabel('Number of Companies', fontsize=10, fontweight='bold')
ax7.legend(title='Job Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
ax7.grid(True, alpha=0.3)

# ============================================================================
# VISUALIZATION 8: Average CTC by Month
# ============================================================================
ax8 = plt.subplot(6, 2, 8)
monthly_ctc = df_valid.groupby('MonthName')['CTC_LPA'].agg(['mean', 'median', 'max'])
monthly_ctc = monthly_ctc.reindex(['July', 'August', 'September', 'October', 'November', 'December', 'January'])

x = range(len(monthly_ctc))
width = 0.25

bars1 = ax8.bar([i - width for i in x], monthly_ctc['mean'].fillna(0), 
                width, label='Mean', color='#3498db', alpha=0.8)
bars2 = ax8.bar(x, monthly_ctc['median'].fillna(0), 
                width, label='Median', color='#2ecc71', alpha=0.8)
bars3 = ax8.bar([i + width for i in x], monthly_ctc['max'].fillna(0), 
                width, label='Max', color='#e74c3c', alpha=0.8)

ax8.set_xlabel('Month', fontsize=10, fontweight='bold')
ax8.set_ylabel('CTC (LPA)', fontsize=10, fontweight='bold')
ax8.set_title('Average CTC Trends by Month', fontsize=12, fontweight='bold')
ax8.set_xticks(x)
ax8.set_xticklabels(monthly_ctc.index, rotation=45, ha='right')
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

# ============================================================================
# VISUALIZATION 9: Internship Stipend Over Time
# ============================================================================
ax9 = plt.subplot(6, 2, 9)
df_stipend = df_valid[df_valid['Internship_Stipend_Monthly_INR'].notna()].copy()
if len(df_stipend) > 0:
    scatter = ax9.scatter(df_stipend['Date_OA'], df_stipend['Internship_Stipend_Monthly_INR'], 
                         c=df_stipend['Internship_Stipend_Monthly_INR'], 
                         cmap='plasma', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax9.set_title('Internship Stipend Trends Over Time', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Date', fontsize=10, fontweight='bold')
    ax9.set_ylabel('Monthly Stipend (INR)', fontsize=10, fontweight='bold')
    plt.xticks(rotation=45)
    ax9.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax9, label='Stipend (INR)')
else:
    ax9.text(0.5, 0.5, 'No Stipend Data Available', ha='center', va='center')

# ============================================================================
# VISUALIZATION 10: Offline vs Online Test Trend
# ============================================================================
ax10 = plt.subplot(6, 2, 10)
monthly_test = df_valid.groupby(['MonthName', 'Offline_Test']).size().unstack(fill_value=0)
monthly_test = monthly_test.reindex(['July', 'August', 'September', 'October', 'November', 'December', 'January'], fill_value=0)
monthly_test.plot(kind='bar', ax=ax10, color=['#3498db', '#e74c3c'])
ax10.set_title('Online vs Offline Test Distribution by Month', fontsize=12, fontweight='bold')
ax10.set_xlabel('Month', fontsize=10, fontweight='bold')
ax10.set_ylabel('Number of Companies', fontsize=10, fontweight='bold')
ax10.legend(['Online', 'Offline'], title='Test Type')
plt.xticks(rotation=45)
ax10.grid(True, alpha=0.3, axis='y')

# ============================================================================
# VISUALIZATION 11: Tier Distribution Over Time (Line Chart)
# ============================================================================
ax11 = plt.subplot(6, 2, 11)
monthly_tier_line = df_valid.groupby(['MonthName', 'Tier']).size().unstack(fill_value=0)
monthly_tier_line = monthly_tier_line.reindex(['July', 'August', 'September', 'October', 'November', 'December', 'January'], fill_value=0)
monthly_tier_line.plot(ax=ax11, marker='o', linewidth=3, markersize=8)
ax11.set_title('Tier Distribution Trends Over Time', fontsize=12, fontweight='bold')
ax11.set_xlabel('Month', fontsize=10, fontweight='bold')
ax11.set_ylabel('Number of Companies', fontsize=10, fontweight='bold')
ax11.legend(title='Tier')
plt.xticks(rotation=45)
ax11.grid(True, alpha=0.3)

# ============================================================================
# VISUALIZATION 12: CGPA Cutoff Distribution Over Time
# ============================================================================
ax12 = plt.subplot(6, 2, 12)
df_cgpa = df_valid[df_valid['CGPA_Cutoff'].notna()].copy()
if len(df_cgpa) > 0:
    # Create box plots for each month
    monthly_cgpa = []
    months = []
    for month in ['July', 'August', 'September', 'October', 'November', 'December', 'January']:
        data = df_cgpa[df_cgpa['MonthName'] == month]['CGPA_Cutoff']
        if len(data) > 0:
            monthly_cgpa.append(data.values)
            months.append(month)
    
    bp = ax12.boxplot(monthly_cgpa, labels=months, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.6)
    
    ax12.set_title('CGPA Cutoff Distribution by Month', fontsize=12, fontweight='bold')
    ax12.set_xlabel('Month', fontsize=10, fontweight='bold')
    ax12.set_ylabel('CGPA Cutoff', fontsize=10, fontweight='bold')
    plt.xticks(rotation=45)
    ax12.grid(True, alpha=0.3, axis='y')
else:
    ax12.text(0.5, 0.5, 'No CGPA Data Available', ha='center', va='center')

plt.tight_layout()
plt.savefig('temporal_analysis_visualizations.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Comprehensive visualization saved as 'temporal_analysis_visualizations.png'")

# ============================================================================
# DETAILED STATISTICS AND INSIGHTS
# ============================================================================
print("\n" + "=" * 80)
print("DETAILED TEMPORAL INSIGHTS")
print("=" * 80)

# Monthly Statistics
print("\n📊 MONTHLY HIRING STATISTICS:")
print("-" * 80)
monthly_stats = df_valid.groupby('MonthName').agg({
    'Company': 'count',
    'CTC_LPA': ['mean', 'median', 'max'],
    'CGPA_Cutoff': 'mean',
    'Total_Offers': 'sum'
}).round(2)

monthly_stats = monthly_stats.reindex(['July', 'August', 'September', 'October', 'November', 'December', 'January'])
print(monthly_stats.to_string())

# Peak hiring periods
print("\n📈 PEAK HIRING ANALYSIS:")
print("-" * 80)
daily_counts = df_valid.groupby('Date_OA').size()
peak_days = daily_counts.nlargest(5)
print("Top 5 Busiest Days:")
for date, count in peak_days.items():
    print(f"  • {date.strftime('%d %B %Y (%A)')}: {count} companies")

# Tier-wise temporal analysis
print("\n🏆 TIER-WISE TEMPORAL DISTRIBUTION:")
print("-" * 80)
tier_monthly = df_valid.groupby(['MonthName', 'Tier']).size().unstack(fill_value=0)
tier_monthly = tier_monthly.reindex(['July', 'August', 'September', 'October', 'November', 'December', 'January'])
print(tier_monthly.to_string())

# Test type trends
print("\n💻 TEST TYPE TRENDS:")
print("-" * 80)
test_stats = df_valid.groupby('MonthName')['Offline_Test'].value_counts().unstack(fill_value=0)
test_stats = test_stats.reindex(['July', 'August', 'September', 'October', 'November', 'December', 'January'])
test_stats.columns = ['Online', 'Offline']
print(test_stats.to_string())

# Average time between companies
print("\n⏱️ HIRING PACE ANALYSIS:")
print("-" * 80)
df_sorted = df_valid.sort_values('Date_OA')
time_diffs = df_sorted['Date_OA'].diff().dt.days
print(f"Average days between company visits: {time_diffs.mean():.2f} days")
print(f"Median days between company visits: {time_diffs.median():.2f} days")
print(f"Shortest gap: {time_diffs.min():.0f} days")
print(f"Longest gap: {time_diffs.max():.0f} days")

# Weekly distribution
print("\n📅 WEEKLY DISTRIBUTION:")
print("-" * 80)
weekly_dist = df_valid['DayOfWeek'].value_counts().reindex(day_order)
print(weekly_dist.to_string())

# CTC trends
print("\n💰 COMPENSATION TRENDS:")
print("-" * 80)
ctc_stats = df_valid.groupby('MonthName')['CTC_LPA'].describe()[['count', 'mean', 'std', 'min', '50%', 'max']]
ctc_stats = ctc_stats.reindex(['July', 'August', 'September', 'October', 'November', 'December', 'January'])
print(ctc_stats.round(2).to_string())

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
