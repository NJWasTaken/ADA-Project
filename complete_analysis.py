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
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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

# Filter valid data
df_ctc = df[df['total_ctc'].notna() & (df['total_ctc'] > 0)].copy()
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

# Compensation statistics
if len(df_ctc) > 0:
    print("\n💰 Compensation Statistics (LPA):")
    print(f"  • Mean CTC: ₹{df_ctc['total_ctc'].mean():.2f}")
    print(f"  • Median CTC: ₹{df_ctc['total_ctc'].median():.2f}")
    print(f"  • Std Dev: ₹{df_ctc['total_ctc'].std():.2f}")
    print(f"  • Range: ₹{df_ctc['total_ctc'].min():.2f} - ₹{df_ctc['total_ctc'].max():.2f}")
    print(f"  • 90th Percentile: ₹{df_ctc['total_ctc'].quantile(0.90):.2f}")

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
    
    axes[0, 0].hist(df_ctc['total_ctc'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Total CTC (LPA)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('CTC Distribution', fontsize=13, fontweight='bold')
    axes[0, 0].axvline(df_ctc['total_ctc'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ₹{df_ctc["total_ctc"].mean():.2f}')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    yearly_ctc = df_ctc.groupby('batch_year')['total_ctc'].agg(['mean', 'median'])
    axes[0, 1].plot(yearly_ctc.index, yearly_ctc['mean'], marker='o', linewidth=2, markersize=8, label='Mean', color='blue')
    axes[0, 1].plot(yearly_ctc.index, yearly_ctc['median'], marker='s', linewidth=2, markersize=8, label='Median', color='green')
    axes[0, 1].set_xlabel('Batch Year', fontsize=11)
    axes[0, 1].set_ylabel('CTC (LPA)', fontsize=11)
    axes[0, 1].set_title('CTC Trends Over Years', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    tier_ctc = df_ctc.groupby('placement_tier')['total_ctc'].mean().sort_values()
    axes[1, 0].barh(tier_ctc.index, tier_ctc.values, color='purple', edgecolor='black')
    axes[1, 0].set_xlabel('Average CTC (LPA)', fontsize=11)
    axes[1, 0].set_ylabel('Placement Tier', fontsize=11)
    axes[1, 0].set_title('Average CTC by Tier', fontsize=13, fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    role_ctc = df_ctc.groupby('role_type')['total_ctc'].mean().sort_values(ascending=False).head(10)
    axes[1, 1].barh(range(len(role_ctc)), role_ctc.values, color='orange', edgecolor='black')
    axes[1, 1].set_yticks(range(len(role_ctc)))
    axes[1, 1].set_yticklabels(role_ctc.index)
    axes[1, 1].set_xlabel('Average CTC (LPA)', fontsize=11)
    axes[1, 1].set_ylabel('Role Type', fontsize=11)
    axes[1, 1].set_title('Top 10 Roles by Average CTC', fontsize=13, fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_outputs/eda/ctc_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Top companies visualization
company_ctc = df_ctc.groupby('company_name').agg({
    'total_ctc': ['mean', 'count']
}).reset_index()
company_ctc.columns = ['company_name', 'avg_ctc', 'count']
company_ctc = company_ctc[company_ctc['count'] >= 2]
top_paying = company_ctc.nlargest(15, 'avg_ctc').sort_values('avg_ctc')

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
top_recruiters.sort_values().plot(kind='barh', ax=axes[0], color='teal', edgecolor='black')
axes[0].set_xlabel('Number of Offers', fontsize=12)
axes[0].set_ylabel('Company Name', fontsize=12)
axes[0].set_title('Top 10 Recruiters by Offer Count', fontsize=14, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

axes[1].barh(range(len(top_paying)), top_paying['avg_ctc'], color='gold', edgecolor='black')
axes[1].set_yticks(range(len(top_paying)))
axes[1].set_yticklabels(top_paying['company_name'])
axes[1].set_xlabel('Average CTC (LPA)', fontsize=12)
axes[1].set_ylabel('Company Name', fontsize=12)
axes[1].set_title('Top 15 Highest Paying Companies', fontsize=14, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('analysis_outputs/eda/top_companies.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Saved: year_tier_distribution.png")
print("  ✓ Saved: ctc_comprehensive_analysis.png")
print("  ✓ Saved: top_companies.png")

# ============================================================================
# PART 3: TEMPORAL TREND ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("PART 3: TEMPORAL TREND ANALYSIS")
print("="*100)

# Year-over-year growth analysis
yearly_stats = df.groupby('batch_year').agg({
    'company_name': ['count', lambda x: x.nunique()]
}).reset_index()
yearly_stats.columns = ['batch_year', 'total_records', 'unique_companies']
yearly_stats = yearly_stats.set_index('batch_year')

# Add CTC statistics
ctc_stats = df_ctc.groupby('batch_year')['total_ctc'].agg(['mean', 'median', 'std'])
yearly_stats = yearly_stats.join(ctc_stats, how='left')

# Calculate growth rates
yearly_stats['record_growth_%'] = yearly_stats['total_records'].pct_change() * 100
yearly_stats['ctc_growth_%'] = yearly_stats['mean'].pct_change() * 100

print("\n📈 Year-over-Year Growth Analysis:")
print(yearly_stats.round(2))

# Overall growth
if len(yearly_stats) >= 2:
    total_growth = ((yearly_stats.iloc[-1]['total_records'] - yearly_stats.iloc[0]['total_records']) / 
                   yearly_stats.iloc[0]['total_records'] * 100)
    ctc_growth = ((yearly_stats.iloc[-1]['mean'] - yearly_stats.iloc[0]['mean']) / 
                  yearly_stats.iloc[0]['mean'] * 100)
    
    print(f"\n🎯 Overall Growth ({yearly_stats.index[0]} to {yearly_stats.index[-1]}):")
    print(f"  • Total Records: {total_growth:+.1f}%")
    print(f"  • Average CTC: {ctc_growth:+.1f}%")

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

# ============================================================================
# PART 4: CROSS-COLLEGE COMPARATIVE ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("PART 4: CROSS-COLLEGE COMPARATIVE ANALYSIS")
print("="*100)

colleges = df['college'].unique()
print(f"\n🏫 Colleges in dataset: {colleges}")

if len(colleges) > 1:
    college_stats = df_ctc.groupby('college')['total_ctc'].agg(['count', 'mean', 'median', 'std']).round(2)
    print("\n📊 College-wise CTC Statistics:")
    print(college_stats)
    
    # Visualizations
    print("\n📈 Generating cross-college visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # CTC comparison
    college_ctc = df_ctc.groupby('college')['total_ctc'].mean().sort_values()
    college_ctc.plot(kind='barh', ax=axes[0, 0], color='purple', edgecolor='black')
    axes[0, 0].set_xlabel('Average CTC (LPA)', fontsize=12)
    axes[0, 0].set_ylabel('College', fontsize=12)
    axes[0, 0].set_title('Average CTC by College', fontsize=14, fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Box plot comparison
    sns.boxplot(data=df_ctc, x='college', y='total_ctc', ax=axes[0, 1], palette='Set2')
    axes[0, 1].set_xlabel('College', fontsize=12)
    axes[0, 1].set_ylabel('Total CTC (LPA)', fontsize=12)
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
numeric_cols = ['batch_year', 'total_ctc', 'cgpa_cutoff', 'num_offers_total', 
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
df_both = df[(df['cgpa_cutoff'].notna()) & (df['total_ctc'].notna()) & (df['total_ctc'] > 0)]
if len(df_both) > 20:
    corr_cgpa_ctc = df_both['cgpa_cutoff'].corr(df_both['total_ctc'])
    print(f"\n📊 CGPA vs CTC Correlation: {corr_cgpa_ctc:.3f}")
    
    plt.figure(figsize=(12, 6))
    plt.scatter(df_both['cgpa_cutoff'], df_both['total_ctc'], alpha=0.5, color='blue')
    plt.xlabel('CGPA Cutoff', fontsize=12)
    plt.ylabel('Total CTC (LPA)', fontsize=12)
    plt.title(f'CGPA Cutoff vs CTC (Correlation: {corr_cgpa_ctc:.3f})', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('analysis_outputs/statistical/cgpa_vs_ctc.png', dpi=300, bbox_inches='tight')
    plt.close()

# Statistical tests
if len(colleges) > 1:
    print("\n🧪 Statistical Tests:")
    college_groups = [df_ctc[df_ctc['college'] == college]['total_ctc'].values 
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
    
    features = ['batch_year', 'cgpa_cutoff', 'company_encoded', 'role_type_encoded', 
                'tier_encoded', 'has_internship', 'has_stocks', 'has_joining_bonus']
    target = 'total_ctc'
    
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
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
        
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
    axes[0].set_xlabel('Actual CTC (LPA)', fontsize=12)
    axes[0].set_ylabel('Predicted CTC (LPA)', fontsize=12)
    axes[0].set_title(f'Actual vs Predicted CTC (R² = {results["Random Forest"]["test_r2"]:.3f})', 
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
    print("  ⚠ Insufficient data for modeling")

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
    'compensation': {
        'avg_ctc': float(df_ctc['total_ctc'].mean()) if len(df_ctc) > 0 else 0,
        'median_ctc': float(df_ctc['total_ctc'].median()) if len(df_ctc) > 0 else 0,
        'max_ctc': float(df_ctc['total_ctc'].max()) if len(df_ctc) > 0 else 0,
        'p90_ctc': float(df_ctc['total_ctc'].quantile(0.90)) if len(df_ctc) > 0 else 0
    },
    'top_performers': {
        'top_recruiter': top_recruiters.index[0] if len(top_recruiters) > 0 else 'N/A',
        'top_recruiter_offers': int(top_recruiters.iloc[0]) if len(top_recruiters) > 0 else 0,
        'highest_paying': top_paying.iloc[-1]['company_name'] if len(top_paying) > 0 else 'N/A',
        'highest_avg_ctc': float(top_paying.iloc[-1]['avg_ctc']) if len(top_paying) > 0 else 0
    },
    'trends': {
        'most_common_role': df['role_type'].value_counts().index[0],
        'avg_cgpa': float(df_cgpa['cgpa_cutoff'].mean()) if len(df_cgpa) > 0 else 0,
        'peak_year': int(year_dist.idxmax()) if len(year_dist) > 0 else 0
    },
    'temporal': {
        'overall_growth_%': float(total_growth) if 'total_growth' in locals() else 0,
        'ctc_growth_%': float(ctc_growth) if 'ctc_growth' in locals() else 0
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

print(f"\n💰 COMPENSATION INSIGHTS:")
print(f"  • Average CTC: ₹{insights['compensation']['avg_ctc']:.2f} LPA")
print(f"  • Median CTC: ₹{insights['compensation']['median_ctc']:.2f} LPA")
print(f"  • Highest Package: ₹{insights['compensation']['max_ctc']:.2f} LPA")
print(f"  • 90th Percentile: ₹{insights['compensation']['p90_ctc']:.2f} LPA")

print(f"\n🏆 TOP PERFORMERS:")
print(f"  • Top Recruiter: {insights['top_performers']['top_recruiter']} ({insights['top_performers']['top_recruiter_offers']} offers)")
print(f"  • Highest Paying: {insights['top_performers']['highest_paying']} (₹{insights['top_performers']['highest_avg_ctc']:.2f} LPA)")

print(f"\n📈 TRENDS:")
print(f"  • Most Common Role: {insights['trends']['most_common_role']}")
print(f"  • Average CGPA Cutoff: {insights['trends']['avg_cgpa']:.2f}")
print(f"  • Peak Placement Year: {insights['trends']['peak_year']}")

if 'total_growth' in locals():
    print(f"\n🎯 TEMPORAL ANALYSIS:")
    print(f"  • Overall Growth: {insights['temporal']['overall_growth_%']:+.1f}%")
    print(f"  • CTC Growth: {insights['temporal']['ctc_growth_%']:+.1f}%")

print("\n💡 ACTIONABLE RECOMMENDATIONS FOR STUDENTS:")
print("  1. Focus on companies with consistent recruitment patterns")
print("  2. Target roles with growing demand (check temporal analysis)")
print("  3. Maintain CGPA above {:.1f} for better opportunities".format(insights['trends']['avg_cgpa']))
print("  4. Prepare for companies in top-paying categories")
print("  5. Consider internship opportunities that often lead to PPOs")

# Save insights to JSON
with open('analysis_outputs/complete_insights.json', 'w') as f:
    json.dump(insights, f, indent=4)

print("\n  ✓ Saved: complete_insights.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*100)
print(" "*35 + "ANALYSIS COMPLETE!")
print("="*100)

print("\n📁 Generated Files:")
print("  • analysis_outputs/eda/")
print("    - year_tier_distribution.png")
print("    - ctc_comprehensive_analysis.png")
print("    - top_companies.png")
print("  • analysis_outputs/temporal/")
print("    - comprehensive_temporal_analysis.png")
print("  • analysis_outputs/cross_college/")
print("    - college_comparison.png (if applicable)")
print("  • analysis_outputs/statistical/")
print("    - correlation_matrix.png")
print("    - cgpa_vs_ctc.png (if applicable)")
print("  • analysis_outputs/predictive/")
print("    - model_performance.png")
print("    - model_comparison.png")
print("  • analysis_outputs/complete_insights.json")

print("\n📊 All visualizations and insights have been generated successfully!")
print("🎓 Use these insights to prepare better for your placement season!")
print("\n" + "="*100)
print(f"Analysis completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*100)
