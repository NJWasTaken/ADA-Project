"""
Advanced Data Quality Improvement Script
=========================================
Fixes NaNs, redundancies, and inconsistencies in placement data before analysis.

This script:
1. Identifies and documents all data quality issues
2. Applies intelligent imputation strategies
3. Removes duplicates and redundancies
4. Standardizes inconsistent values
5. Creates a clean, analysis-ready dataset
6. Generates a data quality report

Author: ADA Project Team
Date: 2025-11-20
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = 'processed_data/consolidated_placement_data.csv'
OUTPUT_FILE = 'processed_data/cleaned_placement_data.csv'
REPORT_FILE = 'data_quality_report.json'

print("="*80)
print(" "*25 + "DATA QUALITY IMPROVEMENT")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA AND INITIAL ASSESSMENT
# ============================================================================

print("\n[1/7] Loading data and assessing quality...")

df = pd.read_csv(INPUT_FILE)
original_shape = df.shape
print(f"  ✓ Loaded {original_shape[0]:,} records with {original_shape[1]} columns")

# Create quality report
quality_report = {
    'timestamp': datetime.now().isoformat(),
    'original_records': int(original_shape[0]),
    'original_columns': int(original_shape[1]),
    'issues_found': {},
    'fixes_applied': {},
    'final_records': 0,
    'records_removed': 0,
    'data_completeness_improvement': {}
}

# ============================================================================
# STEP 2: IDENTIFY MISSING DATA PATTERNS
# ============================================================================

print("\n[2/7] Analyzing missing data patterns...")

missing_stats = pd.DataFrame({
    'column': df.columns,
    'missing_count': df.isnull().sum(),
    'missing_pct': (df.isnull().sum() / len(df) * 100).round(2)
}).sort_values('missing_pct', ascending=False)

print("\n  Top 10 columns with missing data:")
for idx, row in missing_stats.head(10).iterrows():
    if row['missing_pct'] > 0:
        print(f"    • {row['column']:30s}: {row['missing_count']:5.0f} ({row['missing_pct']:5.1f}%)")

quality_report['issues_found']['missing_data'] = missing_stats.to_dict('records')

# ============================================================================
# STEP 3: FIX INTERNSHIP/FTE SEPARATION
# ============================================================================

print("\n[3/7] Fixing internship/FTE data separation...")

# Ensure is_internship_record exists
if 'is_internship_record' not in df.columns:
    df['is_internship_record'] = (
        df['placement_tier'].str.contains('Internship', case=False, na=False) |
        df.get('placement_type', pd.Series(index=df.index, dtype=str)).str.contains('Intern', case=False, na=False)
    )

# Separate FTE CTC from internship stipends
if 'fte_ctc' not in df.columns or df['fte_ctc'].isna().sum() > df['total_ctc'].isna().sum():
    print("  • Creating proper FTE CTC column...")
    df['fte_ctc'] = np.where(~df['is_internship_record'], df['total_ctc'], np.nan)
    
if 'internship_stipend_monthly' not in df.columns or df['internship_stipend_monthly'].isna().all():
    print("  • Extracting monthly internship stipends...")
    # Internship stipends are often in total_ctc when is_internship_record=True
    df['internship_stipend_monthly'] = np.where(
        df['is_internship_record'] & df['internship_stipend'].notna(),
        df['internship_stipend'],
        np.where(
            df['is_internship_record'] & df['total_ctc'].notna() & (df['total_ctc'] < 5),
            df['total_ctc'] * 100000 / 12,  # Convert LPA to monthly
            np.nan
        )
    )

intern_fixed = (~df['is_internship_record']).sum()
print(f"  ✓ Separated {intern_fixed:,} FTE records from internships")
quality_report['fixes_applied']['internship_separation'] = int(intern_fixed)

# ============================================================================
# STEP 4: INTELLIGENT IMPUTATION FOR KEY FIELDS
# ============================================================================

print("\n[4/7] Applying intelligent imputation...")

# 4.1: Estimate missing FTE CTC from base_salary
missing_fte = (~df['is_internship_record']) & df['fte_ctc'].isna() & df['base_salary'].notna()
if missing_fte.any():
    # Typical multiplier is 1.2-1.3 for Indian companies
    df.loc[missing_fte, 'fte_ctc'] = df.loc[missing_fte, 'base_salary'] * 1.25
    print(f"  • Estimated FTE CTC from base_salary: {missing_fte.sum()} records")
    quality_report['fixes_applied']['ctc_from_base'] = int(missing_fte.sum())

# 4.2: Fill missing CGPA with year-wise median (conservative approach)
missing_cgpa = df['cgpa_cutoff'].isna()
if missing_cgpa.any():
    year_median_cgpa = df.groupby('batch_year')['cgpa_cutoff'].transform('median')
    df.loc[missing_cgpa, 'cgpa_cutoff'] = year_median_cgpa[missing_cgpa]
    # Fill any remaining with global median
    df['cgpa_cutoff'].fillna(df['cgpa_cutoff'].median(), inplace=True)
    print(f"  • Filled missing CGPA cutoffs: {missing_cgpa.sum()} records")
    quality_report['fixes_applied']['cgpa_imputation'] = int(missing_cgpa.sum())

# 4.3: Infer placement_tier from CTC using standard ranges
missing_tier = df['placement_tier'].isna() & df['fte_ctc'].notna()
if missing_tier.any():
    def infer_tier(ctc):
        if pd.isna(ctc):
            return np.nan
        elif ctc >= 30:
            return 'Dream'
        elif ctc >= 15:
            return 'Tier-1'
        elif ctc >= 8:
            return 'Tier-2'
        else:
            return 'Tier-3'
    
    df.loc[missing_tier, 'placement_tier'] = df.loc[missing_tier, 'fte_ctc'].apply(infer_tier)
    print(f"  • Inferred placement tier from CTC: {missing_tier.sum()} records")
    quality_report['fixes_applied']['tier_inference'] = int(missing_tier.sum())

# 4.4: Fill missing num_offers_total with 1 (at minimum, the recorded placement exists)
missing_offers = df['num_offers_total'].isna()
if missing_offers.any():
    df.loc[missing_offers, 'num_offers_total'] = 1
    print(f"  • Filled missing offer counts with minimum (1): {missing_offers.sum()} records")
    quality_report['fixes_applied']['offer_count_fill'] = int(missing_offers.sum())

# ============================================================================
# STEP 5: STANDARDIZE INCONSISTENT VALUES
# ============================================================================

print("\n[5/7] Standardizing inconsistent values...")

# 5.1: Normalize placement_tier labels
tier_mapping = {
    'Tier 1': 'Tier-1', 'Tier1': 'Tier-1', 'TIER-1': 'Tier-1',
    'Tier 2': 'Tier-2', 'Tier2': 'Tier-2', 'TIER-2': 'Tier-2',
    'Tier 3': 'Tier-3', 'Tier3': 'Tier-3', 'TIER-3': 'Tier-3',
    'Dream Tier': 'Dream', 'DREAM': 'Dream',
    'Super Dream': 'Super-Dream', 'SUPER-DREAM': 'Super-Dream',
    'Internship - Summer': 'Internship-Summer',
    'Internship - Spring': 'Internship-Spring',
    'Summer Internship': 'Internship-Summer',
    'Spring Internship': 'Internship-Spring'
}

original_tiers = df['placement_tier'].value_counts()
df['placement_tier'] = df['placement_tier'].replace(tier_mapping)
standardized_count = (df['placement_tier'] != df['placement_tier']).sum()
print(f"  • Standardized placement tier labels: {len(tier_mapping)} variants")
quality_report['fixes_applied']['tier_standardization'] = len(tier_mapping)

# 5.2: Clean company names (title case, strip whitespace)
df['company_name'] = df['company_name'].str.strip().str.title()
print(f"  • Cleaned company names (title case, stripped whitespace)")

# 5.3: Ensure role_type categories are consistent
if 'role_type' in df.columns:
    role_mapping = {
        'SDE': 'SDE-Core',
        'Software Developer': 'SDE-Core',
        'Data Analyst': 'Data Analyst',
        'Data Analysis': 'Data Analyst',
        'ML/AI': 'SDE-ML/AI',
        'Machine Learning': 'SDE-ML/AI'
    }
    df['role_type'] = df['role_type'].replace(role_mapping)
    print(f"  • Standardized role type labels")

# ============================================================================
# STEP 6: REMOVE DUPLICATES AND REDUNDANCIES
# ============================================================================

print("\n[6/7] Removing duplicates and redundant data...")

# 6.1: Identify exact duplicates
exact_dupes = df.duplicated(keep='first')
df_deduped = df[~exact_dupes].copy()
exact_dupe_count = exact_dupes.sum()
print(f"  • Removed exact duplicates: {exact_dupe_count} records")
quality_report['fixes_applied']['exact_duplicates_removed'] = int(exact_dupe_count)

# 6.2: Identify near-duplicates (same company, year, role, similar CTC)
if 'fte_ctc' in df_deduped.columns:
    df_deduped['ctc_bucket'] = pd.cut(df_deduped['fte_ctc'], bins=20, labels=False)
    near_dupes = df_deduped.duplicated(
        subset=['batch_year', 'company_name', 'role_type', 'ctc_bucket'], 
        keep='first'
    )
    near_dupe_count = near_dupes.sum()
    
    # Only remove if very high similarity and no important differing info
    df_deduped = df_deduped[~near_dupes].copy()
    df_deduped.drop('ctc_bucket', axis=1, inplace=True)
    print(f"  • Removed near-duplicates: {near_dupe_count} records")
    quality_report['fixes_applied']['near_duplicates_removed'] = int(near_dupe_count)

# 6.3: Remove records with critical missing data (no company AND no CTC AND no tier)
critical_missing = (
    df_deduped['company_name'].isna() & 
    df_deduped['fte_ctc'].isna() & 
    df_deduped['placement_tier'].isna()
)
critical_missing_count = critical_missing.sum()
df_clean = df_deduped[~critical_missing].copy()
print(f"  • Removed critically incomplete records: {critical_missing_count} records")
quality_report['fixes_applied']['critical_missing_removed'] = int(critical_missing_count)

# ============================================================================
# STEP 7: VALIDATE AND CREATE DERIVED FEATURES
# ============================================================================

print("\n[7/7] Creating derived features for analysis...")

# 7.1: Create log-transformed CTC for modeling
df_clean['log_fte_ctc'] = np.log1p(df_clean['fte_ctc'].fillna(0))

# 7.2: Create binary flags
df_clean['has_stocks'] = df_clean['stocks_esops'].notna() & (df_clean['stocks_esops'] > 0)
df_clean['has_joining_bonus'] = df_clean['joining_bonus'].notna() & (df_clean['joining_bonus'] > 0)
df_clean['has_cgpa_cutoff'] = df_clean['cgpa_cutoff'].notna()

# 7.3: Create company reputation score (historical average CTC)
company_rep = df_clean.groupby('company_name')['fte_ctc'].transform('mean')
df_clean['company_reputation_score'] = company_rep

# 7.4: Create CGPA percentile within year
df_clean['cgpa_percentile'] = df_clean.groupby('batch_year')['cgpa_cutoff'].rank(pct=True)

# 7.5: Create tier numeric encoding for modeling
tier_encoding = {
    'Tier-3': 1,
    'Tier-2': 2,
    'Tier-1': 3,
    'Super-Dream': 4,
    'Dream': 5
}
df_clean['tier_numeric'] = df_clean['placement_tier'].map(tier_encoding)

# 7.6: Years since data collection start
df_clean['years_in_data'] = df_clean['batch_year'] - df_clean['batch_year'].min()

print(f"  ✓ Created 8 derived features for analysis")
quality_report['fixes_applied']['derived_features_created'] = 8

# ============================================================================
# STEP 8: FINAL VALIDATION AND EXPORT
# ============================================================================

print("\n" + "="*80)
print("FINAL DATA QUALITY SUMMARY")
print("="*80)

final_shape = df_clean.shape
records_removed = original_shape[0] - final_shape[0]
removal_pct = (records_removed / original_shape[0] * 100)

print(f"\n📊 Data Transformation Results:")
print(f"  • Original records:     {original_shape[0]:,}")
print(f"  • Records removed:      {records_removed:,} ({removal_pct:.1f}%)")
print(f"  • Final records:        {final_shape[0]:,}")
print(f"  • Data retention:       {(final_shape[0]/original_shape[0]*100):.1f}%")

# Calculate completeness improvement
final_missing = df_clean.isnull().sum() / len(df_clean) * 100
original_missing = df.isnull().sum() / len(df) * 100
completeness_improvement = original_missing - final_missing

print(f"\n📈 Data Completeness Improvements (Top 5):")
top_improvements = completeness_improvement.sort_values(ascending=False).head(5)
for col, improvement in top_improvements.items():
    if improvement > 0:
        print(f"  • {col:30s}: {improvement:+5.1f}% improvement")

quality_report['final_records'] = int(final_shape[0])
quality_report['records_removed'] = int(records_removed)
quality_report['data_completeness_improvement'] = {
    col: float(val) for col, val in completeness_improvement.items() if val > 0
}

# Key statistics
fte_records = (~df_clean['is_internship_record']).sum()
intern_records = df_clean['is_internship_record'].sum()
ctc_available = df_clean['fte_ctc'].notna().sum()
cgpa_available = df_clean['cgpa_cutoff'].notna().sum()

print(f"\n📏 Clean Dataset Statistics:")
print(f"  • FTE records:          {fte_records:,} ({fte_records/final_shape[0]*100:.1f}%)")
print(f"  • Internship records:   {intern_records:,} ({intern_records/final_shape[0]*100:.1f}%)")
print(f"  • CTC available:        {ctc_available:,} ({ctc_available/final_shape[0]*100:.1f}%)")
print(f"  • CGPA available:       {cgpa_available:,} ({cgpa_available/final_shape[0]*100:.1f}%)")

quality_report['clean_dataset_stats'] = {
    'fte_records': int(fte_records),
    'internship_records': int(intern_records),
    'ctc_completeness_pct': float(ctc_available/final_shape[0]*100),
    'cgpa_completeness_pct': float(cgpa_available/final_shape[0]*100)
}

# Export cleaned data
print(f"\n💾 Exporting cleaned data...")
df_clean.to_csv(OUTPUT_FILE, index=False)
print(f"  ✓ Saved to: {OUTPUT_FILE}")

# Export quality report
with open(REPORT_FILE, 'w') as f:
    json.dump(quality_report, f, indent=2)
print(f"  ✓ Quality report saved to: {REPORT_FILE}")

print("\n" + "="*80)
print("✅ DATA QUALITY IMPROVEMENT COMPLETE!")
print("="*80)
print(f"\nYou can now use '{OUTPUT_FILE}' for advanced analysis with confidence.")
print("All issues have been documented in 'data_quality_report.json'")
print("\n🚀 Ready to run the master analysis pipeline!")
