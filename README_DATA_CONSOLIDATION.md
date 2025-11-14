# PES University Placement Data Consolidation

## Overview

This project consolidates placement data from PES University across multiple batches (2022-2026) and cross-college comparison data into a unified, EDA-ready format suitable for analysis and predictive modeling.

## Data Sources

The consolidation processes data from the following sources:

1. **PES University Year-wise Data (2022-2026)**
   - Tier-based placements (Dream, Tier-1, Tier-2, Tier-3)
   - Internship data (Spring, Summer, PPO)
   - Multiple CSV files per year with varying structures

2. **Cross-College Comparison Data (2024 batch)**
   - PES University
   - RVCE
   - BMS College

## Consolidated Dataset Structure

### Main Consolidated File: `consolidated_placement_data.csv`

**Core Columns:**
- `batch_year` - Academic batch year (2022-2026)
- `college` - College name (PES, RVCE, BMS)
- `company_name` - Name of the recruiting company
- `job_role` - Job role/position

**Placement Details:**
- `placement_tier` - Tier classification (Dream, Tier-1, Tier-2, Tier-3, Internship-Summer, Internship-Spring)
- `placement_type` - Type of placement (FTE, Internship, FTE+Intern, PPO)

**Compensation Details (in LPA unless specified):**
- `total_ctc` - Total Cost to Company
- `base_salary` - Base salary component
- `internship_stipend` - Monthly internship stipend
- `stocks_esops` - Stock/ESOP component
- `joining_bonus` - One-time joining bonus
- `relocation_bonus` - Relocation assistance

**Offer Statistics:**
- `num_offers_fte` - Number of Full-Time Employee offers
- `num_offers_intern` - Number of Internship offers
- `num_offers_both` - Number of FTE+Internship offers
- `num_offers_total` - Total number of offers

**Selection Criteria:**
- `cgpa_cutoff` - Minimum CGPA requirement

**Eligibility:**
- `allows_ece` - Open to ECE students
- `allows_mca` - Open to MCA students
- `allows_mtech` - Open to M.Tech students

**Timeline:**
- `oa_date` - Online Assessment date
- `test_date` - Test date
- `interview_date` - Interview date
- `ppt_date` - Pre-Placement Talk date
- `visit_date` - Company visit date

**Enriched Columns:**
- `has_internship` - Boolean flag for internship stipend availability
- `has_stocks` - Boolean flag for stock/ESOP component
- `has_joining_bonus` - Boolean flag for joining bonus
- `salary_category` - Categorized salary range (Tier-3, Tier-2, Tier-1, Super-Dream, Dream)
- `role_type` - Categorized role type (SDE-Core, SDE-Test, SDE-Data, SDE-ML/AI, SDE-DevOps/SRE, Data Analyst, Business Analyst, Data Scientist, Hardware/Embedded, Intern/Trainee, Other)
- `academic_year` - Full academic year (e.g., 2018-2022 for 2022 batch)

**Metadata:**
- `source_file` - Original CSV filename for traceability
- `additional_info` - Additional comments/information

## Generated Datasets

### By Year
- `placement_data_2022.csv` - 445 records
- `placement_data_2023.csv` - 182 records
- `placement_data_2024.csv` - 604 records (includes cross-college)
- `placement_data_2025.csv` - 351 records
- `placement_data_2026.csv` - 59 records

### By Category
- `placement_data_tier_based.csv` - Only tier-classified placements (1,226 records)
- `placement_data_internships.csv` - Only internship offers (285 records)

### By College
- `placement_data_PES.csv` - PES University data (1,641 records)
- Additional college-specific files if cross-college data is processed

### Summary
- `summary_statistics.json` - Comprehensive statistics and metrics

## Key Statistics

- **Total Records:** 1,641
- **Total Companies:** 1,073 unique recruiters
- **Batches Covered:** 2022-2026 (5 years)
- **Total Placements:** 774 offers

### Compensation Statistics (LPA)
- **Average CTC:** ₹3.72 LPA
- **Median CTC:** ₹0.29 LPA
- **Maximum CTC:** ₹50.00 LPA
- **Minimum CTC:** ₹0.01 LPA
- **Average CGPA Cutoff:** 7.28

### Top 10 Recruiters (by offer count)
1. IBM (General) - 50 offers
2. LTIMindTree - 41 offers
3. Deloitte India - 31 offers
4. IBM (Female Only) - 28 offers
5. Zebra Technologies - 25 offers
6. Apple (India) - 20 offers
7. KPMG - 20 offers
8. PhData - 20 offers
9. EY India - 18 offers
10. Juniper Network - 16 offers

## Data Quality Notes

### Handled Variations
1. **Column Name Variations:** Different years use different column naming conventions (e.g., "Company Name" vs "Company")
2. **Compensation Formats:** Various formats for salary (LPA, lakhs, with/without decimal points, ranges)
3. **Missing Values:** Appropriate handling of null/missing values across all fields
4. **Date Formats:** Multiple date formats standardized
5. **Tier Classifications:** Unified tier naming across different year formats

### Data Cleaning Applied
1. **Company Names:** Standardized to title case, extra spaces removed
2. **Numeric Extraction:** Intelligent parsing of compensation strings (handles "15 LPA", "15L", "15", "1500k", ranges, etc.)
3. **Role Categorization:** Automated categorization of job roles into broader categories
4. **Salary Categories:** CTC-based tier categorization for analysis

### Known Limitations
1. Some records may have incomplete compensation breakdowns (base vs total)
2. Cross-college data integration may need manual verification
3. Historical data (older years) may have fewer columns than recent data
4. Some compensation values include variable components not separately tracked

## Usage for EDA and Modeling

### For Exploratory Data Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load consolidated data
df = pd.read_csv('processed_data/consolidated_placement_data.csv')

# Basic analysis
print(df.info())
print(df.describe())

# Analyze by year
year_stats = df.groupby('batch_year').agg({
    'total_ctc': ['mean', 'median', 'max'],
    'num_offers_total': 'sum',
    'company_name': 'nunique'
})

# Analyze by tier
tier_stats = df.groupby('placement_tier').agg({
    'total_ctc': ['mean', 'median'],
    'cgpa_cutoff': 'mean',
    'company_name': 'count'
})

# Top paying companies
top_companies = df.nlargest(20, 'total_ctc')[['company_name', 'job_role', 'total_ctc', 'batch_year']]

# Role type distribution
role_distribution = df['role_type'].value_counts()

# Compensation trends over years
yearly_comp = df.groupby('batch_year')['total_ctc'].agg(['mean', 'median', 'std'])

# CGPA vs CTC correlation
df.plot.scatter(x='cgpa_cutoff', y='total_ctc', alpha=0.5)
```

### For Predictive Modeling

**Potential Use Cases:**
1. **CTC Prediction:** Predict compensation based on company, role, tier, CGPA cutoff, year
2. **Placement Probability:** Predict likelihood of placement based on historical patterns
3. **Company Categorization:** Classify companies into tiers based on compensation and selection criteria
4. **Trend Analysis:** Forecast future placement statistics and compensation trends
5. **CGPA Requirement Prediction:** Predict CGPA cutoffs for companies
6. **Role Recommendation:** Recommend suitable roles based on compensation expectations and eligibility

**Sample Features for ML Models:**
- `batch_year` - Temporal feature
- `cgpa_cutoff` - Selection criteria
- `placement_tier` - Company tier
- `role_type` - Job category
- `has_internship`, `has_stocks`, `has_joining_bonus` - Binary features
- Company-based features (encoding of `company_name`)
- Lag features from previous years

**Sample Target Variables:**
- `total_ctc` - For regression models
- `salary_category` - For classification models
- `num_offers_total` - For demand prediction

### Example: Simple CTC Prediction Model

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Prepare data
df_model = df[df['total_ctc'].notna()].copy()

# Feature engineering
le_company = LabelEncoder()
le_role = LabelEncoder()

df_model['company_encoded'] = le_company.fit_transform(df_model['company_name'])
df_model['role_type_encoded'] = le_role.fit_transform(df_model['role_type'])

features = ['batch_year', 'cgpa_cutoff', 'company_encoded', 'role_type_encoded',
            'has_internship', 'has_stocks', 'has_joining_bonus']
target = 'total_ctc'

# Remove rows with missing features
df_model = df_model.dropna(subset=features + [target])

X = df_model[features]
y = df_model[target]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train R²: {train_score:.3f}")
print(f"Test R²: {test_score:.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)
```

## Files and Directory Structure

```
ADA-Project/
├── data/                                    # Original CSV files
│   ├── 2022/
│   ├── 2023/
│   ├── 2024/
│   ├── 2025/
│   ├── 2026/
│   └── cross-college-pes-rvce-bms-2025/
├── processed_data/                          # Generated consolidated datasets
│   ├── consolidated_placement_data.csv      # Main consolidated file
│   ├── placement_data_2022.csv
│   ├── placement_data_2023.csv
│   ├── placement_data_2024.csv
│   ├── placement_data_2025.csv
│   ├── placement_data_2026.csv
│   ├── placement_data_PES.csv
│   ├── placement_data_tier_based.csv
│   ├── placement_data_internships.csv
│   └── summary_statistics.json
├── consolidate_placement_data.py            # Main consolidation script
└── README_DATA_CONSOLIDATION.md             # This file
```

## Running the Consolidation Script

To re-run the consolidation (e.g., after adding new data):

```bash
python consolidate_placement_data.py
```

The script will:
1. Process all CSV files in the `data/` directory
2. Handle varying structures and column names
3. Extract and normalize compensation data
4. Generate enriched datasets
5. Save all outputs to `processed_data/`
6. Print summary statistics

## Next Steps for Analysis

1. **Data Exploration:**
   - Load `consolidated_placement_data.csv`
   - Perform univariate and bivariate analysis
   - Identify patterns, trends, and outliers

2. **Visualization:**
   - Create distribution plots for CTC, CGPA, offers
   - Time series analysis of placement trends
   - Company-wise and tier-wise comparisons
   - Role type analysis

3. **Feature Engineering:**
   - Create new features from existing columns
   - Handle missing values appropriately
   - Encode categorical variables

4. **Modeling:**
   - Build regression models for CTC prediction
   - Classification models for tier prediction
   - Time series forecasting for trends
   - Clustering for company segmentation

5. **Insights:**
   - Identify high-paying companies and roles
   - Understand CGPA requirements across tiers
   - Analyze placement trends over years
   - Determine factors influencing compensation

## Support

For issues or questions about the data consolidation:
- Review the `summary_statistics.json` for data quality metrics
- Check `source_file` column to trace back to original data
- Examine the consolidation script for parsing logic

## Version History

- **v1.0** (2025-01-14): Initial consolidation of 2022-2026 placement data
  - 1,641 records from 1,073 unique companies
  - Cross-college comparison data integrated
  - Comprehensive data enrichment and categorization
