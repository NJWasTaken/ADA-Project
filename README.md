# PES Placement Data Analysis

Clean, robust pipeline for analyzing PES University placement data (2022-2026 batches).

## 📁 Project Structure

```
ADA-Project/
├── consolidate_placement_data.py  # Data consolidation pipeline
├── dashboard.py                   # Interactive Streamlit dashboard
├── data/                          # Raw CSV files (2022-2026)
│   ├── 2022/ (6 files)
│   ├── 2023/ (5 files)
│   ├── 2024/ (6 files)
│   ├── 2025/ (4 files)
│   └── 2026/ (5 files)
└── processed_data/                # Generated outputs
    ├── placement_data.csv         # Clean consolidated data
    └── summary_statistics.json    # Summary statistics
```

## 🚀 Quick Start

### 1. Process the Data

```bash
# Install dependencies
pip install pandas numpy

# Run consolidation
python3 consolidate_placement_data.py
```

**Output:**
- `processed_data/placement_data.csv` - Clean dataset (1,902 records)
- `processed_data/summary_statistics.json` - Summary stats

### 2. Launch Dashboard

```bash
# Install dashboard dependencies
pip install streamlit plotly

# Launch interactive dashboard
streamlit run dashboard.py
```

Opens at `http://localhost:8501`

## 📊 Data Statistics

**Current dataset (as of latest consolidation):**

- **Total Records:** 1,902
  - FTE: 1,162
  - Internships: 511
- **Companies:** 1,327 unique
- **Years:** 2022-2026 (5 batches)

**FTE CTC Statistics:**
- Mean: ₹12.51 LPA
- Median: ₹10.00 LPA
- Range: ₹1.00 - ₹83.00 LPA

**Data Completeness:**
- CTC: 70.0%
- Base Salary: 39.6%
- CGPA Cutoffs: 11.9%

**Top Recruiters:**
Oracle, Cisco, Goldman Sachs, Walmart, Infosys, Zscaler, KPMG, HPE

## 🔧 Data Pipeline Features

The consolidation script handles:

✅ **Robust CSV Parsing**
- Handles 5 different year formats
- Automatic header detection
- Intelligent column mapping

✅ **Smart Data Cleaning**
- Filters row numbers (1.0, 2.0, etc.)
- Removes header artifacts
- Standardizes company names
- Validates numeric values

✅ **Data Validation**
- CTC range checks (1-250 LPA)
- CGPA validation (0-10)
- Suspicious value detection
- Internship stipend conversion

✅ **Separation**
- FTE vs Internship records
- Clear boolean flags
- Proper data typing

## 📈 Dashboard Features

**5 Interactive Tabs:**

1. **Overview** - Distribution charts, yearly trends, tier analysis
2. **Companies** - Top recruiters, highest paying, company details
3. **Salary Analysis** - Box plots, percentiles, CGPA vs CTC
4. **Insights** - Key statistics and auto-generated insights
5. **Data Explorer** - Filterable table with CSV export

**Filters:**
- Year selection
- Tier selection
- Company search
- Real-time updates

## 📝 Data Schema

### Consolidated Data (`placement_data.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `batch_year` | int | Batch year (2022-2026) |
| `company_name` | str | Company name (standardized) |
| `job_role` | str | Job role/position |
| `tier` | str | Placement tier (Dream, Tier-1/2/3, Internship-*) |
| `internship_stipend_monthly` | float | Monthly stipend (if internship) |
| `base_salary` | float | Base salary (LPA) |
| `total_ctc` | float | Total CTC (LPA) |
| `num_fte` | int | Number of FTE offers |
| `num_intern` | int | Number of intern offers |
| `num_fte_intern` | int | Number of FTE+Intern offers |
| `cgpa_cutoff` | float | CGPA cutoff (0-10) |
| `is_internship` | bool | Is internship record |
| `has_ctc_data` | bool | Has CTC data |
| `has_base_data` | bool | Has base salary data |
| `has_cgpa_data` | bool | Has CGPA cutoff data |

## 🔍 Usage Examples

### Load and Analyze Data

```python
import pandas as pd

# Load data
df = pd.read_csv('processed_data/placement_data.csv')

# Get FTE records with CTC
fte = df[~df['is_internship'] & df['has_ctc_data']]

# Top 10 companies by average CTC
top_companies = fte.groupby('company_name')['total_ctc'].mean().nlargest(10)
print(top_companies)

# Year-wise trends
yearly_avg = fte.groupby('batch_year')['total_ctc'].mean()
print(yearly_avg)

# Filter by tier
dream_tier = fte[fte['tier'] == 'Dream']
print(f"Dream tier avg: ₹{dream_tier['total_ctc'].mean():.2f} LPA")
```

### Re-run Consolidation

To update the data after adding new CSV files:

```bash
# Add new CSV files to data/YYYY/ directories
# Then re-run consolidation
python3 consolidate_placement_data.py
```

The script will:
1. Process all CSV files in `data/2022-2026/`
2. Clean and validate data
3. Generate new `placement_data.csv`
4. Update `summary_statistics.json`

## 🐛 Known Limitations

- CGPA cutoff data only available for ~12% of records
- Base salary breakdown available for ~40% of records
- Some older records (2022-2023) have less detailed information
- Cross-college data (PES/RVCE/BMS) not currently included

## 📄 License

Internal project - PES University

## 🤝 Contributing

1. Add new CSV files to appropriate `data/YYYY/` directory
2. Run consolidation script
3. Verify data in dashboard
4. Commit changes

---

**Built with:** Python, Pandas, Streamlit, Plotly
**Data Sources:** PES University Placement Records (2022-2026)
