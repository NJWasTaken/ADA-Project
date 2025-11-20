"""
Comprehensive Placement Data Merger
Merges ALL placement records including Tier 1/2/3, Internships, PPOs, and Cross-College data
"""

import pandas as pd
import numpy as np
import os
import re
from glob import glob
import warnings
warnings.filterwarnings('ignore')

def extract_salary(value):
    """Extract numeric salary from string, handling k, L, and various formats"""
    if pd.isna(value) or value == '' or value == '-':
        return np.nan

    value_str = str(value).strip()

    # Remove common non-numeric characters
    value_str = re.sub(r'[~\$₹,]', '', value_str)

    # Handle ranges (take average)
    if '-' in value_str and not value_str.startswith('-'):
        parts = value_str.split('-')
        try:
            nums = [float(re.search(r'[\d.]+', p).group()) for p in parts if re.search(r'[\d.]+', p)]
            if nums:
                value_str = str(np.mean(nums))
        except:
            pass

    # Extract number
    match = re.search(r'([\d.]+)\s*([kKlL])?', value_str)
    if match:
        try:
            num = float(match.group(1))
        except ValueError:
            return np.nan
        unit = match.group(2)

        if unit and unit.lower() == 'k':
            return num / 100  # Convert k to lakhs (50k = 0.5L)
        elif unit and unit.lower() == 'l':
            return num
        else:
            # If no unit and number is small, likely in lakhs
            if num < 200:
                return num
            else:
                # If large number, likely in thousands
                return num / 100000

    return np.nan

def categorize_file(filename):
    """Categorize the placement file type"""
    filename_lower = filename.lower()

    # Cross-college files
    if 'cross-college' in filename_lower or 'rvce' in filename_lower or 'bms' in filename_lower:
        return 'Cross-College'

    # Tier files
    if 'tier 1' in filename_lower or 'tier-1' in filename_lower or 'tier1' in filename_lower or 'tier_1' in filename_lower:
        return 'Tier 1'
    elif 'tier 2' in filename_lower or 'tier-2' in filename_lower or 'tier2' in filename_lower or 'tier_2' in filename_lower:
        return 'Tier 2'
    elif 'tier 3' in filename_lower or 'tier-3' in filename_lower or 'tier3' in filename_lower or 'tier_3' in filename_lower:
        return 'Tier 3'
    elif 'dream' in filename_lower:
        return 'Dream'
    # Internship files
    elif 'spring internship' in filename_lower:
        return 'Spring Internship'
    elif 'summer internship' in filename_lower:
        return 'Summer Internship'
    elif 'internship only' in filename_lower:
        return 'Internship Only'
    elif 'ppo' in filename_lower:
        return 'PPO'
    # Default
    else:
        return 'Other'

def robust_csv_read(file_path):
    """Robustly read CSV with various encodings and formats"""
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, header=None)
            return df
        except:
            continue

    # Last resort - read as bytes
    try:
        df = pd.read_csv(file_path, encoding='utf-8', errors='ignore', header=None)
        return df
    except:
        return None

def parse_placement_file(file_path, year, category):
    """Parse a single placement CSV file with robust extraction"""

    try:
        # Read CSV without headers
        df = robust_csv_read(file_path)

        if df is None or len(df) == 0:
            return []

        records = []

        # Find company name column and data columns
        # Usually: Column 0 = row number, Column 1 = company name, rest = data

        for idx in range(len(df)):
            row = df.iloc[idx]

            # Skip completely empty rows
            if row.isna().all():
                continue

            # Try to get company name from column 1 (or column 0 if column 1 is empty)
            company = None

            # Check column 1 first (most common)
            if len(row) > 1 and pd.notna(row.iloc[1]):
                company = str(row.iloc[1]).strip()

            # If not found, try column 0
            if not company or company == '' and len(row) > 0 and pd.notna(row.iloc[0]):
                company = str(row.iloc[0]).strip()

            # Skip if no company or if it's a header-like value
            if not company or company == '':
                continue

            # Skip header rows and metadata
            skip_keywords = ['tier', 'name', 'company', 'job', 'compensation', 'ctc', 'base',
                           'stipend', 'fte', 'ppt', 'test', 'interview', 'cgpa', 'color', 'legend',
                           'note', 'placements', 'batch', 'visited', 'highest', 'average', 'median']

            if any(keyword in company.lower() for keyword in skip_keywords):
                continue

            # Skip numeric-only company names (likely row numbers)
            try:
                float(company)
                continue
            except:
                pass

            # Extract salary information from subsequent columns
            internship_stipend = ''
            base_salary = ''
            total_ctc = ''
            cgpa_cutoff = ''
            job_title = ''

            # Try to extract values from various column positions
            # Different files have different structures, so we try multiple positions

            if len(row) > 2 and pd.notna(row.iloc[2]):
                job_title = str(row.iloc[2]).strip()

            # Look for numeric values in columns 3-10
            for col_idx in range(3, min(len(row), 15)):
                val = row.iloc[col_idx]
                if pd.notna(val):
                    val_str = str(val).strip()
                    # Check if it looks like a salary (contains numbers)
                    if re.search(r'\d', val_str):
                        if not internship_stipend:
                            internship_stipend = val_str
                        elif not base_salary:
                            base_salary = val_str
                        elif not total_ctc:
                            total_ctc = val_str
                        else:
                            break

            # Look for CGPA (usually between 5-10)
            for col_idx in range(3, min(len(row), 20)):
                val = row.iloc[col_idx]
                if pd.notna(val):
                    val_str = str(val).strip()
                    # Check if it's a CGPA-like value
                    try:
                        num = float(val_str)
                        if 4.0 <= num <= 10.0:
                            cgpa_cutoff = val_str
                            break
                    except:
                        pass

            # Create record only if we have at least a company name and some data
            if company:
                record = {
                    'year': year,
                    'category': category,
                    'company_name': company,
                    'job_title': job_title,
                    'internship_stipend_raw': internship_stipend,
                    'base_salary_raw': base_salary,
                    'total_ctc_raw': total_ctc,
                    'cgpa_cutoff_raw': cgpa_cutoff,
                    'fte_only': '',
                    'remarks': ''
                }
                records.append(record)

        return records

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def merge_all_placement_data():
    """Merge all placement data from 2022-2026 including cross-college"""

    all_records = []

    # Define year ranges
    years = [2022, 2023, 2024, 2025, 2026]

    for year in years:
        year_path = f'data/{year}/'

        if not os.path.exists(year_path):
            print(f"⚠️  Year {year} directory not found")
            continue

        # Find all CSV files
        csv_files = glob(f'{year_path}*.csv')

        if not csv_files:
            print(f"⚠️  No CSV files found in: {year_path}")
            continue

        for csv_file in csv_files:
            filename = os.path.basename(csv_file)

            # Categorize the file
            category = categorize_file(filename)

            # Parse ALL files (no skipping)
            records = parse_placement_file(csv_file, year, category)
            print(f"✓ Parsed {len(records)} records from {filename} [{category}]")
            all_records.extend(records)

    # Process cross-college data
    cross_college_path = 'data/cross-college-pes-rvce-bms-2025/'
    if os.path.exists(cross_college_path):
        print(f"\n📊 Processing Cross-College Data...")
        csv_files = glob(f'{cross_college_path}*.csv')

        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            category = 'Cross-College'

            # Determine which college from filename
            if 'pes' in filename.lower():
                category = 'Cross-College-PES'
            elif 'rvce' in filename.lower():
                category = 'Cross-College-RVCE'
            elif 'bms' in filename.lower():
                category = 'Cross-College-BMS'

            records = parse_placement_file(csv_file, 2025, category)
            print(f"✓ Parsed {len(records)} records from {filename} [{category}]")
            all_records.extend(records)

    # Create DataFrame
    df = pd.DataFrame(all_records)

    # Check if we have any data
    if len(df) == 0:
        print("\n⚠️  WARNING: No records were parsed from the CSV files!")
        print("Please check that the data files exist and have the correct format.")
        return df

    # Ensure required columns exist
    required_columns = ['internship_stipend_raw', 'base_salary_raw', 'total_ctc_raw', 'cgpa_cutoff_raw']
    for col in required_columns:
        if col not in df.columns:
            df[col] = ''

    # Convert salary columns to numeric
    print(f"\n💰 Converting salaries to numeric format...")
    df['internship_stipend'] = df['internship_stipend_raw'].apply(extract_salary)
    df['base_salary'] = df['base_salary_raw'].apply(extract_salary)
    df['total_ctc'] = df['total_ctc_raw'].apply(extract_salary)

    # Extract CGPA cutoff
    def extract_cgpa(value):
        if pd.isna(value) or value == '' or value == '-':
            return np.nan
        match = re.search(r'([\d.]+)', str(value))
        if match:
            cgpa = float(match.group(1))
            # Sanity check - CGPA should be between 4 and 10
            if 4.0 <= cgpa <= 10.0:
                return cgpa
        return np.nan

    df['cgpa_cutoff'] = df['cgpa_cutoff_raw'].apply(extract_cgpa)

    # Clean company names
    df['company_name'] = df['company_name'].str.replace(r'\(PPO\)|\(2nd Time\)|\(3rd Time\)', '', regex=True).str.strip()

    # Add derived features
    df['has_internship'] = df['internship_stipend'].notna()
    df['has_base'] = df['base_salary'].notna()
    df['has_ctc'] = df['total_ctc'].notna()
    df['has_cgpa_cutoff'] = df['cgpa_cutoff'].notna()

    # Calculate CTC to base ratio
    df['ctc_to_base_ratio'] = df['total_ctc'] / df['base_salary']

    # Add timestamp
    df['timestamp'] = pd.Timestamp.now()

    # Remove duplicates (same company, year, category with no meaningful data)
    df = df.drop_duplicates(subset=['company_name', 'year', 'category'], keep='first')

    return df

if __name__ == "__main__":
    print("=" * 80)
    print("COMPREHENSIVE PLACEMENT DATA MERGER - PESU 2022-2026")
    print("=" * 80)
    print(f"\n📁 Current working directory: {os.getcwd()}")
    print(f"📁 Looking for data in: {os.path.abspath('data/')}")

    # Check if data directory exists
    if not os.path.exists('data/'):
        print("\n❌ ERROR: 'data/' directory not found!")
        print("   Please make sure you're running this script from the project root directory.")
        exit(1)

    # Merge all data
    merged_df = merge_all_placement_data()

    # Save consolidated dataset
    output_file = 'consolidated_placement_data.csv'
    merged_df.to_csv(output_file, index=False)

    print(f"\n✓ Successfully merged {len(merged_df)} placement records")
    print(f"✓ Saved to: {output_file}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nTotal Records: {len(merged_df)}")
    print(f"\nRecords by Year:")
    print(merged_df['year'].value_counts().sort_index())

    print(f"\nRecords by Category:")
    print(merged_df['category'].value_counts().sort_values(ascending=False))

    print(f"\nTop 15 Companies by Frequency:")
    print(merged_df['company_name'].value_counts().head(15))

    print(f"\nSalary Statistics (in Lakhs):")
    ctc_valid = merged_df['total_ctc'].dropna()
    base_valid = merged_df['base_salary'].dropna()
    if len(ctc_valid) > 0:
        print(f"Total CTC - Mean: ₹{ctc_valid.mean():.2f}L, Median: ₹{ctc_valid.median():.2f}L")
    if len(base_valid) > 0:
        print(f"Base Salary - Mean: ₹{base_valid.mean():.2f}L, Median: ₹{base_valid.median():.2f}L")

    cgpa_valid = merged_df['cgpa_cutoff'].dropna()
    if len(cgpa_valid) > 0:
        print(f"\nCGPA Cutoff Statistics:")
        print(f"Mean: {cgpa_valid.mean():.2f}, Median: {cgpa_valid.median():.2f}")

    print("\n" + "=" * 80)
    print(f"✅ Dataset ready for analysis with {len(merged_df)} total records!")
    print("=" * 80)
