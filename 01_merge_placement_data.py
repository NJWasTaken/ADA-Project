"""
Comprehensive Placement Data Merger
Merges Tier 1, 2, 3 placement records from 2022-2026
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

def parse_placement_file(file_path, year, category):
    """Parse a single placement CSV file"""

    try:
        # Try reading with different encodings
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except:
            df = pd.read_csv(file_path, encoding='latin1')

        # Find the header row (look for common column names)
        header_keywords = ['company', 'name', 'compensation', 'ctc', 'base', 'role', 'job', 'title']
        header_row = 0

        for idx, row in df.iterrows():
            row_str = ' '.join([str(x).lower() for x in row if pd.notna(x)])
            if any(keyword in row_str for keyword in header_keywords):
                header_row = idx
                break

        # Re-read with correct header
        if header_row > 0:
            try:
                df = pd.read_csv(file_path, encoding='utf-8', skiprows=header_row)
            except:
                df = pd.read_csv(file_path, encoding='latin1', skiprows=header_row)

        # Standardize column names
        df.columns = [str(col).strip().lower() for col in df.columns]

        # Map common column variations
        column_mapping = {}
        for col in df.columns:
            if 'company' in col and 'company' not in column_mapping:
                column_mapping[col] = 'company_name'
            elif ('job' in col or 'role' in col or 'title' in col) and 'job_title' not in column_mapping:
                column_mapping[col] = 'job_title'
            elif 'intern' in col and 'stipend' not in col and 'internship_stipend' not in column_mapping:
                column_mapping[col] = 'internship_stipend'
            elif 'base' in col and 'base_salary' not in column_mapping:
                column_mapping[col] = 'base_salary'
            elif ('ctc' in col or 'compensation' in col) and 'total_ctc' not in column_mapping:
                column_mapping[col] = 'total_ctc'
            elif 'cgpa' in col or 'gpa' in col and 'cgpa_cutoff' not in column_mapping:
                column_mapping[col] = 'cgpa_cutoff'
            elif 'fte' in col and 'intern' not in col and 'fte_only' not in column_mapping:
                column_mapping[col] = 'fte_only'

        df = df.rename(columns=column_mapping)

        # Extract relevant columns
        records = []

        for _, row in df.iterrows():
            # Skip empty rows
            if row.isna().all():
                continue

            # Get company name
            company = None
            for col in ['company_name', 'company', 'name']:
                if col in df.columns and pd.notna(row.get(col)) and str(row.get(col)).strip():
                    company = str(row.get(col)).strip()
                    break

            if not company or str(company).lower() in ['', 'nan', 'company', 'none']:
                continue

            # Create record
            record = {
                'year': year,
                'category': category,
                'company_name': company,
                'job_title': str(row.get('job_title', '')).strip() if pd.notna(row.get('job_title')) else '',
                'internship_stipend_raw': str(row.get('internship_stipend', '')).strip() if pd.notna(row.get('internship_stipend')) else '',
                'base_salary_raw': str(row.get('base_salary', '')).strip() if pd.notna(row.get('base_salary')) else '',
                'total_ctc_raw': str(row.get('total_ctc', '')).strip() if pd.notna(row.get('total_ctc')) else '',
                'cgpa_cutoff_raw': str(row.get('cgpa_cutoff', '')).strip() if pd.notna(row.get('cgpa_cutoff')) else '',
                'fte_only': str(row.get('fte_only', '')).strip() if pd.notna(row.get('fte_only')) else '',
                'remarks': str(row.get('note', row.get('remarks', row.get('additional information', '')))).strip() if pd.notna(row.get('note', row.get('remarks', row.get('additional information', '')))) else ''
            }

            records.append(record)

        return records

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def merge_all_placement_data():
    """Merge all placement data from 2022-2026"""

    all_records = []

    # Define year ranges
    years = [2022, 2023, 2024, 2025, 2026]

    for year in years:
        year_path = f'data/{year}/'

        if not os.path.exists(year_path):
            print(f"⚠️  Year {year} directory not found at: {os.path.abspath(year_path)}")
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

    # Create DataFrame
    df = pd.DataFrame(all_records)

    # Check if we have any data
    if len(df) == 0:
        print("\n⚠️  WARNING: No records were parsed from the CSV files!")
        print("Please check that the data files exist and have the correct format.")
        return df

    # Ensure required columns exist (in case some records are missing fields)
    required_columns = ['internship_stipend_raw', 'base_salary_raw', 'total_ctc_raw', 'cgpa_cutoff_raw']
    for col in required_columns:
        if col not in df.columns:
            df[col] = ''

    # Convert salary columns to numeric
    df['internship_stipend'] = df['internship_stipend_raw'].apply(extract_salary)
    df['base_salary'] = df['base_salary_raw'].apply(extract_salary)
    df['total_ctc'] = df['total_ctc_raw'].apply(extract_salary)

    # Extract CGPA cutoff
    def extract_cgpa(value):
        if pd.isna(value) or value == '' or value == '-':
            return np.nan
        match = re.search(r'([\d.]+)', str(value))
        if match:
            return float(match.group(1))
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

    return df

if __name__ == "__main__":
    print("=" * 80)
    print("PLACEMENT DATA MERGER - PESU 2022-2026")
    print("=" * 80)
    print(f"\n📁 Current working directory: {os.getcwd()}")
    print(f"📁 Looking for data in: {os.path.abspath('data/')}")

    # Check if data directory exists
    if not os.path.exists('data/'):
        print("\n❌ ERROR: 'data/' directory not found!")
        print("   Please make sure you're running this script from the project root directory.")
        print(f"   Current directory: {os.getcwd()}")
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

    print(f"\nTop 10 Companies by Frequency:")
    print(merged_df['company_name'].value_counts().head(10))

    print(f"\nSalary Statistics (in Lakhs):")
    print(f"Base Salary - Mean: {merged_df['base_salary'].mean():.2f}, Median: {merged_df['base_salary'].median():.2f}")
    print(f"Total CTC - Mean: {merged_df['total_ctc'].mean():.2f}, Median: {merged_df['total_ctc'].median():.2f}")

    print(f"\nCGPA Cutoff Statistics:")
    print(f"Mean: {merged_df['cgpa_cutoff'].mean():.2f}, Median: {merged_df['cgpa_cutoff'].median():.2f}")

    print("\n" + "=" * 80)
