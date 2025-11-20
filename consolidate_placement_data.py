#!/usr/bin/env python3
"""
PES Placement Data Consolidation Script
Consolidates placement data from multiple years (2022-2026) into a clean, analysis-ready format.
"""

import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from typing import Dict, List, Optional


class PlacementDataConsolidator:
    """Consolidates placement data from various sources into unified format."""

    def __init__(self, data_dir: str = "data", output_dir: str = "processed_data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def clean_numeric(self, value: any) -> Optional[float]:
        """Extract numeric value from various formats."""
        if pd.isna(value) or value == '':
            return None

        # Convert to string
        if not isinstance(value, str):
            try:
                return float(value)
            except:
                return None

        value = str(value).strip()
        if value == '' or value.lower() in ['na', 'nan', 'n/a', '-', 'nil', '.']:
            return None

        # Remove common currency symbols and text
        value = re.sub(r'[₹$,\s]', '', value)
        value = re.sub(r'(?i)(lpa|lakh|lakhs|k|per month)', '', value)

        # Handle ranges - take the higher value (more conservative)
        if '-' in value or '–' in value:
            parts = re.split(r'[-–]', value)
            try:
                nums = [float(re.sub(r'[^\d.]', '', p)) for p in parts if re.search(r'\d', p)]
                if nums:
                    return max(nums)  # Take higher value
            except:
                pass

        # Extract first number found
        match = re.search(r'(\d+\.?\d*)', value)
        if match:
            try:
                return float(match.group(1))
            except:
                return None

        return None

    def clean_company_name(self, name: str) -> str:
        """Standardize company names."""
        if pd.isna(name) or name == '':
            return 'Unknown'

        name = str(name).strip()
        # Remove extra whitespace and newlines
        name = re.sub(r'\s+', ' ', name)
        # Title case
        name = name.title()

        # Common standardizations
        replacements = {
            'Ibm': 'IBM',
            'Hp ': 'HP ',
            'Hpe': 'HPE',
            'Sap': 'SAP',
            'Ey ': 'EY ',
            'Ai': 'AI',
            'Aws': 'AWS',
            'Gcp': 'GCP',
            'Iot': 'IoT',
            'Kpmg': 'KPMG',
            'Pwc': 'PwC',
            'Jpmc': 'JPMC',
            'Jpmorgan': 'JPMorgan'
        }

        for old, new in replacements.items():
            name = re.sub(r'\b' + old + r'\b', new, name, flags=re.IGNORECASE)

        return name

    def extract_tier_from_filename(self, filename: str) -> str:
        """Extract tier from filename."""
        filename_lower = filename.lower()

        if 'dream' in filename_lower:
            return 'Dream'
        elif 'tier 1' in filename_lower or 'tier-1' in filename_lower or 'tier1' in filename_lower:
            return 'Tier-1'
        elif 'tier 2' in filename_lower or 'tier-2' in filename_lower or 'tier2' in filename_lower:
            return 'Tier-2'
        elif 'tier 3' in filename_lower or 'tier-3' in filename_lower or 'tier3' in filename_lower:
            return 'Tier-3'
        elif 'spring' in filename_lower:
            return 'Internship-Spring'
        elif 'summer' in filename_lower:
            return 'Internship-Summer'
        elif 'internship' in filename_lower:
            return 'Internship'

        return 'Unknown'

    def is_likely_row_number(self, value: str) -> bool:
        """Check if a value is likely a row number."""
        val_str = str(value).strip()
        # Check if it's just a number (with optional .0)
        if re.match(r'^\d+\.?0*$', val_str):
            return True
        # Check if it matches common row number patterns
        if val_str in ['#', 'No', 'S.No', 'Sr.No']:
            return True
        return False

    def process_file(self, filepath: Path, batch_year: int) -> pd.DataFrame:
        """Process a CSV file and extract placement data."""
        try:
            # Read CSV, skipping initial rows if needed
            df = pd.read_csv(filepath)

            # Try to detect header row
            header_row = 0
            for i in range(min(5, len(df))):
                row_str = ' '.join([str(x).lower() for x in df.iloc[i].values])
                if 'company' in row_str or 'name' in row_str:
                    header_row = i
                    break

            if header_row > 0:
                df = pd.read_csv(filepath, skiprows=header_row)

        except Exception as e:
            print(f"    ✗ Could not read file: {e}")
            return pd.DataFrame()

        tier = self.extract_tier_from_filename(filepath.name)

        records = []

        for idx, row in df.iterrows():
            # Find company name column (usually column 1 or 2, skip row numbers)
            company_name = None
            company_col_idx = None

            for i in range(min(4, len(row))):
                val = str(row.iloc[i]).strip()
                if val and val not in ['', 'nan']:
                    # Skip if it looks like a row number
                    if self.is_likely_row_number(val):
                        continue
                    # Skip header-like values
                    if val.lower() in ['company', 'name', 'company name', 'job title', 'job role']:
                        continue
                    # This is likely the company name
                    company_name = val
                    company_col_idx = i
                    break

            if not company_name or company_name == 'nan':
                continue

            company_name = self.clean_company_name(company_name)
            if company_name == 'Unknown':
                continue

            # Skip header-like company names and job role terms
            if company_name.lower() in ['internship', 'base', 'compensation', 'ctc', 'placed',
                                        'fte', 'intern', 'job title', 'job role', 'company name',
                                        'remarks', 'cgpa', 'cut-off', 'cutoff', 'sde', 'sdet',
                                        'engineer', 'analyst', 'developer', 'trainee']:
                continue

            # Find job role (next non-numeric column after company name)
            job_role = ''
            if company_col_idx is not None and company_col_idx + 1 < len(row):
                jr = str(row.iloc[company_col_idx + 1]).strip()
                if jr and jr not in ['', 'nan'] and not jr.replace('.', '').isdigit():
                    job_role = jr

            # Extract numeric data starting from after job role column
            start_col = (company_col_idx or 1) + 2
            row_data = [self.clean_numeric(x) for x in row.iloc[start_col:start_col+10]]

            # Typical order: internship_stipend, base, ctc, fte, intern, fte+intern, cgpa
            record = {
                'batch_year': batch_year,
                'company_name': company_name,
                'job_role': job_role,
                'tier': tier,
                'internship_stipend_monthly': row_data[0] if len(row_data) > 0 else None,
                'base_salary': row_data[1] if len(row_data) > 1 else None,
                'total_ctc': row_data[2] if len(row_data) > 2 else None,
                'num_fte': row_data[3] if len(row_data) > 3 else None,
                'num_intern': row_data[4] if len(row_data) > 4 else None,
                'num_fte_intern': row_data[5] if len(row_data) > 5 else None,
                'cgpa_cutoff': row_data[6] if len(row_data) > 6 else None,
            }

            records.append(record)

        return pd.DataFrame(records)

    def consolidate_all(self) -> pd.DataFrame:
        """Consolidate all data files."""
        all_records = []

        for year in [2022, 2023, 2024, 2025, 2026]:
            year_dir = self.data_dir / str(year)

            if not year_dir.exists():
                print(f"⚠️  Directory not found: {year_dir}")
                continue

            print(f"\n📂 Processing {year} data...")

            csv_files = list(year_dir.glob("*.csv"))

            for filepath in csv_files:
                try:
                    print(f"  - {filepath.name}")
                    df = self.process_file(filepath, year)
                    if len(df) > 0:
                        all_records.append(df)
                        print(f"    ✓ {len(df)} records")
                    else:
                        print(f"    ⚠ No valid records found")
                except Exception as e:
                    print(f"    ✗ Error: {e}")

        if not all_records:
            raise ValueError("No data was successfully processed!")

        # Combine all records
        combined_df = pd.concat(all_records, ignore_index=True)

        return combined_df

    def clean_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the consolidated data."""
        print("\n🧹 Cleaning and validating data...")

        initial_count = len(df)

        # Remove completely empty rows
        df = df.dropna(how='all')

        # Remove rows without company name
        df = df[df['company_name'].notna() & (df['company_name'] != 'Unknown')]

        # Data type conversions
        numeric_cols = ['internship_stipend_monthly', 'base_salary', 'total_ctc',
                       'num_fte', 'num_intern', 'num_fte_intern', 'cgpa_cutoff']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # If total_ctc is missing but base is present, use base as ctc
        mask = df['total_ctc'].isna() & df['base_salary'].notna()
        df.loc[mask, 'total_ctc'] = df.loc[mask, 'base_salary']

        # Validate CTC ranges
        # Internship stipends might be in monthly format (need conversion)
        # FTE should be annual (LPA)
        is_internship = df['tier'].str.contains('Internship', case=False, na=False)

        # For internships with suspiciously high CTC (>50), likely monthly stipend entered as annual
        # For internships with very low CTC (<0.5), likely monthly stipend in thousands
        df.loc[is_internship & (df['total_ctc'] > 0) & (df['total_ctc'] < 2), 'total_ctc'] = df.loc[is_internship & (df['total_ctc'] > 0) & (df['total_ctc'] < 2), 'total_ctc'] * 12

        # Remove unrealistic values
        df.loc[df['total_ctc'] <= 0, 'total_ctc'] = None  # Changed < to <=
        df.loc[df['total_ctc'] > 250, 'total_ctc'] = None

        # If CTC is way higher than base (> 10x), it's likely data entry error
        # (placement count entered in CTC column)
        suspicious = (df['base_salary'].notna()) & (df['total_ctc'] > df['base_salary'] * 10)
        df.loc[suspicious, 'total_ctc'] = None

        # Validate CGPA (should be between 0 and 10)
        df.loc[(df['cgpa_cutoff'] < 0) | (df['cgpa_cutoff'] > 10), 'cgpa_cutoff'] = None

        # Add derived fields
        df['is_internship'] = is_internship
        df['has_ctc_data'] = df['total_ctc'].notna()
        df['has_base_data'] = df['base_salary'].notna()
        df['has_cgpa_data'] = df['cgpa_cutoff'].notna()

        removed_count = initial_count - len(df)
        print(f"✓ Cleaned: {len(df)} valid records ({removed_count} removed)")

        return df

    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics."""
        # Separate FTE and internship data
        df_fte = df[~df['is_internship'] & df['total_ctc'].notna()]
        df_intern = df[df['is_internship']]

        summary = {
            'total_records': int(len(df)),
            'fte_records': int(len(df_fte)),
            'internship_records': int(len(df_intern)),
            'unique_companies': int(df['company_name'].nunique()),
            'years_covered': sorted(df['batch_year'].unique().astype(int).tolist()),

            'fte_statistics': {
                'mean_ctc': round(float(df_fte['total_ctc'].mean()), 2) if len(df_fte) > 0 else None,
                'median_ctc': round(float(df_fte['total_ctc'].median()), 2) if len(df_fte) > 0 else None,
                'min_ctc': round(float(df_fte['total_ctc'].min()), 2) if len(df_fte) > 0 else None,
                'max_ctc': round(float(df_fte['total_ctc'].max()), 2) if len(df_fte) > 0 else None,
                'std_ctc': round(float(df_fte['total_ctc'].std()), 2) if len(df_fte) > 0 else None,
            },

            'records_by_year': {int(k): int(v) for k, v in df['batch_year'].value_counts().to_dict().items()},
            'records_by_tier': {str(k): int(v) for k, v in df['tier'].value_counts().to_dict().items()},

            'data_completeness': {
                'ctc_completeness': round((df['total_ctc'].notna().sum() / len(df) * 100), 1),
                'base_completeness': round((df['base_salary'].notna().sum() / len(df) * 100), 1),
                'cgpa_completeness': round((df['cgpa_cutoff'].notna().sum() / len(df) * 100), 1),
            },

            'top_10_companies': {str(k): int(v) for k, v in df['company_name'].value_counts().head(10).to_dict().items()}
        }

        return summary

    def save_outputs(self, df: pd.DataFrame):
        """Save all output files."""
        print("\n💾 Saving outputs...")

        # Main consolidated file
        output_path = self.output_dir / 'placement_data.csv'
        df.to_csv(output_path, index=False)
        print(f"✓ Saved: {output_path} ({len(df)} records)")

        # Summary statistics
        summary = self.generate_summary_statistics(df)
        summary_path = self.output_dir / 'summary_statistics.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved: {summary_path}")

        return summary

    def run(self):
        """Run the complete consolidation pipeline."""
        print("=" * 70)
        print("         PES PLACEMENT DATA CONSOLIDATION")
        print("=" * 70)

        # Consolidate
        df = self.consolidate_all()

        # Clean and validate
        df = self.clean_and_validate(df)

        # Save outputs
        summary = self.save_outputs(df)

        # Print summary
        print("\n" + "=" * 70)
        print("         CONSOLIDATION COMPLETE")
        print("=" * 70)
        print(f"\n📊 TOTAL RECORDS: {summary['total_records']:,}")
        print(f"   ├─ FTE Records: {summary['fte_records']:,}")
        print(f"   └─ Internship Records: {summary['internship_records']:,}")
        print(f"\n🏢 UNIQUE COMPANIES: {summary['unique_companies']:,}")
        print(f"\n📅 YEARS COVERED: {', '.join(map(str, summary['years_covered']))}")

        if summary['fte_statistics']['mean_ctc']:
            print(f"\n💰 FTE CTC STATISTICS:")
            print(f"   ├─ Mean:   ₹{summary['fte_statistics']['mean_ctc']:>6.2f} LPA")
            print(f"   ├─ Median: ₹{summary['fte_statistics']['median_ctc']:>6.2f} LPA")
            print(f"   ├─ Min:    ₹{summary['fte_statistics']['min_ctc']:>6.2f} LPA")
            print(f"   └─ Max:    ₹{summary['fte_statistics']['max_ctc']:>6.2f} LPA")

        print(f"\n📈 DATA COMPLETENESS:")
        print(f"   ├─ CTC:  {summary['data_completeness']['ctc_completeness']:>5.1f}%")
        print(f"   ├─ Base: {summary['data_completeness']['base_completeness']:>5.1f}%")
        print(f"   └─ CGPA: {summary['data_completeness']['cgpa_completeness']:>5.1f}%")

        print(f"\n🏆 TOP 5 RECRUITERS:")
        for i, (company, count) in enumerate(list(summary['top_10_companies'].items())[:5], 1):
            print(f"   {i}. {company:<30} {count:>3} placements")

        print("\n" + "=" * 70)


def main():
    """Main entry point."""
    consolidator = PlacementDataConsolidator()
    consolidator.run()


if __name__ == "__main__":
    main()
