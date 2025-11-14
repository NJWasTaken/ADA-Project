"""
Consolidate PES University Placement Data (2022-2026) and Cross-College Data
This script processes multiple CSV files with varying structures and creates
a unified, EDA-ready dataset for analysis and predictive modeling.
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PlacementDataConsolidator:
    """Consolidates placement data from multiple sources and years"""

    def __init__(self, data_dir: str = '/home/user/ADA-Project/data'):
        self.data_dir = Path(data_dir)
        self.consolidated_data = []

    def extract_numeric(self, value: str) -> Optional[float]:
        """Extract numeric value from string (handles LPA, k, L, etc.)"""
        if pd.isna(value) or value == '' or value == '-' or value == '.':
            return None

        # Convert to string and clean
        value = str(value).strip()

        # Handle ranges (take the higher value)
        if '-' in value and not value.startswith('-'):
            parts = value.split('-')
            value = parts[-1].strip()

        # Remove common text
        value = value.replace('LPA', '').replace('lpa', '').replace('CTC', '').replace('Base', '')
        value = value.replace('Package', '').replace('Lakh', '').replace('L', '')

        # Extract number
        numbers = re.findall(r'[\d.]+', value)
        if numbers:
            try:
                num = float(numbers[0])
                # Handle thousands (k)
                if 'k' in value.lower() or 'K' in value:
                    num = num / 100  # Convert to LPA
                return num
            except ValueError:
                return None
        return None

    def extract_compensation_details(self, comp_str: str) -> Dict:
        """Extract detailed compensation breakdown from string"""
        result = {
            'base_salary': None,
            'total_ctc': None,
            'stocks_esops': None,
            'joining_bonus': None,
            'relocation_bonus': None,
            'variable_pay': None
        }

        if pd.isna(comp_str) or comp_str == '':
            return result

        comp_str = str(comp_str).lower()

        # Extract base
        base_match = re.search(r'(\d+\.?\d*)\s*(?:l\s*)?base', comp_str)
        if base_match:
            result['base_salary'] = float(base_match.group(1))

        # Extract stocks/ESOs
        stock_patterns = [
            r'(\d+\.?\d*)\s*(?:l\s*)?esop',
            r'(\d+\.?\d*)\s*(?:l\s*)?stock',
            r'(\d+\.?\d*)\s*(?:l\s*)?rsu',
            r'(\d+)\s*usd.*(?:stock|rsu)',
            r'(\d+k?)\s*usd'
        ]
        for pattern in stock_patterns:
            match = re.search(pattern, comp_str)
            if match:
                val = match.group(1)
                if 'k' in val:
                    result['stocks_esops'] = float(val.replace('k', '')) / 1000
                else:
                    result['stocks_esops'] = self.extract_numeric(val)
                break

        # Extract joining bonus
        jb_match = re.search(r'(\d+\.?\d*)\s*(?:l\s*)?(?:jb|joining\s*bonus)', comp_str)
        if jb_match:
            result['joining_bonus'] = float(jb_match.group(1))

        # Extract relocation bonus
        rb_match = re.search(r'(\d+\.?\d*)\s*(?:l\s*)?(?:rb|relocation)', comp_str)
        if rb_match:
            result['relocation_bonus'] = float(rb_match.group(1))

        return result

    def parse_2022_files(self, year_dir: Path) -> List[Dict]:
        """Parse 2022 batch data files"""
        records = []

        for file_path in year_dir.glob('*.csv'):
            filename = file_path.name

            # Determine tier from filename
            if 'Dream' in filename:
                tier = 'Dream'
            elif 'Tier-1' in filename or 'Tier 1' in filename:
                tier = 'Tier-1'
            elif 'Tier-2' in filename or 'Tier 2' in filename:
                tier = 'Tier-2'
            elif 'Tier-3' in filename or 'Tier 3' in filename:
                tier = 'Tier-3'
            elif 'Summer' in filename:
                tier = 'Internship-Summer'
            elif 'Spring' in filename:
                tier = 'Internship-Spring'
            else:
                tier = 'Unknown'

            try:
                df = pd.read_csv(file_path, skiprows=lambda x: x in [0, 1])

                for _, row in df.iterrows():
                    if pd.isna(row.get('Name')) or str(row.get('Name')).strip() == '':
                        continue

                    record = {
                        'batch_year': 2022,
                        'college': 'PES',
                        'source_file': filename,
                        'company_name': str(row.get('Name', '')).strip(),
                        'job_role': str(row.get('Job Title', '')).strip(),
                        'placement_tier': tier,
                        'internship_stipend': self.extract_numeric(row.get('Internship Stipend', '')),
                        'base_salary': self.extract_numeric(row.get('Base', '')),
                        'total_ctc': self.extract_numeric(row.get('CTC', '')),
                        'num_offers_fte': self.extract_numeric(row.get('FTE', '')),
                        'num_offers_intern': self.extract_numeric(row.get('Intern', '')),
                        'num_offers_both': self.extract_numeric(row.get('FTE+Intern', '')),
                        'test_date': str(row.get('Test Date', '')),
                        'interview_date': str(row.get('Interview', '')),
                        'ppt_date': str(row.get('PPT', '')),
                        'additional_info': str(row.get('Additional Information', ''))
                    }

                    records.append(record)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        return records

    def parse_2023_files(self, year_dir: Path) -> List[Dict]:
        """Parse 2023 batch data files"""
        records = []

        for file_path in year_dir.glob('*.csv'):
            filename = file_path.name

            # Determine tier from filename
            if 'Dream' in filename:
                tier = 'Dream'
            elif 'Tier 1' in filename:
                tier = 'Tier-1'
            elif 'Tier 2' in filename:
                tier = 'Tier-2'
            elif 'Tier 3' in filename:
                tier = 'Tier-3'
            elif 'Spring' in filename:
                tier = 'Internship-Spring'
            else:
                tier = 'Unknown'

            try:
                # Read with skiprows=1 to skip the first row
                df = pd.read_csv(file_path, skiprows=1)

                # Clean column names
                df.columns = df.columns.str.strip()

                for _, row in df.iterrows():
                    # Try different possible column names for company
                    company = row.get('Company Name', row.get('Company', row.get('Name', '')))

                    # Skip if no company name or if it's a number (row index)
                    if pd.isna(company) or str(company).strip() == '' or str(company).strip() == '#':
                        continue

                    try:
                        int(str(company).strip())
                        continue  # Skip if it's just a number
                    except ValueError:
                        pass  # Not a number, proceed

                    record = {
                        'batch_year': 2023,
                        'college': 'PES',
                        'source_file': filename,
                        'company_name': str(company).strip(),
                        'job_role': str(row.get('Job Title', '')).strip(),
                        'placement_tier': tier,
                        'internship_stipend': self.extract_numeric(row.get('Internship', '')),
                        'base_salary': self.extract_numeric(row.get('Base', '')),
                        'total_ctc': self.extract_numeric(row.get('CTC', '')),
                        'num_offers_fte': self.extract_numeric(row.get('FTE', '')),
                        'num_offers_intern': self.extract_numeric(row.get('Intern', '')),
                        'num_offers_both': self.extract_numeric(row.get('FTE + Intern', row.get('FTE+Intern', ''))),
                        'test_date': str(row.get('Test Date', '')),
                        'interview_date': str(row.get('Interview', '')),
                        'ppt_date': str(row.get('PPT', '')),
                        'additional_info': str(row.get('Additional Comments', ''))
                    }

                    records.append(record)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        return records

    def parse_2024_files(self, year_dir: Path) -> List[Dict]:
        """Parse 2024 batch data files"""
        records = []

        for file_path in year_dir.glob('*.csv'):
            filename = file_path.name

            # Determine tier from filename
            if 'Dream' in filename:
                tier = 'Dream'
            elif 'Tier 1' in filename:
                tier = 'Tier-1'
            elif 'Tier 2' in filename:
                tier = 'Tier-2'
            elif 'Tier 3' in filename:
                tier = 'Tier-3'
            elif 'Spring' in filename:
                tier = 'Internship-Spring'
            elif 'Summer' in filename:
                tier = 'Internship-Summer'
            else:
                tier = 'Unknown'

            try:
                df = pd.read_csv(file_path)

                for _, row in df.iterrows():
                    company = row.get('Company Name', '')
                    if pd.isna(company) or str(company).strip() == '' or str(company).strip() == '#':
                        continue

                    # Extract compensation details
                    comp_str = f"{row.get('Internship', '')} {row.get('Base', '')} {row.get('CTC', '')}"
                    comp_details = self.extract_compensation_details(comp_str)

                    record = {
                        'batch_year': 2024,
                        'college': 'PES',
                        'source_file': filename,
                        'company_name': str(company).strip(),
                        'job_role': str(row.get('Job Title', row.get('Job Role', ''))).strip(),
                        'placement_tier': tier,
                        'internship_stipend': self.extract_numeric(row.get('Internship', '')),
                        'base_salary': self.extract_numeric(row.get('Base', '')) or comp_details['base_salary'],
                        'total_ctc': self.extract_numeric(row.get('CTC', '')),
                        'stocks_esops': comp_details['stocks_esops'],
                        'joining_bonus': comp_details['joining_bonus'],
                        'num_offers_fte': self.extract_numeric(row.get('FTE', '')),
                        'num_offers_intern': self.extract_numeric(row.get('Intern', '')),
                        'num_offers_both': self.extract_numeric(row.get('FTE+Intern', row.get('Both', ''))),
                        'cgpa_cutoff': self.extract_numeric(row.get('CGPA \nCut-off', row.get('CGPA Cut-off', ''))),
                        'additional_info': str(row.get('Remarks', ''))
                    }

                    records.append(record)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        return records

    def parse_2025_files(self, year_dir: Path) -> List[Dict]:
        """Parse 2025 batch data files"""
        records = []

        for file_path in year_dir.glob('*.csv'):
            filename = file_path.name

            # Determine tier from filename
            if 'Tier 1' in filename:
                tier = 'Tier-1'
            elif 'Tier 2' in filename:
                tier = 'Tier-2'
            elif 'Tier 3' in filename:
                tier = 'Tier-3'
            elif 'Spring' in filename:
                tier = 'Internship-Spring'
            elif 'Summer' in filename:
                tier = 'Internship-Summer'
            else:
                tier = 'Unknown'

            try:
                df = pd.read_csv(file_path)

                for _, row in df.iterrows():
                    company = row.get('Company Name', '')
                    if pd.isna(company) or str(company).strip() == '' or str(company).strip() == '#':
                        continue

                    # Extract compensation details
                    ctc_str = str(row.get('Compensation (LPA)', row.get('CTC', '')))
                    comp_details = self.extract_compensation_details(ctc_str)

                    record = {
                        'batch_year': 2025,
                        'college': 'PES',
                        'source_file': filename,
                        'company_name': str(company).strip(),
                        'job_role': str(row.get('Job Role', row.get('Role', ''))).strip(),
                        'placement_tier': tier,
                        'internship_stipend': self.extract_numeric(row.get('Internship', '')),
                        'base_salary': self.extract_numeric(row.get('Base', '')) or comp_details['base_salary'],
                        'total_ctc': self.extract_numeric(row.get('CTC', '')),
                        'stocks_esops': comp_details['stocks_esops'],
                        'joining_bonus': comp_details['joining_bonus'],
                        'num_offers_fte': self.extract_numeric(row.get('FTE Only', row.get('FTE', ''))),
                        'num_offers_intern': self.extract_numeric(row.get('Intern (PBC)', row.get('Intern', ''))),
                        'num_offers_both': self.extract_numeric(row.get('FTE + Intern', row.get('Both', ''))),
                        'cgpa_cutoff': self.extract_numeric(row.get('Final CGPA Cut-off', row.get('GPA Cutoff', ''))),
                        'visit_date': str(row.get('Date of Visit', '')),
                        'additional_info': str(row.get('Note', ''))
                    }

                    records.append(record)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        return records

    def parse_2026_files(self, year_dir: Path) -> List[Dict]:
        """Parse 2026 batch data files"""
        records = []

        for file_path in year_dir.glob('*.csv'):
            filename = file_path.name

            # Determine tier from filename
            if 'Tier 1' in filename:
                tier = 'Tier-1'
            elif 'Tier 2' in filename:
                tier = 'Tier-2'
            elif 'Tier 3' in filename:
                tier = 'Tier-3'
            elif 'Internship' in filename:
                tier = 'Internship'
            elif 'Summer' in filename:
                tier = 'Internship-Summer'
            else:
                tier = 'Unknown'

            try:
                df = pd.read_csv(file_path)

                for _, row in df.iterrows():
                    company = row.get('Company', '')
                    if pd.isna(company) or str(company).strip() == '':
                        continue

                    # Extract compensation details
                    comp_str = f"{row.get('Internship', '')} {row.get('Base', '')} {row.get('CTC', '')} {row.get('Compensation (LPA)', '')}"
                    comp_details = self.extract_compensation_details(comp_str)

                    record = {
                        'batch_year': 2026,
                        'college': 'PES',
                        'source_file': filename,
                        'company_name': str(company).strip(),
                        'job_role': str(row.get('Role', '')).strip(),
                        'placement_tier': tier,
                        'internship_stipend': self.extract_numeric(row.get('Internship', '')),
                        'base_salary': self.extract_numeric(row.get('Base', '')) or comp_details['base_salary'],
                        'total_ctc': self.extract_numeric(row.get('CTC', '')),
                        'stocks_esops': comp_details['stocks_esops'],
                        'joining_bonus': comp_details['joining_bonus'],
                        'num_offers_fte': self.extract_numeric(row.get('FTE', '')),
                        'num_offers_intern': self.extract_numeric(row.get('Intern', row.get('Internship', ''))),
                        'num_offers_both': self.extract_numeric(row.get('Both', '')),
                        'cgpa_cutoff': self.extract_numeric(row.get('GPA Cutoff', '')),
                        'oa_date': str(row.get('OA Date', '')),
                        'additional_info': str(row.get('Note', ''))
                    }

                    records.append(record)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        return records

    def parse_cross_college_files(self, cross_college_dir: Path) -> List[Dict]:
        """Parse cross-college comparison data"""
        records = []

        for file_path in cross_college_dir.glob('*.csv'):
            filename = file_path.name

            # Determine college from filename
            if 'PES' in filename:
                college = 'PES'
            elif 'RVCE' in filename:
                college = 'RVCE'
            elif 'BMS' in filename:
                college = 'BMS'
            else:
                college = 'Unknown'

            try:
                # Read file with skiprows to skip header information
                df = pd.read_csv(file_path, skiprows=lambda x: x in range(9))

                # Clean column names
                df.columns = df.columns.str.strip()

                for _, row in df.iterrows():
                    company = row.get('Company', '')

                    # Skip if no company name or invalid row
                    if pd.isna(company) or str(company).strip() == '':
                        continue

                    # Skip summary rows and headers
                    if any(x in str(company).lower() for x in ['total', 'average', 'median', 'final', 'ppo', 'color', 'blue', 'yellow']):
                        continue

                    # Extract compensation details
                    ctc_str = str(row.get('CTC', ''))
                    comp_details = self.extract_compensation_details(ctc_str)

                    # Determine placement type
                    placement_type = str(row.get('Type', ''))

                    record = {
                        'batch_year': 2024,  # Cross-college data is for 2024 batch (PES data is for 2024)
                        'college': college,
                        'source_file': filename,
                        'company_name': str(company).strip(),
                        'job_role': str(row.get('Role', '')).strip(),
                        'placement_tier': str(row.get('Tier', '')),
                        'placement_type': placement_type,
                        'total_ctc': self.extract_numeric(row.get('CTC', '')),
                        'base_salary': comp_details['base_salary'],
                        'stocks_esops': comp_details['stocks_esops'],
                        'joining_bonus': comp_details['joining_bonus'],
                        'num_offers_total': self.extract_numeric(row.get('Total Offers', row.get('Offers (CSE only)', ''))),
                        'cgpa_cutoff': self.extract_numeric(row.get('CGPA cutoff', '')),
                        'oa_date': str(row.get('Date of OA', '')),
                        'allows_ece': str(row.get('Allows ECE', '')),
                        'allows_mca': str(row.get('MCA', '')),
                        'allows_mtech': str(row.get('MTech (CS)', '')),
                        'additional_info': str(row.get('Any comments/questions/topics asked', row.get('Any comments/questions/topics asked ', '')))
                    }

                    records.append(record)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        return records

    def consolidate_all_data(self):
        """Main method to consolidate all placement data"""
        print("Starting data consolidation...")

        # Process year-wise PES data
        for year in [2022, 2023, 2024, 2025, 2026]:
            year_dir = self.data_dir / str(year)
            if year_dir.exists():
                print(f"\nProcessing {year} batch data...")

                if year == 2022:
                    records = self.parse_2022_files(year_dir)
                elif year == 2023:
                    records = self.parse_2023_files(year_dir)
                elif year == 2024:
                    records = self.parse_2024_files(year_dir)
                elif year == 2025:
                    records = self.parse_2025_files(year_dir)
                elif year == 2026:
                    records = self.parse_2026_files(year_dir)

                print(f"  Found {len(records)} records")
                self.consolidated_data.extend(records)

        # Process cross-college data
        cross_college_dir = self.data_dir / 'cross-college-pes-rvce-bms-2025'
        if cross_college_dir.exists():
            print(f"\nProcessing cross-college data...")
            records = self.parse_cross_college_files(cross_college_dir)
            print(f"  Found {len(records)} records")
            self.consolidated_data.extend(records)

        print(f"\nTotal records consolidated: {len(self.consolidated_data)}")

        # Convert to DataFrame
        df = pd.DataFrame(self.consolidated_data)

        # Calculate total offers where not provided
        df['num_offers_total'] = df.apply(
            lambda row: row.get('num_offers_total') if pd.notna(row.get('num_offers_total'))
            else sum([row.get('num_offers_fte', 0) or 0,
                     row.get('num_offers_intern', 0) or 0,
                     row.get('num_offers_both', 0) or 0]),
            axis=1
        )

        return df

    def clean_and_enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and enrich the consolidated data"""
        print("\nCleaning and enriching data...")

        # Add derived columns
        df['has_internship'] = df['internship_stipend'].notna()
        df['has_stocks'] = df['stocks_esops'].notna()
        df['has_joining_bonus'] = df['joining_bonus'].notna()

        # Categorize companies by compensation
        df['salary_category'] = pd.cut(
            df['total_ctc'],
            bins=[0, 6, 12, 20, 60, float('inf')],
            labels=['Tier-3', 'Tier-2', 'Tier-1', 'Super-Dream', 'Dream']
        )

        # Categorize by role type
        df['role_type'] = df['job_role'].apply(self._categorize_role)

        # Clean company names (remove extra spaces, standardize)
        df['company_name'] = df['company_name'].str.strip().str.title()

        # Add academic year column (e.g., 2022 batch = 2018-2022)
        df['academic_year'] = df['batch_year'].apply(lambda x: f"{x-4}-{x}")

        return df

    def _categorize_role(self, role: str) -> str:
        """Categorize job role into broader categories"""
        if pd.isna(role):
            return 'Unknown'

        role = str(role).lower()

        if any(x in role for x in ['sde', 'software', 'developer', 'engineer']):
            if 'test' in role or 'qa' in role or 'sdet' in role:
                return 'SDE-Test'
            elif 'data' in role:
                return 'SDE-Data'
            elif 'ml' in role or 'ai' in role or 'machine learning' in role:
                return 'SDE-ML/AI'
            elif 'devops' in role or 'sre' in role:
                return 'SDE-DevOps/SRE'
            else:
                return 'SDE-Core'
        elif any(x in role for x in ['analyst', 'business']):
            if 'data' in role:
                return 'Data Analyst'
            else:
                return 'Business Analyst'
        elif any(x in role for x in ['data scientist', 'data science']):
            return 'Data Scientist'
        elif any(x in role for x in ['hardware', 'embedded', 'vlsi', 'digital', 'analog']):
            return 'Hardware/Embedded'
        elif any(x in role for x in ['intern', 'trainee']):
            return 'Intern/Trainee'
        else:
            return 'Other'

    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics"""
        stats = {
            'total_records': len(df),
            'total_companies': df['company_name'].nunique(),
            'batches_covered': sorted(df['batch_year'].unique().tolist()),
            'colleges': df['college'].unique().tolist(),
            'avg_ctc': df['total_ctc'].mean(),
            'median_ctc': df['total_ctc'].median(),
            'max_ctc': df['total_ctc'].max(),
            'min_ctc': df['total_ctc'].min(),
            'total_placements': df['num_offers_total'].sum(),
            'avg_cgpa_cutoff': df['cgpa_cutoff'].mean(),
            'records_by_year': df.groupby('batch_year').size().to_dict(),
            'records_by_tier': df.groupby('placement_tier').size().to_dict(),
            'records_by_college': df.groupby('college').size().to_dict(),
            'top_10_recruiters': df.groupby('company_name')['num_offers_total'].sum().nlargest(10).to_dict()
        }

        return stats

    def save_datasets(self, df: pd.DataFrame, output_dir: str = '/home/user/ADA-Project/processed_data'):
        """Save consolidated datasets"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\nSaving datasets to {output_dir}...")

        # Main consolidated dataset
        df.to_csv(output_path / 'consolidated_placement_data.csv', index=False)
        print(f"  Saved: consolidated_placement_data.csv ({len(df)} records)")

        # Year-wise datasets
        for year in df['batch_year'].unique():
            year_df = df[df['batch_year'] == year]
            year_df.to_csv(output_path / f'placement_data_{year}.csv', index=False)
            print(f"  Saved: placement_data_{year}.csv ({len(year_df)} records)")

        # College-wise datasets (for cross-college analysis)
        for college in df['college'].unique():
            college_df = df[df['college'] == college]
            college_df.to_csv(output_path / f'placement_data_{college}.csv', index=False)
            print(f"  Saved: placement_data_{college}.csv ({len(college_df)} records)")

        # Tier-wise datasets
        tier_df = df[df['placement_tier'].str.contains('Tier', na=False)]
        tier_df.to_csv(output_path / 'placement_data_tier_based.csv', index=False)
        print(f"  Saved: placement_data_tier_based.csv ({len(tier_df)} records)")

        # Internship-only dataset
        intern_df = df[df['has_internship'] == True]
        intern_df.to_csv(output_path / 'placement_data_internships.csv', index=False)
        print(f"  Saved: placement_data_internships.csv ({len(intern_df)} records)")

        print(f"\nAll datasets saved successfully!")


def main():
    """Main execution function"""
    print("=" * 80)
    print("PES University Placement Data Consolidation")
    print("=" * 80)

    # Initialize consolidator
    consolidator = PlacementDataConsolidator()

    # Consolidate all data
    df = consolidator.consolidate_all_data()

    # Clean and enrich
    df = consolidator.clean_and_enrich_data(df)

    # Generate summary statistics
    stats = consolidator.generate_summary_statistics(df)

    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Total Records: {stats['total_records']}")
    print(f"Total Companies: {stats['total_companies']}")
    print(f"Batches Covered: {stats['batches_covered']}")
    print(f"Colleges: {stats['colleges']}")
    print(f"\nCompensation Statistics (LPA):")
    print(f"  Average CTC: {stats['avg_ctc']:.2f}")
    print(f"  Median CTC: {stats['median_ctc']:.2f}")
    print(f"  Maximum CTC: {stats['max_ctc']:.2f}")
    print(f"  Minimum CTC: {stats['min_ctc']:.2f}")
    print(f"\nTotal Placements: {stats['total_placements']:.0f}")
    print(f"Average CGPA Cutoff: {stats['avg_cgpa_cutoff']:.2f}")

    print(f"\nRecords by Year:")
    for year, count in sorted(stats['records_by_year'].items()):
        print(f"  {year}: {count}")

    print(f"\nTop 10 Recruiters:")
    for idx, (company, offers) in enumerate(stats['top_10_recruiters'].items(), 1):
        print(f"  {idx}. {company}: {offers:.0f} offers")

    # Save all datasets
    consolidator.save_datasets(df)

    # Save summary statistics
    import json
    with open('/home/user/ADA-Project/processed_data/summary_statistics.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        stats_serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                             for k, v in stats.items()}
        json.dump(stats_serializable, f, indent=2)

    print("\n" + "=" * 80)
    print("Data consolidation completed successfully!")
    print("=" * 80)

    return df


if __name__ == "__main__":
    df = main()
