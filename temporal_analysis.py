"""
Temporal Analysis and Prediction Module for Placement Data
This module provides time series analysis, trend detection, and predictive modeling
for placement data across multiple years.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class TemporalAnalyzer:
    """Main class for temporal analysis and predictions on placement data"""

    def __init__(self, data_path: str = "processed_data/placement_data.csv"):
        """Initialize with placement data"""
        self.df = pd.read_csv(data_path)
        self.df['batch_year'] = pd.to_numeric(self.df['batch_year'], errors='coerce')
        self.fte_df = self.df[~self.df['is_internship']].copy()
        self.intern_df = self.df[self.df['is_internship']].copy()

    def get_yearly_statistics(self) -> pd.DataFrame:
        """Calculate comprehensive yearly statistics"""
        yearly_stats = []

        for year in sorted(self.df['batch_year'].unique()):
            year_data = self.fte_df[self.fte_df['batch_year'] == year]
            year_data_ctc = year_data[year_data['has_ctc_data']]

            stats_dict = {
                'year': int(year),
                'total_placements': len(year_data),
                'unique_companies': year_data['company_name'].nunique(),
                'mean_ctc': year_data_ctc['total_ctc'].mean() if len(year_data_ctc) > 0 else None,
                'median_ctc': year_data_ctc['total_ctc'].median() if len(year_data_ctc) > 0 else None,
                'std_ctc': year_data_ctc['total_ctc'].std() if len(year_data_ctc) > 0 else None,
                'min_ctc': year_data_ctc['total_ctc'].min() if len(year_data_ctc) > 0 else None,
                'max_ctc': year_data_ctc['total_ctc'].max() if len(year_data_ctc) > 0 else None,
                'q25_ctc': year_data_ctc['total_ctc'].quantile(0.25) if len(year_data_ctc) > 0 else None,
                'q75_ctc': year_data_ctc['total_ctc'].quantile(0.75) if len(year_data_ctc) > 0 else None,
                'dream_count': len(year_data[year_data['tier'] == 'Dream']),
                'tier1_count': len(year_data[year_data['tier'] == 'Tier-1']),
                'tier2_count': len(year_data[year_data['tier'] == 'Tier-2']),
                'tier3_count': len(year_data[year_data['tier'] == 'Tier-3']),
            }
            yearly_stats.append(stats_dict)

        return pd.DataFrame(yearly_stats)

    def calculate_yoy_growth(self) -> pd.DataFrame:
        """Calculate year-over-year growth rates"""
        yearly_stats = self.get_yearly_statistics()

        growth_metrics = []
        for i in range(1, len(yearly_stats)):
            prev_year = yearly_stats.iloc[i-1]
            curr_year = yearly_stats.iloc[i]

            growth_dict = {
                'year': int(curr_year['year']),
                'placement_growth_%': ((curr_year['total_placements'] - prev_year['total_placements']) /
                                        prev_year['total_placements'] * 100) if prev_year['total_placements'] > 0 else None,
                'company_growth_%': ((curr_year['unique_companies'] - prev_year['unique_companies']) /
                                      prev_year['unique_companies'] * 100) if prev_year['unique_companies'] > 0 else None,
                'mean_ctc_growth_%': ((curr_year['mean_ctc'] - prev_year['mean_ctc']) /
                                       prev_year['mean_ctc'] * 100) if prev_year['mean_ctc'] else None,
                'median_ctc_growth_%': ((curr_year['median_ctc'] - prev_year['median_ctc']) /
                                         prev_year['median_ctc'] * 100) if prev_year['median_ctc'] else None,
            }
            growth_metrics.append(growth_dict)

        return pd.DataFrame(growth_metrics)

    def analyze_tier_distribution_trends(self) -> Dict:
        """Analyze how tier distributions change over years"""
        tier_trends = {}

        for year in sorted(self.fte_df['batch_year'].unique()):
            year_data = self.fte_df[self.fte_df['batch_year'] == year]
            tier_dist = year_data['tier'].value_counts(normalize=True) * 100
            tier_trends[int(year)] = tier_dist.to_dict()

        return tier_trends

    def analyze_company_hiring_patterns(self, top_n: int = 20) -> pd.DataFrame:
        """Analyze hiring patterns of top companies over years"""
        top_companies = self.fte_df['company_name'].value_counts().head(top_n).index

        company_patterns = []
        for company in top_companies:
            company_data = self.fte_df[self.fte_df['company_name'] == company]
            yearly_hires = company_data.groupby('batch_year').size()

            pattern_dict = {
                'company': company,
                'total_hires': len(company_data),
                'years_active': len(yearly_hires),
                'avg_hires_per_year': len(company_data) / len(yearly_hires),
                'first_year': int(yearly_hires.index.min()),
                'last_year': int(yearly_hires.index.max()),
                'trend': 'increasing' if yearly_hires.iloc[-1] > yearly_hires.iloc[0] else 'decreasing'
                         if len(yearly_hires) > 1 else 'stable',
            }

            # Add year-wise breakdown
            for year in sorted(self.fte_df['batch_year'].unique()):
                pattern_dict[f'hires_{int(year)}'] = yearly_hires.get(year, 0)

            company_patterns.append(pattern_dict)

        return pd.DataFrame(company_patterns)

    def detect_seasonal_patterns(self) -> Dict:
        """Analyze seasonal patterns in internship hiring"""
        if len(self.intern_df) == 0:
            return {"message": "No internship data available"}

        seasonal_patterns = {}

        # Analyze by tier (Spring vs Summer internships)
        for tier in self.intern_df['tier'].unique():
            tier_data = self.intern_df[self.intern_df['tier'] == tier]
            yearly_counts = tier_data.groupby('batch_year').size()

            seasonal_patterns[tier] = {
                'total_count': len(tier_data),
                'avg_per_year': len(tier_data) / len(yearly_counts) if len(yearly_counts) > 0 else 0,
                'yearly_breakdown': yearly_counts.to_dict()
            }

        return seasonal_patterns

    def forecast_ctc_simple(self, forecast_years: int = 2) -> Dict:
        """Simple linear regression forecast for CTC trends"""
        yearly_stats = self.get_yearly_statistics()
        yearly_stats = yearly_stats.dropna(subset=['mean_ctc'])

        if len(yearly_stats) < 2:
            return {"error": "Insufficient data for forecasting"}

        X = yearly_stats['year'].values.reshape(-1, 1)
        y_mean = yearly_stats['mean_ctc'].values
        y_median = yearly_stats['median_ctc'].values

        # Linear regression for mean CTC
        model_mean = LinearRegression()
        model_mean.fit(X, y_mean)

        # Linear regression for median CTC
        model_median = LinearRegression()
        model_median.fit(X, y_median)

        # Make predictions
        last_year = int(yearly_stats['year'].max())
        future_years = np.array([last_year + i for i in range(1, forecast_years + 1)]).reshape(-1, 1)

        mean_predictions = model_mean.predict(future_years)
        median_predictions = model_median.predict(future_years)

        # Calculate confidence intervals (simple approach using std of residuals)
        mean_residuals = y_mean - model_mean.predict(X)
        residual_std = np.std(mean_residuals)

        forecast_results = {
            'model_type': 'Linear Regression',
            'training_years': yearly_stats['year'].tolist(),
            'training_mean_ctc': yearly_stats['mean_ctc'].tolist(),
            'training_median_ctc': yearly_stats['median_ctc'].tolist(),
            'slope_mean': float(model_mean.coef_[0]),
            'intercept_mean': float(model_mean.intercept_),
            'slope_median': float(model_median.coef_[0]),
            'intercept_median': float(model_median.intercept_),
            'r2_score_mean': float(r2_score(y_mean, model_mean.predict(X))),
            'r2_score_median': float(r2_score(y_median, model_median.predict(X))),
            'predictions': []
        }

        for i, year in enumerate(future_years.flatten()):
            forecast_results['predictions'].append({
                'year': int(year),
                'predicted_mean_ctc': float(mean_predictions[i]),
                'predicted_median_ctc': float(median_predictions[i]),
                'confidence_interval_mean': {
                    'lower': float(mean_predictions[i] - 1.96 * residual_std),
                    'upper': float(mean_predictions[i] + 1.96 * residual_std)
                }
            })

        return forecast_results

    def forecast_ctc_polynomial(self, degree: int = 2, forecast_years: int = 2) -> Dict:
        """Polynomial regression forecast for non-linear trends"""
        yearly_stats = self.get_yearly_statistics()
        yearly_stats = yearly_stats.dropna(subset=['mean_ctc'])

        if len(yearly_stats) < degree + 1:
            return {"error": f"Insufficient data for degree {degree} polynomial"}

        X = yearly_stats['year'].values.reshape(-1, 1)
        y_mean = yearly_stats['mean_ctc'].values

        # Polynomial features
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)

        # Fit model
        model = LinearRegression()
        model.fit(X_poly, y_mean)

        # Make predictions
        last_year = int(yearly_stats['year'].max())
        future_years = np.array([last_year + i for i in range(1, forecast_years + 1)]).reshape(-1, 1)
        future_years_poly = poly_features.transform(future_years)

        predictions = model.predict(future_years_poly)

        forecast_results = {
            'model_type': f'Polynomial Regression (degree={degree})',
            'training_years': yearly_stats['year'].tolist(),
            'training_mean_ctc': yearly_stats['mean_ctc'].tolist(),
            'r2_score': float(r2_score(y_mean, model.predict(X_poly))),
            'predictions': []
        }

        for i, year in enumerate(future_years.flatten()):
            forecast_results['predictions'].append({
                'year': int(year),
                'predicted_mean_ctc': float(predictions[i])
            })

        return forecast_results

    def predict_placement_volume(self, forecast_years: int = 2) -> Dict:
        """Predict future placement volumes using trend analysis"""
        yearly_counts = self.fte_df.groupby('batch_year').size().reset_index()
        yearly_counts.columns = ['year', 'placements']

        if len(yearly_counts) < 2:
            return {"error": "Insufficient data for prediction"}

        X = yearly_counts['year'].values.reshape(-1, 1)
        y = yearly_counts['placements'].values

        model = LinearRegression()
        model.fit(X, y)

        last_year = int(yearly_counts['year'].max())
        future_years = np.array([last_year + i for i in range(1, forecast_years + 1)]).reshape(-1, 1)
        predictions = model.predict(future_years)

        result = {
            'model_type': 'Linear Regression',
            'historical_data': yearly_counts.to_dict('records'),
            'trend': 'increasing' if model.coef_[0] > 0 else 'decreasing',
            'avg_yearly_change': float(model.coef_[0]),
            'r2_score': float(r2_score(y, model.predict(X))),
            'predictions': [
                {'year': int(year[0]), 'predicted_placements': int(round(pred))}
                for year, pred in zip(future_years, predictions)
            ]
        }

        return result

    def analyze_ctc_volatility(self) -> Dict:
        """Analyze CTC volatility and stability over years"""
        yearly_stats = self.get_yearly_statistics()
        yearly_stats = yearly_stats.dropna(subset=['mean_ctc', 'std_ctc'])

        volatility_metrics = {
            'coefficient_of_variation_by_year': {},
            'overall_trend_stability': None,
            'most_stable_year': None,
            'most_volatile_year': None
        }

        for _, row in yearly_stats.iterrows():
            cv = (row['std_ctc'] / row['mean_ctc']) * 100 if row['mean_ctc'] > 0 else None
            volatility_metrics['coefficient_of_variation_by_year'][int(row['year'])] = float(cv) if cv else None

        cv_values = [v for v in volatility_metrics['coefficient_of_variation_by_year'].values() if v is not None]
        if cv_values:
            volatility_metrics['overall_trend_stability'] = 'stable' if np.std(cv_values) < 10 else 'volatile'

            cv_dict = {k: v for k, v in volatility_metrics['coefficient_of_variation_by_year'].items() if v is not None}
            volatility_metrics['most_stable_year'] = int(min(cv_dict, key=cv_dict.get))
            volatility_metrics['most_volatile_year'] = int(max(cv_dict, key=cv_dict.get))

        return volatility_metrics

    def identify_emerging_companies(self, recent_years: int = 2) -> List[Dict]:
        """Identify companies that have recently started hiring"""
        all_years = sorted(self.fte_df['batch_year'].unique())
        if len(all_years) < recent_years + 1:
            return []

        recent_year_list = all_years[-recent_years:]
        older_years = all_years[:-recent_years]

        recent_companies = set(self.fte_df[self.fte_df['batch_year'].isin(recent_year_list)]['company_name'].unique())
        older_companies = set(self.fte_df[self.fte_df['batch_year'].isin(older_years)]['company_name'].unique())

        emerging = recent_companies - older_companies

        emerging_list = []
        for company in emerging:
            company_data = self.fte_df[self.fte_df['company_name'] == company]
            company_data_ctc = company_data[company_data['has_ctc_data']]

            emerging_list.append({
                'company': company,
                'first_appearance': int(company_data['batch_year'].min()),
                'total_hires': len(company_data),
                'avg_ctc': float(company_data_ctc['total_ctc'].mean()) if len(company_data_ctc) > 0 else None,
                'tier': company_data['tier'].mode()[0] if len(company_data) > 0 else None
            })

        return sorted(emerging_list, key=lambda x: x['total_hires'], reverse=True)

    def generate_temporal_insights(self) -> Dict:
        """Generate comprehensive temporal insights and predictions"""
        yearly_stats = self.get_yearly_statistics()
        yoy_growth = self.calculate_yoy_growth()
        tier_trends = self.analyze_tier_distribution_trends()
        ctc_forecast = self.forecast_ctc_simple(forecast_years=2)
        volume_forecast = self.predict_placement_volume(forecast_years=2)
        volatility = self.analyze_ctc_volatility()
        emerging = self.identify_emerging_companies(recent_years=2)

        insights = {
            'summary': {
                'total_years_analyzed': len(yearly_stats),
                'year_range': f"{int(yearly_stats['year'].min())}-{int(yearly_stats['year'].max())}",
                'total_placements': int(self.fte_df.shape[0]),
                'avg_placements_per_year': float(self.fte_df.shape[0] / len(yearly_stats)),
            },
            'yearly_statistics': yearly_stats.to_dict('records'),
            'year_over_year_growth': yoy_growth.to_dict('records'),
            'tier_distribution_trends': tier_trends,
            'ctc_forecast': ctc_forecast,
            'placement_volume_forecast': volume_forecast,
            'volatility_analysis': volatility,
            'emerging_companies': emerging[:10],  # Top 10
            'key_insights': []
        }

        # Generate key insights
        if len(yearly_stats) > 1:
            first_year_mean = yearly_stats.iloc[0]['mean_ctc']
            last_year_mean = yearly_stats.iloc[-1]['mean_ctc']
            if first_year_mean and last_year_mean:
                overall_growth = ((last_year_mean - first_year_mean) / first_year_mean) * 100
                insights['key_insights'].append(
                    f"Overall CTC growth from {int(yearly_stats.iloc[0]['year'])} to "
                    f"{int(yearly_stats.iloc[-1]['year'])}: {overall_growth:.2f}%"
                )

        if ctc_forecast.get('predictions'):
            next_year_pred = ctc_forecast['predictions'][0]
            insights['key_insights'].append(
                f"Predicted mean CTC for {next_year_pred['year']}: "
                f"₹{next_year_pred['predicted_mean_ctc']:.2f} LPA"
            )

        if volume_forecast.get('predictions'):
            next_year_vol = volume_forecast['predictions'][0]
            insights['key_insights'].append(
                f"Predicted placements for {next_year_vol['year']}: "
                f"{next_year_vol['predicted_placements']} students"
            )

        if len(emerging) > 0:
            insights['key_insights'].append(
                f"Identified {len(emerging)} emerging companies in recent years"
            )

        return insights


def save_temporal_analysis(output_path: str = "processed_data/temporal_analysis.json"):
    """Run complete temporal analysis and save results"""
    analyzer = TemporalAnalyzer()
    insights = analyzer.generate_temporal_insights()

    # Also get company patterns
    company_patterns = analyzer.analyze_company_hiring_patterns(top_n=20)
    insights['top_company_hiring_patterns'] = company_patterns.to_dict('records')

    # Seasonal patterns
    seasonal = analyzer.detect_seasonal_patterns()
    insights['seasonal_patterns'] = seasonal

    # Save to JSON
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(insights, f, indent=2)

    print(f"Temporal analysis saved to {output_path}")
    print(f"\nKey Insights:")
    for insight in insights['key_insights']:
        print(f"  - {insight}")

    return insights


if __name__ == "__main__":
    # Run temporal analysis
    print("Running Temporal Analysis...")
    print("=" * 60)

    analyzer = TemporalAnalyzer()

    # Generate and save complete analysis
    insights = save_temporal_analysis()

    print(f"\n{'=' * 60}")
    print("Analysis complete!")
