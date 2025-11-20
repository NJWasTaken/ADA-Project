"""
Standalone Time Series Forecasting Script
=========================================
Prophet-based CTC forecasting with fallback options

This script:
1. Aggregates yearly CTC trends
2. Trains Prophet model (or ARIMA if Prophet unavailable)
3. Forecasts 2027-2029
4. Creates publication-quality visualizations

Runtime: ~1-2 minutes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300

print("="*80)
print(" "*20 + "TIME SERIES FORECASTING")
print("="*80)

# Create output directory
Path('analysis_outputs/03_timeseries').mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

print("\n[1/4] Loading and preparing data...")
df = pd.read_csv('processed_data/cleaned_placement_data.csv')
df_fte = df[~df['is_internship_record'] & df['fte_ctc'].notna()].copy()

# Aggregate yearly
yearly_ctc = df_fte.groupby('batch_year')['fte_ctc'].agg(['mean', 'median', 'count']).reset_index()
yearly_ctc = yearly_ctc[yearly_ctc['count'] >= 3]  # Minimum 3 records

print(f"  ✓ {len(yearly_ctc)} years of data")
print(f"\n  Historical CTC:")
for _, row in yearly_ctc.iterrows():
    print(f"    {int(row['batch_year'])}: ₹{row['mean']:.2f} LPA (n={int(row['count'])})")

# ============================================================================
# STEP 2: TRAIN MODEL
# ============================================================================

print("\n[2/4] Training forecasting model...")

forecast_results = {}

# Try Prophet first
try:
    from prophet import Prophet
    
    prophet_df = yearly_ctc[['batch_year', 'mean']].rename(columns={'batch_year': 'ds', 'mean': 'y'})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%Y')
    
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.5,
        interval_width=0.95
    )
    model.fit(prophet_df)
    
    # Forecast
    future = model.make_future_dataframe(periods=3, freq='Y')
    forecast = model.predict(future)
    
    forecast_years = forecast[forecast['ds'] > prophet_df['ds'].max()]
    
    print(f"  ✓ Prophet model trained")
    print(f"\n  📊 CTC Forecast (Prophet):")
    for _, row in forecast_years.iterrows():
        year = row['ds'].year
        pred = row['yhat']
        lower = row['yhat_lower']
        upper = row['yhat_upper']
        print(f"    {year}: ₹{pred:.2f} LPA (95% CI: [{lower:.2f}, {upper:.2f}])")
    
    forecast_results = {
        'model': 'Prophet',
        'forecasts': forecast_years[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')
    }
    
    # ============================================================================
    # STEP 3: VISUALIZATIONS
    # ============================================================================
    
    print("\n[3/4] Creating visualizations...")
    
    # Main forecast plot
    fig = model.plot(forecast, figsize=(12, 6))
    plt.title('CTC Forecast 2027-2029 (Prophet)', fontsize=14, fontweight='bold')
    plt.ylabel('Average CTC (LPA)', fontsize=12)
    plt.xlabel('Year', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('analysis_outputs/03_timeseries/forecast_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: forecast_plot.png")
    
    # Components
    fig = model.plot_components(forecast, figsize=(12, 6))
    plt.tight_layout()
    plt.savefig('analysis_outputs/03_timeseries/forecast_components.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: forecast_components.png")
    
except ImportError:
    print("  ℹ️  Prophet not available, using simple linear trend...")
    
    # Fallback: Simple linear regression
    from sklearn.linear_model import LinearRegression
    
    X = yearly_ctc[['batch_year']].values
    y = yearly_ctc['mean'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Forecast
    future_years = np.array([[2027], [2028], [2029]])
    predictions = model.predict(future_years)
    
    # Estimate confidence interval (simple approach)
    residuals = y - model.predict(X)
    std_error = np.std(residuals)
    margin = 1.96 * std_error  # 95% CI
    
    print(f"  ✓ Linear trend model trained")
    print(f"\n  📊 CTC Forecast (Linear Trend):")
    for year, pred in zip([2027, 2028, 2029], predictions):
        print(f"    {year}: ₹{pred:.2f} LPA (95% CI: [{pred-margin:.2f}, {pred+margin:.2f}])")
    
    forecast_results = {
        'model': 'Linear Regression',
        'forecasts': [
            {'year': int(year), 'predicted_ctc': float(pred), 
             'lower_ci': float(pred-margin), 'upper_ci': float(pred+margin)}
            for year, pred in zip([2027, 2028, 2029], predictions)
        ]
    }
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    # Historical data
    plt.scatter(yearly_ctc['batch_year'], yearly_ctc['mean'], 
               s=100, color='blue', edgecolor='black', linewidth=1.5,
               label='Historical Data', zorder=3)
    
    # Trend line
    all_years = np.arange(yearly_ctc['batch_year'].min(), 2030).reshape(-1, 1)
    trend_line = model.predict(all_years)
    plt.plot(all_years, trend_line, 'b--', linewidth=2, alpha=0.7, label='Trend Line')
    
    # Forecasts
    plt.scatter([2027, 2028, 2029], predictions,
               s=100, color='red', edgecolor='black', linewidth=1.5,
               label='Forecast', zorder=3)
    
    # Confidence bands
    plt.fill_between(future_years.flatten(), 
                     predictions - margin, 
                     predictions + margin,
                     alpha=0.2, color='red', label='95% Confidence Interval')
    
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average CTC (LPA)', fontsize=12)
    plt.title('CTC Forecast 2027-2029 (Linear Trend)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('analysis_outputs/03_timeseries/forecast_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: forecast_plot.png")

# ============================================================================
# STEP 4: SAVE RESULTS
# ============================================================================

print("\n[4/4] Saving results...")

with open('analysis_outputs/03_timeseries/forecast_results.json', 'w') as f:
    json.dump(forecast_results, f, indent=2, default=str)
print(f"  ✓ Saved: forecast_results.json")

print("\n" + "="*80)
print("✅ TIME SERIES FORECASTING COMPLETE!")
print("="*80)
print(f"\nModel: {forecast_results['model']}")
print("All forecasts and visualizations saved to analysis_outputs/03_timeseries/")
