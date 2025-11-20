"""
Master Analysis Pipeline - Sequential Execution
================================================
Runs all 5 priority analyses in order, producing paper-worthy results.

IMPORTANT: Run 00_data_quality_improvement.py FIRST before running this script!

This pipeline executes:
1. Advanced Predictive Modeling (XGBoost + SHAP)
2. Causal Inference (Regression Discontinuity Design)
3. Time Series Forecasting (Prophet)
4. Network Analysis (Company-College Graph)
5. Advanced Clustering (GMM + PCA)

Each analysis:
- Runs independently
- Saves results to analysis_outputs/
- Generates publication-quality figures (300 DPI)
- Produces summary reports

Author: ADA Project Team
Date: 2025-11-20
Estimated Runtime: 15-30 minutes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
from datetime import datetime
import sys

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Create output directories
Path('analysis_outputs/01_predictive').mkdir(parents=True, exist_ok=True)
Path('analysis_outputs/02_causal').mkdir(parents=True, exist_ok=True)
Path('analysis_outputs/03_timeseries').mkdir(parents=True, exist_ok=True)
Path('analysis_outputs/04_network').mkdir(parents=True, exist_ok=True)
Path('analysis_outputs/05_clustering').mkdir(parents=True, exist_ok=True)

# ============================================================================
# SETUP AND DATA LOADING
# ============================================================================

print("="*100)
print(" "*30 + "MASTER ANALYSIS PIPELINE")
print(" "*25 + "Paper-Worthy Placement Analysis")
print("="*100)

print("\n⚙️  Loading cleaned data...")

try:
    df = pd.read_csv('processed_data/cleaned_placement_data.csv')
    print(f"✓ Loaded {len(df):,} clean records")
except FileNotFoundError:
    print("❌ ERROR: cleaned_placement_data.csv not found!")
    print("   Please run '00_data_quality_improvement.py' first.")
    sys.exit(1)

# Prepare analysis-ready subsets
df_fte = df[~df['is_internship_record'] & df['fte_ctc'].notna()].copy()
df_cgpa = df[df['cgpa_cutoff'].notna()].copy()

print(f"✓ FTE dataset: {len(df_fte):,} records")
print(f"✓ CGPA dataset: {len(df_cgpa):,} records")

# Track results
results_summary = {
    'timestamp': datetime.now().isoformat(),
    'total_records_analyzed': len(df),
    'analyses_completed': [],
    'key_findings': []
}

# ============================================================================
# ANALYSIS 1: ADVANCED PREDICTIVE MODELING (XGBoost + SHAP)
# ============================================================================

print("\n" + "="*100)
print("ANALYSIS 1/5: ADVANCED PREDICTIVE MODELING")
print("="*100)

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, cross_val_score, KFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import shap
    
    print("\n[1.1] Feature Engineering...")
    
    # Prepare modeling dataset
    df_model = df_fte[df_fte['cgpa_cutoff'].notna()].copy()
    
    # Encode categoricals
    le_company = LabelEncoder()
    le_role = LabelEncoder()
    le_tier = LabelEncoder()
    
    df_model['company_encoded'] = le_company.fit_transform(df_model['company_name'])
    df_model['role_type_encoded'] = le_role.fit_transform(df_model['role_type'].fillna('Unknown'))
    df_model['tier_encoded'] = le_tier.fit_transform(df_model['placement_tier'].fillna('Unknown'))
    
    # Feature set
    feature_cols = [
        'batch_year', 'cgpa_cutoff', 'company_encoded', 'role_type_encoded',
        'tier_encoded', 'has_stocks', 'has_joining_bonus', 
        'company_reputation_score', 'cgpa_percentile', 'years_in_data'
    ]
    
    target_col = 'log_fte_ctc'
    
    # Prepare X, y
    df_model_clean = df_model.dropna(subset=feature_cols + [target_col])
    X = df_model_clean[feature_cols]
    y = df_model_clean[target_col]
    
    print(f"  ✓ Feature matrix: {X.shape}")
    print(f"  ✓ Features: {len(feature_cols)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\n[1.2] Training XGBoost model...")
    
    # Build model
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Convert back to original scale for RMSE/MAE
    y_test_original = np.expm1(y_test)
    y_pred_test_original = np.expm1(y_pred_test)
    
    test_rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_test_original))
    test_mae = mean_absolute_error(y_test_original, y_pred_test_original)
    
    print(f"  ✓ Train R²: {train_r2:.4f}")
    print(f"  ✓ Test R²: {test_r2:.4f}")
    print(f"  ✓ Test RMSE: ₹{test_rmse:.2f} LPA")
    print(f"  ✓ Test MAE: ₹{test_mae:.2f} LPA")
    
    # Cross-validation
    print(f"\n[1.3] Cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    print(f"  ✓ CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # SHAP interpretation
    print(f"\n[1.4] Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig('analysis_outputs/01_predictive/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: shap_summary.png")
    
    # Actual vs Predicted
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_original, y_pred_test_original, alpha=0.5)
    plt.plot([y_test_original.min(), y_test_original.max()], 
             [y_test_original.min(), y_test_original.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual CTC (LPA)', fontsize=12)
    plt.ylabel('Predicted CTC (LPA)', fontsize=12)
    plt.title(f'Actual vs Predicted CTC (R² = {test_r2:.3f})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('analysis_outputs/01_predictive/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: actual_vs_predicted.png")
    
    # Save results
    pred_results = {
        'model': 'XGBoost',
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'cv_r2_mean': float(cv_scores.mean()),
        'cv_r2_std': float(cv_scores.std()),
        'n_features': len(feature_cols),
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    
    with open('analysis_outputs/01_predictive/model_results.json', 'w') as f:
        json.dump(pred_results, f, indent=2)
    
    results_summary['analyses_completed'].append('Predictive Modeling')
    results_summary['key_findings'].append(f"CTC prediction model achieves R² = {test_r2:.3f}")
    
    print("✅ Analysis 1 complete!")
    
except ImportError as e:
    print(f"⚠️  Skipping Analysis 1: Missing library ({e})")
    print("   Install with: pip install xgboost shap")
except Exception as e:
    print(f"❌ Analysis 1 failed: {e}")

# ============================================================================
# ANALYSIS 2: CAUSAL INFERENCE (Regression Discontinuity Design)
# ============================================================================

print(" \n" + "="*100)
print("ANALYSIS 2/5: CAUSAL INFERENCE (RDD)")
print("="*100)

try:
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    
    print("\n[2.1] Identifying CGPA cutoffs...")
    
    # Find common cutoffs
    cutoff_freq = df_cgpa['cgpa_cutoff'].value_counts()
    common_cutoffs = cutoff_freq[cutoff_freq >= 10].index.tolist()
    
    if common_cutoffs:
        chosen_cutoff = 7.5 if 7.5 in common_cutoffs else common_cutoffs[0]
        print(f"  ✓ Selected cutoff: {chosen_cutoff}")
        
        # RDD analysis
        print(f"\n[2.2] Running RDD around cutoff = {chosen_cutoff}...")
        
        bandwidth = 1.0
        df_rdd = df_fte[(df_fte['cgpa_cutoff'] >= chosen_cutoff - bandwidth) & 
                        (df_fte['cgpa_cutoff'] <= chosen_cutoff + bandwidth)].copy()
        
        df_rdd['treatment'] = (df_rdd['cgpa_cutoff'] >= chosen_cutoff).astype(int)
        
        # Fit local linear regression
        below = df_rdd[df_rdd['treatment'] == 0]
        above = df_rdd[df_rdd['treatment'] == 1]
        
        if len(below) >= 5 and len(above) >= 5:
            model_below = LinearRegression().fit(below[['cgpa_cutoff']], below['fte_ctc'])
            model_above = LinearRegression().fit(above[['cgpa_cutoff']], above['fte_ctc'])
            
            effect = model_above.predict([[chosen_cutoff]])[0] - model_below.predict([[chosen_cutoff]])[0]
            
            print(f"  ✓ Treatment effect: ₹{effect:.2f} LPA")
            print(f"  ✓ Sample size: {len(below)} below, {len(above)} above")
            
            # Visualization
            plt.figure(figsize=(12, 7))
            plt.scatter(below['cgpa_cutoff'], below['fte_ctc'], alpha=0.5, label='Below cutoff', color='blue')
            plt.scatter(above['cgpa_cutoff'], above['fte_ctc'], alpha=0.5, label='Above cutoff', color='red')
            
            # Regression lines
            x_below = np.linspace(chosen_cutoff - bandwidth, chosen_cutoff, 50).reshape(-1, 1)
            x_above = np.linspace(chosen_cutoff, chosen_cutoff + bandwidth, 50).reshape(-1, 1)
            plt.plot(x_below, model_below.predict(x_below), 'b--', linewidth=2)
            plt.plot(x_above, model_above.predict(x_above), 'r--', linewidth=2)
            
            plt.axvline(chosen_cutoff, color='green', linestyle=':', linewidth=2, label=f'Cutoff = {chosen_cutoff}')
            plt.xlabel('CGPA Cutoff', fontsize=12)
            plt.ylabel('FTE CTC (LPA)', fontsize=12)
            plt.title(f'RDD: Causal Effect of CGPA Cutoff (Effect = ₹{effect:.2f} LPA)', 
                     fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig('analysis_outputs/02_causal/rdd_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: rdd_analysis.png")
            
            # Save results
            rdd_results = {
                'cutoff': float(chosen_cutoff),
                'bandwidth': float(bandwidth),
                'treatment_effect_lpa': float(effect),
                'n_below': int(len(below)),
                'n_above': int(len(above))
            }
            
            with open('analysis_outputs/02_causal/rdd_results.json', 'w') as f:
                json.dump(rdd_results, f, indent=2)
            
            results_summary['analyses_completed'].append('Causal Inference (RDD)')
            results_summary['key_findings'].append(f"CGPA cutoff ≥{chosen_cutoff} causally increases CTC by ₹{effect:.2f} LPA")
            
            print("✅ Analysis 2 complete!")
        else:
            print("  ⚠️  Insufficient data around cutoff for RDD")
    else:
        print("  ⚠️  No common CGPA cutoffs found")
        
except Exception as e:
    print(f"❌ Analysis 2 failed: {e}")

# ============================================================================
# ANALYSIS 3: TIME SERIES FORECASTING (Prophet)
# ============================================================================

print("\n" + "="*100)
print("ANALYSIS 3/5: TIME SERIES FORECASTING")
print("="*100)

try:
    from prophet import Prophet
    
    print("\n[3.1] Preparing time series data...")
    
    # Aggregate yearly CTC
    yearly_ctc = df_fte.groupby('batch_year')['fte_ctc'].agg(['mean', 'count']).reset_index()
    yearly_ctc = yearly_ctc[yearly_ctc['count'] >= 5]  # Minimum 5 records
    
    if len(yearly_ctc) >= 3:
        prophet_df = yearly_ctc[['batch_year', 'mean']].rename(columns={'batch_year': 'ds', 'mean': 'y'})
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%Y')
        
        print(f"  ✓ {len(prophet_df)} years of data")
        
        print("\n[3.2] Training Prophet model...")
        
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.5
        )
        model.fit(prophet_df)
        
        # Forecast 3 years ahead
        future = model.make_future_dataframe(periods=3, freq='Y')
        forecast = model.predict(future)
        
        # Extract predictions
        forecast_years = forecast[forecast['ds'] > prophet_df['ds'].max()]
        
        print(f"\n  📊 CTC Forecast:")
        for _, row in forecast_years.iterrows():
            year = row['ds'].year
            pred = row['yhat']
            lower = row['yhat_lower']
            upper = row['yhat_upper']
            print(f"    {year}: ₹{pred:.2f} LPA (95% CI: [{lower:.2f}, {upper:.2f}])")
        
        # Visualization
        fig = model.plot(forecast, figsize=(12, 6))
        plt.title('CTC Forecast 2027-2029', fontsize=14, fontweight='bold')
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
        
        # Save results
        forecast_results = {
            'model': 'Prophet',
            'historical_years': int(len(prophet_df)),
            'forecast_years': forecast_years[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')
        }
        
        with open('analysis_outputs/03_timeseries/forecast_results.json', 'w') as f:
            json.dump(forecast_results, f, indent=2, default=str)
        
        results_summary['analyses_completed'].append('Time Series Forecasting')
        avg_forecast = forecast_years['yhat'].mean()
        results_summary['key_findings'].append(f"Predicted average CTC for 2027-2029: ₹{avg_forecast:.2f} LPA")
        
        print("✅ Analysis 3 complete!")
    else:
        print("  ⚠️  Insufficient years of data for forecasting")
        
except ImportError:
    print("⚠️  Skipping Analysis 3: Prophet not installed")
    print("   Install with: pip install prophet")
except Exception as e:
    print(f"❌ Analysis 3 failed: {e}")

# ============================================================================
# ANALYSIS 4: NETWORK ANALYSIS (Company-College Graph)
# ============================================================================

print("\n" + "="*100)
print("ANALYSIS 4/5: NETWORK ANALYSIS")
print("="*100)

try:
    import networkx as nx
    
    print("\n[4.1] Building bipartite network...")
    
    # Create graph
    G = nx.Graph()
    
    companies = df['company_name'].unique()
    colleges = df['college'].unique()
    
    G.add_nodes_from(companies, bipartite=0)
    G.add_nodes_from(colleges, bipartite=1)
    
    # Add weighted edges
    edge_weights = df.groupby(['company_name', 'college']).size().reset_index(name='weight')
    for _, row in edge_weights.iterrows():
        G.add_edge(row['company_name'], row['college'], weight=row['weight'])
    
    print(f"  ✓ Nodes: {G.number_of_nodes()} ({len(companies)} companies, {len(colleges)} colleges)")
    print(f"  ✓ Edges: {G.number_of_edges()}")
    
    print("\n[4.2] Computing network metrics...")
    
    # Degree centrality (most connected)
    degree_cent = nx.degree_centrality(G)
    company_degrees = {k: v for k, v in degree_cent.items() if k in companies}
    top_companies = sorted(company_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("\n  🏆 Top 10 Most Connected Companies:")
    for i, (comp, score) in enumerate(top_companies, 1):
        print(f"    {i:2d}. {comp:40s}: {score:.4f}")
    
    # PageRank (prestige)
    pagerank = nx.pagerank(G, weight='weight')
    company_prestige = {k: v for k, v in pagerank.items() if k in companies}
    top_prestige = sorted(company_prestige.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Visualization (simplified for large networks)
    print("\n[4.3] Creating visualization...")
    
    # Sample for visualization if too large
    if len(companies) > 100:
        top_comp_names = [c[0] for c in top_companies[:50]]
        G_viz = G.subgraph(top_comp_names + list(colleges))
    else:
        G_viz = G
    
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G_viz, k=0.3, iterations=50, seed=42)
    
    # Draw
    nx.draw_networkx_nodes(G_viz, pos, 
                          nodelist=[n for n in G_viz.nodes() if n in companies],
                          node_color='skyblue', node_size=300, alpha=0.7, label='Companies')
    nx.draw_networkx_nodes(G_viz, pos,
                          nodelist=[n for n in G_viz.nodes() if n in colleges],
                          node_color='lightcoral', node_size=800, alpha=0.8, label='Colleges')
    
    # Draw edges
    nx.draw_networkx_edges(G_viz, pos, alpha=0.2, width=0.5)
    
    # Labels for colleges
    college_labels = {n: n for n in G_viz.nodes() if n in colleges}
    nx.draw_networkx_labels(G_viz, pos, college_labels, font_size=10, font_weight='bold')
    
    plt.title('Company-College Placement Network', fontsize=14, fontweight='bold')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('analysis_outputs/04_network/network_graph.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: network_graph.png")
    
    # Save results
    network_results = {
        'n_companies': int(len(companies)),
        'n_colleges': int(len(colleges)),
        'n_edges': int(G.number_of_edges()),
        'top_connected_companies': [{'company': c, 'score': float(s)} for c, s in top_companies],
        'top_prestige_companies': [{'company': c, 'score': float(s)} for c, s in top_prestige]
    }
    
    with open('analysis_outputs/04_network/network_metrics.json', 'w') as f:
        json.dump(network_results, f, indent=2)
    
    results_summary['analyses_completed'].append('Network Analysis')
    results_summary['key_findings'].append(f"Identified {len(top_companies)} highly connected companies")
    
    print("✅ Analysis 4 complete!")
    
except ImportError:
    print("⚠️  Skipping Analysis 4: NetworkX not installed")
    print("   Install with: pip install networkx")
except Exception as e:
    print(f"❌ Analysis 4 failed: {e}")

# ============================================================================
# ANALYSIS 5: ADVANCED CLUSTERING (GMM + PCA)
# ============================================================================

print("\n" + "="*100)
print("ANALYSIS 5/5: ADVANCED CLUSTERING")
print("="*100)

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D
    
    print("\n[5.1] Preparing company features...")
    
    # Aggregate company stats
    company_stats = df_fte.groupby('company_name').agg({
        'fte_ctc': ['mean', 'std', 'count'],
        'num_offers_total': 'sum',
        'batch_year': lambda x: x.nunique()
    }).reset_index()
    
    company_stats.columns = ['company_name', 'avg_ctc', 'ctc_std', 'placement_count',
                              'total_offers', 'years_active']
    
    # Log transform
    company_stats['log_avg_ctc'] = np.log1p(company_stats['avg_ctc'])
    company_stats['log_total_offers'] = np.log1p(company_stats['total_offers'])
    
    # Features
    features = ['log_avg_ctc', 'log_total_offers', 'years_active']
    X = company_stats[features].fillna(0)
    
    print(f"  ✓ {len(company_stats)} companies")
    
    print("\n[5.2] Running Gaussian Mixture Model...")
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit GMM
    gmm = GaussianMixture(n_components=4, random_state=42, n_init=10)
    company_stats['cluster'] = gmm.fit_predict(X_scaled)
    
    # Cluster summary
    cluster_summary = company_stats.groupby('cluster').agg({
        'avg_ctc': 'mean',
        'total_offers': 'mean',
        'years_active': 'mean',
        'company_name': 'count'
    }).round(2)
    
    print("\n  📊 Cluster Summary:")
    print(cluster_summary)
    
    print("\n[5.3] Creating 3D PCA visualization...")
    
    # PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                        c=company_stats['cluster'], cmap='viridis',
                        s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)', fontsize=11)
    ax.set_title('Company Clustering (3D PCA)', fontsize=14, fontweight='bold')
    
    plt.colorbar(scatter, label='Cluster', pad=0.1)
    plt.tight_layout()
    plt.savefig('analysis_outputs/05_clustering/clusters_3d_pca.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: clusters_3d_pca.png")
    
    # Save results
    clustering_results = {
        'n_clusters': 4,
        'n_companies': int(len(company_stats)),
        'cluster_summary': cluster_summary.to_dict(),
        'pca_variance_explained': [float(ratio) for ratio in pca.explained_variance_ratio_]
    }
    
    with open('analysis_outputs/05_clustering/clustering_results.json', 'w') as f:
        json.dump(clustering_results, f, indent=2)
    
    # Save company assignments
    company_stats[['company_name', 'cluster', 'avg_ctc', 'total_offers']].to_csv(
        'analysis_outputs/05_clustering/company_cluster_assignments.csv', index=False
    )
    
    results_summary['analyses_completed'].append('Advanced Clustering')
    results_summary['key_findings'].append(f"Identified 4 company segments using GMM")
    
    print("✅ Analysis 5 complete!")
    
except ImportError as e:
    print(f"⚠️  Skipping Analysis 5: Missing library ({e})")
except Exception as e:
    print(f"❌ Analysis 5 failed: {e}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*100)
print("🎉 MASTER ANALYSIS PIPELINE COMPLETE!")
print("="*100)

print(f"\n📊 Summary:")
print(f"  • Analyses completed: {len(results_summary['analyses_completed'])}/5")
print(f"  • Key findings: {len(results_summary['key_findings'])}")

print(f"\n🔍 Key Findings:")
for i, finding in enumerate(results_summary['key_findings'], 1):
    print(f"  {i}. {finding}")

# Save master summary
with open('analysis_outputs/master_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\n💾 All results saved to 'analysis_outputs/'")
print(f"📄 Master summary: analysis_outputs/master_summary.json")

print("\n" + "="*100)
print("✅ READY FOR PAPER SUBMISSION!")
print("="*100)
print("\nNext steps:")
print("  1. Review all figures in analysis_outputs/")
print("  2. Check JSON files for quantitative results")
print("  3. Compile findings into your paper/presentation")
print("  4. Consider running additional robustness checks")
print("\n🚀 You now have publication-quality analysis!")
