"""
Standalone Predictive Modeling Script
======================================
XGBoost + SHAP with comprehensive error handling

This script:
1. Loads cleaned data
2. Builds XGBoost model for CTC prediction
3. Generates SHAP interpretations
4. Creates publication-quality visualizations
5. Saves all results

Runtime: ~2-5 minutes
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
print(" "*20 + "PREDICTIVE MODELING (XGBoost + SHAP)")
print("="*80)

# Create output directory
Path('analysis_outputs/01_predictive').mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n[1/6] Loading data...")
df = pd.read_csv('processed_data/cleaned_placement_data.csv')
df_fte = df[~df['is_internship_record'] & df['fte_ctc'].notna()].copy()
print(f"  ✓ Loaded {len(df_fte):,} FTE records")

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================

print("\n[2/6] Feature engineering...")

# Prepare modeling dataset
df_model = df_fte[df_fte['cgpa_cutoff'].notna()].copy()

# Encode categoricals
from sklearn.preprocessing import LabelEncoder

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

# ============================================================================
# STEP 3: TRAIN MODEL
# ============================================================================

print("\n[3/6] Training XGBoost model...")

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, cross_val_score, KFold
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
    
    # Convert back to original scale
    y_test_original = np.expm1(y_test)
    y_pred_test_original = np.expm1(y_pred_test)
    
    test_rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_test_original))
    test_mae = mean_absolute_error(y_test_original, y_pred_test_original)
    
    print(f"  ✓ Train R²: {train_r2:.4f}")
    print(f"  ✓ Test R²: {test_r2:.4f}")
    print(f"  ✓ Test RMSE: ₹{test_rmse:.2f} LPA")
    print(f"  ✓ Test MAE: ₹{test_mae:.2f} LPA")
    
    # ============================================================================
    # STEP 4: CROSS-VALIDATION
    # ============================================================================
    
    print("\n[4/6] Running cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    print(f"  ✓ CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # ============================================================================
    # STEP 5: SHAP INTERPRETATION
    # ============================================================================
    
    print("\n[5/6] Computing SHAP values...")
    
    try:
        import shap
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # SHAP summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig('analysis_outputs/01_predictive/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: shap_summary.png")
        
        # Feature importance bar plot
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(feature_importance)), feature_importance['importance'], color='steelblue', edgecolor='black')
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('XGBoost Feature Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('analysis_outputs/01_predictive/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: feature_importance.png")
        
    except ImportError:
        print("  ⚠️ SHAP not available, using basic feature importance")
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(feature_importance)), feature_importance['importance'], color='steelblue', edgecolor='black')
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Importance Score', fontsize=12)
        plt.title('XGBoost Feature Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('analysis_outputs/01_predictive/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: feature_importance.png")
    
    # ============================================================================
    # STEP 6: VISUALIZATIONS
    # ============================================================================
    
    print("\n[6/6] Creating visualizations...")
    
    # Actual vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Scatter plot
    axes[0].scatter(y_test_original, y_pred_test_original, alpha=0.5, s=50, edgecolor='black', linewidth=0.5)
    axes[0].plot([y_test_original.min(), y_test_original.max()], 
                 [y_test_original.min(), y_test_original.max()], 
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual CTC (LPA)', fontsize=12)
    axes[0].set_ylabel('Predicted CTC (LPA)', fontsize=12)
    axes[0].set_title(f'Actual vs Predicted CTC (R² = {test_r2:.3f})', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Right: Residual plot
    residuals = y_test_original - y_pred_test_original
    axes[1].scatter(y_pred_test_original, residuals, alpha=0.5, s=50, edgecolor='black', linewidth=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted CTC (LPA)', fontsize=12)
    axes[1].set_ylabel('Residuals (LPA)', fontsize=12)
    axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_outputs/01_predictive/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: actual_vs_predicted.png")
    
    # Save results
    results = {
        'model': 'XGBoost',
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'cv_r2_mean': float(cv_scores.mean()),
        'cv_r2_std': float(cv_scores.std()),
        'n_features': len(feature_cols),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'feature_importance': feature_importance.to_dict('records')
    }
    
    with open('analysis_outputs/01_predictive/model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ PREDICTIVE MODELING COMPLETE!")
    print("="*80)
    print(f"\n📊 Model Performance:")
    print(f"  • Test R² = {test_r2:.4f}")
    print(f"  • Test RMSE = ₹{test_rmse:.2f} LPA")
    print(f"  • Cross-validation R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n🎯 Top 3 Important Features:")
    for i, row in feature_importance.head(3).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")
    
except ImportError as e:
    print(f"\n❌ Error: {e}")
    print("   Please install required libraries:")
    print("   pip install xgboost scikit-learn")
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
