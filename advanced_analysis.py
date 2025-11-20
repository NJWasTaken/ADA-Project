"""
Advanced Analysis Pipeline for ADA Project
Focus: NLP on Job Roles, Company Clustering, and Statistical Deep Dives
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Create output directories
Path('analysis_outputs/nlp').mkdir(parents=True, exist_ok=True)
Path('analysis_outputs/clustering').mkdir(parents=True, exist_ok=True)
Path('analysis_outputs/statistical').mkdir(parents=True, exist_ok=True)

def load_data():
    print("Loading consolidated data...")
    df = pd.read_csv('processed_data/consolidated_placement_data.csv')
    print(f"Loaded {len(df)} records.")
    return df

# ============================================================================
# 1. NLP & TEXT ANALYSIS ON JOB ROLES
# ============================================================================
def analyze_job_roles(df):
    print("\n" + "="*50)
    print("Performing NLP on Job Roles...")
    print("="*50)
    
    # 1. Preprocessing
    roles = df['job_role'].dropna().astype(str).str.lower()
    # Remove common stopwords specific to this domain
    stop_words = ['role', 'job', 'title', 'position', 'engineer', 'developer', 'senior', 'junior', 'sr', 'jr', 'ii', 'iii', '1', '2', '3']
    
    # 2. Word Cloud
    text = ' '.join(roles)
    wordcloud = WordCloud(width=1600, height=800, background_color='white', 
                          stopwords=stop_words, colormap='viridis').generate(text)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Job Role Word Cloud', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('analysis_outputs/nlp/job_role_wordcloud.png', dpi=300)
    plt.close()
    print("  ✓ Saved: job_role_wordcloud.png")
    
    # 3. N-gram Analysis (Bigrams)
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(2, 2), min_df=5)
    X = vectorizer.fit_transform(roles)
    counts = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    top_bigrams = counts.sum().sort_values(ascending=False).head(15)
    
    plt.figure(figsize=(12, 6))
    top_bigrams.plot(kind='barh', color='teal', edgecolor='black')
    plt.title('Top 15 Job Role Bigrams', fontsize=16, fontweight='bold')
    plt.xlabel('Frequency')
    plt.tight_layout()
    plt.savefig('analysis_outputs/nlp/top_bigrams.png', dpi=300)
    plt.close()
    print("  ✓ Saved: top_bigrams.png")
    
    # 4. Domain Classification (Rule-based)
    domains = {
        'Software/Web': ['software', 'developer', 'web', 'full stack', 'frontend', 'backend', 'sde', 'java', 'python'],
        'Data/AI': ['data', 'analyst', 'scientist', 'machine learning', 'ai', 'ml', 'intelligence', 'analytics'],
        'Cloud/DevOps': ['cloud', 'devops', 'sre', 'reliability', 'infrastructure', 'aws', 'azure'],
        'Embedded/Hardware': ['embedded', 'hardware', 'vlsi', 'electronics', 'analog', 'digital', 'firmware'],
        'Testing/QA': ['test', 'qa', 'quality', 'automation', 'sdet'],
        'Management/Business': ['business', 'manager', 'product', 'consultant', 'associate', 'management']
    }
    
    def classify_domain(role):
        role = role.lower()
        for domain, keywords in domains.items():
            if any(k in role for k in keywords):
                return domain
        return 'Other'
    
    df['domain_category'] = df['job_role'].apply(classify_domain)
    
    # Domain Distribution
    domain_counts = df['domain_category'].value_counts()
    
    plt.figure(figsize=(10, 10))
    plt.pie(domain_counts, labels=domain_counts.index, autopct='%1.1f%%', startangle=140, 
            colors=sns.color_palette('pastel'))
    plt.title('Job Roles by Domain Category', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('analysis_outputs/nlp/domain_distribution.png', dpi=300)
    plt.close()
    print("  ✓ Saved: domain_distribution.png")
    
    return df

# ============================================================================
# 2. COMPANY CLUSTERING (SEGMENTATION)
# ============================================================================
def cluster_companies(df):
    print("\n" + "="*50)
    print("Performing Company Clustering...")
    print("="*50)
    
    # Feature Engineering per Company
    # We need to handle missing CTCs carefully. We'll use median per company.
    
    company_stats = df.groupby('company_name').agg({
        'total_ctc': 'median',
        'num_offers_total': 'sum',
        'batch_year': 'count'  # Frequency of visits (proxy)
    }).reset_index()
    
    # Filter companies with valid CTC and at least some activity
    company_stats = company_stats.dropna(subset=['total_ctc'])
    company_stats = company_stats[company_stats['total_ctc'] > 0]
    
    # Features for clustering
    features = ['total_ctc', 'num_offers_total']
    X = company_stats[features]
    
    # Log transform to handle skewness (CTC and Offers are highly skewed)
    X_log = np.log1p(X)
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    
    # K-Means Clustering
    # Determine optimal K (Elbow method skipped for brevity, assuming 4 clusters: Mass, Mid, High, Elite)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    company_stats['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Interpret Clusters
    cluster_summary = company_stats.groupby('cluster')[features].mean().sort_values('total_ctc')
    print("\nCluster Centers (Mean Values):")
    print(cluster_summary)
    
    # Map cluster IDs to meaningful names based on CTC
    # Sort clusters by CTC to assign names
    sorted_clusters = cluster_summary.index.tolist()
    cluster_map = {
        sorted_clusters[0]: 'Mass/Entry',
        sorted_clusters[1]: 'Mid-Tier',
        sorted_clusters[2]: 'High-Growth',
        sorted_clusters[3]: 'Elite/Dream'
    }
    company_stats['segment'] = company_stats['cluster'].map(cluster_map)
    
    # Visualization
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=company_stats, x='num_offers_total', y='total_ctc', 
                    hue='segment', style='segment', s=100, palette='deep', alpha=0.8)
    
    # Annotate some top companies
    top_companies = company_stats.nlargest(10, 'total_ctc')
    for _, row in top_companies.iterrows():
        plt.text(row['num_offers_total'], row['total_ctc'], row['company_name'], 
                 fontsize=8, ha='left', va='bottom')
                 
    mass_recruiters = company_stats.nlargest(5, 'num_offers_total')
    for _, row in mass_recruiters.iterrows():
        plt.text(row['num_offers_total'], row['total_ctc'], row['company_name'], 
                 fontsize=8, ha='right', va='top')

    plt.xscale('log')
    plt.yscale('log')
    plt.title('Company Segmentation: CTC vs Hiring Volume (Log Scale)', fontsize=16, fontweight='bold')
    plt.xlabel('Total Offers (Log Scale)')
    plt.ylabel('Median CTC (Log Scale)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig('analysis_outputs/clustering/company_segmentation.png', dpi=300)
    plt.close()
    print("  ✓ Saved: company_segmentation.png")
    
    # Save clustered data
    company_stats.to_csv('analysis_outputs/clustering/company_clusters.csv', index=False)
    print("  ✓ Saved: company_clusters.csv")
    
    return company_stats

# ============================================================================
# 3. STATISTICAL DEEP DIVE
# ============================================================================
def statistical_deep_dive(df):
    print("\n" + "="*50)
    print("Performing Statistical Deep Dive...")
    print("="*50)
    
    # 1. College Comparison (PES vs RVCE vs BMS) - 2024/2025 Data
    # Filter for relevant years and colleges
    cross_df = df[df['college'].isin(['PES', 'RVCE', 'BMS']) & (df['fte_ctc'].notna())]
    
    if len(cross_df['college'].unique()) > 1:
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=cross_df, x='college', y='fte_ctc', inner='quartile', palette='Set2')
        plt.title('CTC Distribution by College', fontsize=16, fontweight='bold')
        plt.ylabel('FTE CTC (LPA)')
        plt.tight_layout()
        plt.savefig('analysis_outputs/statistical/college_ctc_violin.png', dpi=300)
        plt.close()
        print("  ✓ Saved: college_ctc_violin.png")
        
        # ANOVA
        groups = [cross_df[cross_df['college'] == c]['fte_ctc'].values for c in cross_df['college'].unique()]
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"\nANOVA Test for College CTC differences: F={f_stat:.2f}, p={p_val:.4f}")
        
        with open('analysis_outputs/statistical/anova_results.txt', 'w') as f:
            f.write(f"ANOVA Test Results:\nF-statistic: {f_stat:.4f}\nP-value: {p_val:.4f}\n")
            if p_val < 0.05:
                f.write("Conclusion: Significant difference in CTC between colleges.\n")
            else:
                f.write("Conclusion: No significant difference detected.\n")

    # 2. Tier vs CGPA
    # Do higher tiers actually demand higher CGPA?
    tier_cgpa = df[df['cgpa_cutoff'].notna() & df['placement_tier'].isin(['Tier-1', 'Tier-2', 'Tier-3', 'Dream', 'Super-Dream'])]
    
    if not tier_cgpa.empty:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=tier_cgpa, x='placement_tier', y='cgpa_cutoff', 
                    order=['Tier-3', 'Tier-2', 'Tier-1', 'Super-Dream', 'Dream'], palette='coolwarm')
        plt.title('CGPA Cutoff Requirements by Tier', fontsize=16, fontweight='bold')
        plt.ylabel('CGPA Cutoff')
        plt.tight_layout()
        plt.savefig('analysis_outputs/statistical/tier_cgpa_boxplot.png', dpi=300)
        plt.close()
        print("  ✓ Saved: tier_cgpa_boxplot.png")

def main():
    df = load_data()
    
    # Run analyses
    df_enriched = analyze_job_roles(df)
    cluster_companies(df)
    statistical_deep_dive(df)
    
    print("\nAdvanced Analysis Completed Successfully!")

if __name__ == "__main__":
    main()
