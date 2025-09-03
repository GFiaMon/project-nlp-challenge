"""
Modeling Utilities for Fake News Detection Project
Includes experiment tracking, feature engineering, and evaluation functions
"""

import pandas as pd
import numpy as np
import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, roc_curve, confusion_matrix, classification_report,
                            precision_recall_curve, auc)


class ExperimentTracker:
    """Track and manage all experiments with persistent storage"""
    
    def __init__(self, results_dir='../models/experiment_tracking'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.experiment_history_file = self.results_dir / 'experiment_history.json'
        self.results_file = self.results_dir / 'all_results.csv'
        self.load_existing_results()
    
    def load_existing_results(self):
        """Load existing results from previous runs"""
        if self.results_file.exists():
            self.existing_results = pd.read_csv(self.results_file)
            print(f"📋 Loaded {len(self.existing_results)} existing experiments")
        else:
            self.existing_results = pd.DataFrame()
            print("🆕 Starting fresh experiment tracking")
    
    def add_experiment_results(self, new_results):
        """Add new experiment results to tracking"""
        if isinstance(new_results, list):
            new_df = pd.DataFrame(new_results)
        else:
            new_df = new_results
        
        # Add cleaning strategy info
        new_df['cleaning_strategy'] = new_df['experiment'].apply(
            lambda x: 'Basic' if 'Basic' in x else 'Aggressive'
        )
        
        # Combine with existing results
        if len(self.existing_results) > 0:
            self.all_results = pd.concat([self.existing_results, new_df], ignore_index=True)
        else:
            self.all_results = new_df
        
        # Save updated results
        self.all_results.to_csv(self.results_file, index=False)
        
        # Update experiment history
        self.update_experiment_history()
        
        print(f"💾 Saved {len(new_df)} new experiments. Total: {len(self.all_results)}")
        return self.all_results
    
    def update_experiment_history(self):
        """Update experiment history with metadata"""
        if len(self.all_results) == 0:
            return
            
        history = {
            'last_updated': datetime.datetime.now().isoformat(),
            'total_experiments': len(self.all_results),
            'best_accuracy': float(self.all_results['accuracy'].max()),
            'best_roc_auc': float(self.all_results['roc_auc'].max()),
            'best_f1': float(self.all_results['f1'].max()),
            'experiment_configs_tested': int(self.all_results['experiment'].nunique()),
            'models_tested': int(self.all_results['model'].nunique())
        }
        
        with open(self.experiment_history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def get_best_experiments(self, metric='roc_auc', top_n=5):
        """Get top N best experiments by metric"""
        if len(self.all_results) == 0:
            return pd.DataFrame()
        
        return self.all_results.nlargest(top_n, metric)[['experiment', 'model', metric, 'accuracy', 'f1', 'timestamp']]
    
    def get_experiment_summary(self):
        """Get comprehensive experiment summary"""
        if len(self.all_results) == 0:
            return "No experiments completed yet"
        
        summary = f"""
🔬 EXPERIMENT TRACKING SUMMARY
{'='*50}
📊 Total Experiments: {len(self.all_results)}
🏆 Best Accuracy: {self.all_results['accuracy'].max():.4f}
🏆 Best ROC-AUC: {self.all_results['roc_auc'].max():.4f}
🏆 Best F1-Score: {self.all_results['f1'].max():.4f}

📋 Feature Combinations Tested: {self.all_results['experiment'].nunique()}
🤖 Models Tested: {self.all_results['model'].nunique()}

🎯 TOP 3 EXPERIMENTS BY ROC-AUC:
{self.get_best_experiments('roc_auc', 3).to_string(index=False)}
        """
        return summary


def engineer_time_features(df):
    """Engineer time-based features from date column"""
    print("🕒 Engineering time-based features...")
    
    # Create a copy to avoid modifying original
    df_engineered = df.copy()
    
    # Extract year if date column exists
    if 'year' in df_engineered.columns:
        # Convert year to numeric, handling 'Unknown' values
        df_engineered['year_numeric'] = pd.to_numeric(df_engineered['year'].replace('Unknown', np.nan), errors='coerce')
        
        # Create time-based features
        df_engineered['is_recent'] = (df_engineered['year_numeric'] >= 2020).astype(int)
        df_engineered['is_medium_age'] = ((df_engineered['year_numeric'] >= 2015) & (df_engineered['year_numeric'] < 2020)).astype(int)
        df_engineered['is_old'] = (df_engineered['year_numeric'] < 2015).astype(int)
        
        # Fill NaN values with 0 (for unknown years)
        df_engineered[['is_recent', 'is_medium_age', 'is_old']] = df_engineered[['is_recent', 'is_medium_age', 'is_old']].fillna(0)
        
        print(f"   ✅ Added time features: is_recent, is_medium_age, is_old")
    else:
        print("   ⚠️ No year column found, skipping time features")
    
    return df_engineered


def combine_text_features(data, feature_columns):
    """Combine multiple text columns into a single string for embedding"""
    if len(feature_columns) == 1:
        # Single feature
        return data[feature_columns[0]].fillna('').astype(str)
    else:
        # Multiple features - combine them
        combined = data[feature_columns].fillna('').astype(str)
        return combined.apply(lambda x: ' | '.join(x), axis=1)


def evaluate_model_comprehensive(model, X_test, y_test, model_name, config_name):
    """Comprehensive model evaluation with multiple metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    results = {
        'experiment': config_name,
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'timestamp': datetime.datetime.now().isoformat()
    }
    return results


def plot_experiment_results(results_df, save_path=None):
    """Create comprehensive visualization of experiment results"""
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🔬 Comprehensive Experiment Results Analysis', fontsize=16, fontweight='bold')
    
    # 1. Overall Performance Comparison
    ax1 = axes[0, 0]
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    avg_metrics = results_df[metrics].mean()
    ax1.bar(metrics, avg_metrics, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    ax1.set_title('📊 Average Performance Across All Experiments')
    ax1.set_ylabel('Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Model Performance Comparison
    ax2 = axes[0, 1]
    model_performance = results_df.groupby('model')['roc_auc'].mean().sort_values(ascending=False)
    ax2.bar(model_performance.index, model_performance.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('🤖 Model Performance (ROC-AUC)')
    ax2.set_ylabel('ROC-AUC Score')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Feature Combination Performance
    ax3 = axes[0, 2]
    feature_performance = results_df.groupby('experiment')['roc_auc'].mean().sort_values(ascending=False).head(10)
    ax3.barh(range(len(feature_performance)), feature_performance.values, color='#96CEB4')
    ax3.set_yticks(range(len(feature_performance)))
    ax3.set_yticklabels(feature_performance.index, fontsize=8)
    ax3.set_title('📋 Top 10 Feature Combinations')
    ax3.set_xlabel('ROC-AUC Score')
    
    # 4. Cleaning Strategy Comparison
    ax4 = axes[1, 0]
    cleaning_performance = results_df.groupby('cleaning_strategy')['roc_auc'].mean()
    ax4.bar(cleaning_performance.index, cleaning_performance.values, color=['#FF6B6B', '#4ECDC4'])
    ax4.set_title('🧹 Cleaning Strategy Performance')
    ax4.set_ylabel('ROC-AUC Score')
    
    # 5. Performance Distribution
    ax5 = axes[1, 1]
    ax5.hist(results_df['roc_auc'], bins=20, color='#45B7D1', alpha=0.7, edgecolor='black')
    ax5.axvline(results_df['roc_auc'].mean(), color='red', linestyle='--', label=f'Mean: {results_df["roc_auc"].mean():.3f}')
    ax5.axvline(results_df['roc_auc'].max(), color='green', linestyle='--', label=f'Max: {results_df["roc_auc"].max():.3f}')
    ax5.set_title('📈 ROC-AUC Score Distribution')
    ax5.set_xlabel('ROC-AUC Score')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    
    # 6. Correlation Heatmap
    ax6 = axes[1, 2]
    correlation_matrix = results_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax6, fmt='.2f')
    ax6.set_title('🔥 Metric Correlations')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved to: {save_path}")
    
    plt.show()


def plot_model_comparison(results_df, save_path=None):
    """Create detailed model comparison plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🤖 Detailed Model Comparison Analysis', fontsize=16, fontweight='bold')
    
    # 1. ROC-AUC Comparison
    ax1 = axes[0, 0]
    model_roc = results_df.groupby('model')['roc_auc'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    ax1.bar(model_roc.index, model_roc['mean'], yerr=model_roc['std'], capsize=5, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('📊 ROC-AUC Performance (with std)')
    ax1.set_ylabel('ROC-AUC Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Accuracy Comparison
    ax2 = axes[0, 1]
    model_acc = results_df.groupby('model')['accuracy'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    ax2.bar(model_acc.index, model_acc['mean'], yerr=model_acc['std'], capsize=5, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('🎯 Accuracy Performance (with std)')
    ax2.set_ylabel('Accuracy Score')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. F1-Score Comparison
    ax3 = axes[1, 0]
    model_f1 = results_df.groupby('model')['f1'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    ax3.bar(model_f1.index, model_f1['mean'], yerr=model_f1['std'], capsize=5, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax3.set_title('⚖️ F1-Score Performance (with std)')
    ax3.set_ylabel('F1-Score')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Performance Heatmap
    ax4 = axes[1, 1]
    pivot_table = results_df.pivot_table(
        values=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        index='model',
        aggfunc='mean'
    )
    sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', fmt='.3f', ax=ax4)
    ax4.set_title('🔥 Model Performance Heatmap')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Model comparison plot saved to: {save_path}")
    
    plt.show()


def plot_feature_analysis(results_df, save_path=None):
    """Create feature combination analysis plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('📋 Feature Combination Analysis', fontsize=16, fontweight='bold')
    
    # 1. Top Feature Combinations by ROC-AUC
    ax1 = axes[0, 0]
    top_features = results_df.groupby('experiment')['roc_auc'].mean().sort_values(ascending=False).head(15)
    ax1.barh(range(len(top_features)), top_features.values, color='#96CEB4')
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features.index, fontsize=8)
    ax1.set_title('🏆 Top 15 Feature Combinations (ROC-AUC)')
    ax1.set_xlabel('ROC-AUC Score')
    
    # 2. Feature Type Performance
    ax2 = axes[0, 1]
    # Extract feature types from experiment names
    results_df['feature_type'] = results_df['experiment'].apply(
        lambda x: 'Title+Text' if 'Title+Text' in x else 'Title' if 'Title' in x else 'Text' if 'Text' in x else 'Other'
    )
    feature_type_perf = results_df.groupby('feature_type')['roc_auc'].mean().sort_values(ascending=False)
    ax2.bar(feature_type_perf.index, feature_type_perf.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax2.set_title('📝 Feature Type Performance')
    ax2.set_ylabel('ROC-AUC Score')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Cleaning Strategy Impact
    ax3 = axes[1, 0]
    cleaning_impact = results_df.groupby('cleaning_strategy')['roc_auc'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    ax3.bar(cleaning_impact.index, cleaning_impact['mean'], yerr=cleaning_impact['std'], capsize=5, color=['#FF6B6B', '#4ECDC4'])
    ax3.set_title('🧹 Cleaning Strategy Impact')
    ax3.set_ylabel('ROC-AUC Score')
    
    # 4. Feature vs Model Performance
    ax4 = axes[1, 1]
    feature_model_perf = results_df.pivot_table(
        values='roc_auc',
        index='experiment',
        columns='model',
        aggfunc='mean'
    )
    sns.heatmap(feature_model_perf, annot=True, cmap='YlOrRd', fmt='.3f', ax=ax4)
    ax4.set_title('🔥 Feature-Model Performance Matrix')
    ax4.tick_params(axis='x', rotation=45)
    ax4.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Feature analysis plot saved to: {save_path}")
    
    plt.show()


# Make sure these functions are available for import
__all__ = [
    'ExperimentTracker',
    'engineer_time_features', 
    'combine_text_features',
    'evaluate_model_comprehensive',
    'plot_experiment_results',
    'plot_model_comparison',
    'plot_feature_analysis'
]