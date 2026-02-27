"""
Feature Importance Analysis Module for Diabetes Prediction Project
Analyzes and visualizes feature importance for blood glucose prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance for diabetes prediction models.
    """
    
    def __init__(self, feature_names):
        """
        Initialize the analyzer with feature names.
        
        Args:
            feature_names (list): List of feature names
        """
        self.feature_names = feature_names
        self.importance_results = {}
        
    def analyze_random_forest_importance(self, X_train, y_train, n_estimators=100, random_state=42):
        """
        Analyze feature importance using Random Forest.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators (int): Number of trees
            random_state (int): Random seed
        
        Returns:
            DataFrame: Feature importance rankings
        """
        print("\n" + "="*70)
        print("RANDOM FOREST FEATURE IMPORTANCE ANALYSIS")
        print("="*70 + "\n")
        
        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        rf_model.fit(X_train, y_train)
        
        # Get feature importance
        importances = rf_model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        self.importance_results['Random Forest'] = importance_df
        
        print("Top 10 Most Important Features (Random Forest):")
        print(importance_df.head(10).to_string(index=False))
        
        return importance_df, rf_model
    
    def analyze_gradient_boosting_importance(self, X_train, y_train, n_estimators=100, random_state=42):
        """
        Analyze feature importance using Gradient Boosting.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators (int): Number of boosting stages
            random_state (int): Random seed
        
        Returns:
            DataFrame: Feature importance rankings
        """
        print("\n" + "="*70)
        print("GRADIENT BOOSTING FEATURE IMPORTANCE ANALYSIS")
        print("="*70 + "\n")
        
        # Train Gradient Boosting
        gb_model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
        gb_model.fit(X_train, y_train)
        
        # Get feature importance
        importances = gb_model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        self.importance_results['Gradient Boosting'] = importance_df
        
        print("Top 10 Most Important Features (Gradient Boosting):")
        print(importance_df.head(10).to_string(index=False))
        
        return importance_df, gb_model
    
    def analyze_xgboost_importance(self, X_train, y_train, n_estimators=100, random_state=42):
        """
        Analyze feature importance using XGBoost.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators (int): Number of boosting rounds
            random_state (int): Random seed
        
        Returns:
            DataFrame: Feature importance rankings
        """
        if not XGBOOST_AVAILABLE:
            print("✗ XGBoost not available. Install with: pip install xgboost")
            return None, None
        
        print("\n" + "="*70)
        print("XGBOOST FEATURE IMPORTANCE ANALYSIS")
        print("="*70 + "\n")
        
        # Train XGBoost
        xgb_model = XGBClassifier(n_estimators=n_estimators, random_state=random_state, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        
        # Get feature importance
        importances = xgb_model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        self.importance_results['XGBoost'] = importance_df
        
        print("Top 10 Most Important Features (XGBoost):")
        print(importance_df.head(10).to_string(index=False))
        
        return importance_df, xgb_model
    
    def analyze_permutation_importance(self, model, X_test, y_test, n_repeats=10, random_state=42):
        """
        Analyze feature importance using permutation importance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            n_repeats (int): Number of times to permute
            random_state (int): Random seed
        
        Returns:
            DataFrame: Feature importance rankings
        """
        print("\n" + "="*70)
        print("PERMUTATION IMPORTANCE ANALYSIS")
        print("="*70 + "\n")
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X_test, y_test, 
            n_repeats=n_repeats, 
            random_state=random_state,
            n_jobs=-1
        )
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': perm_importance.importances_mean,
            'Std': perm_importance.importances_std
        }).sort_values('Importance', ascending=False)
        
        self.importance_results['Permutation'] = importance_df
        
        print("Top 10 Most Important Features (Permutation):")
        print(importance_df.head(10).to_string(index=False))
        
        return importance_df
    
    def plot_feature_importance(self, importance_df, title='Feature Importance', 
                               top_n=15, save_path='outputs/feature_importance.png'):
        """
        Plot feature importance.
        
        Args:
            importance_df (DataFrame): Feature importance data
            title (str): Plot title
            top_n (int): Number of top features to display
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        # Create color map
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        
        # Create horizontal bar plot
        bars = plt.barh(range(len(top_features)), top_features['Importance'], color=colors)
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Feature importance plot saved to: {save_path}")
        plt.close()
    
    def plot_multiple_importance_comparison(self, top_n=10, save_path='outputs/importance_comparison.png'):
        """
        Compare feature importance across different methods.
        
        Args:
            top_n (int): Number of top features to display
            save_path (str): Path to save the plot
        """
        if len(self.importance_results) < 2:
            print("✗ Need at least 2 importance analyses to compare")
            return
        
        # Get all unique features from top N of each method
        all_top_features = set()
        for method, df in self.importance_results.items():
            all_top_features.update(df.head(top_n)['Feature'].tolist())
        
        # Create comparison DataFrame
        comparison_data = []
        for feature in all_top_features:
            row = {'Feature': feature}
            for method, df in self.importance_results.items():
                feature_row = df[df['Feature'] == feature]
                if not feature_row.empty:
                    row[method] = feature_row['Importance'].values[0]
                else:
                    row[method] = 0
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df['Average'] = comparison_df.iloc[:, 1:].mean(axis=1)
        comparison_df = comparison_df.sort_values('Average', ascending=False).head(top_n)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        methods = [col for col in comparison_df.columns if col not in ['Feature', 'Average']]
        x = np.arange(len(comparison_df))
        width = 0.8 / len(methods)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        
        for i, method in enumerate(methods):
            offset = width * i - (width * len(methods) / 2) + width / 2
            ax.bar(x + offset, comparison_df[method], width, label=method, color=colors[i])
        
        ax.set_xlabel('Features', fontsize=12, fontweight='bold')
        ax.set_ylabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance Comparison Across Methods', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Feature'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Importance comparison plot saved to: {save_path}")
        plt.close()
    
    def plot_top_features_detailed(self, importance_df, top_n=5, save_path='outputs/top_features_detailed.png'):
        """
        Create detailed visualization of top features.
        
        Args:
            importance_df (DataFrame): Feature importance data
            top_n (int): Number of top features to display
            save_path (str): Path to save the plot
        """
        top_features = importance_df.head(top_n)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Top {top_n} Most Important Features', fontsize=16, fontweight='bold')
        
        # Bar chart
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'][:top_n]
        bars = ax1.barh(range(top_n), top_features['Importance'], color=colors)
        ax1.set_yticks(range(top_n))
        ax1.set_yticklabels(top_features['Feature'])
        ax1.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax1.set_title('Feature Importance Ranking', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        # Pie chart
        ax2.pie(top_features['Importance'], labels=top_features['Feature'], 
               autopct='%1.1f%%', startangle=90, colors=colors)
        ax2.set_title('Relative Importance Distribution', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Detailed top features plot saved to: {save_path}")
        plt.close()
    
    def generate_importance_report(self, save_path='outputs/feature_importance_report.txt'):
        """
        Generate a text report of feature importance findings.
        
        Args:
            save_path (str): Path to save the report
        """
        report = []
        report.append("="*70)
        report.append("FEATURE IMPORTANCE ANALYSIS REPORT")
        report.append("Diabetes Prediction Project - NIH-K43 Community Screening")
        report.append("="*70)
        report.append("")
        
        for method, df in self.importance_results.items():
            report.append(f"\n{method} Method:")
            report.append("-" * 70)
            report.append("\nTop 10 Features:")
            report.append(df.head(10).to_string(index=False))
            report.append("")
        
        # Key findings
        report.append("\n" + "="*70)
        report.append("KEY FINDINGS")
        report.append("="*70)
        report.append("\nThe analysis reveals the following insights:")
        report.append("")
        
        if 'Random Forest' in self.importance_results:
            top_feature = self.importance_results['Random Forest'].iloc[0]
            report.append(f"1. Most Important Feature: {top_feature['Feature']}")
            report.append(f"   - Importance Score: {top_feature['Importance']:.4f}")
            report.append("")
        
        report.append("2. Clinical Implications:")
        report.append("   - These features are most strongly associated with elevated blood glucose")
        report.append("   - Healthcare professionals should prioritize monitoring these factors")
        report.append("   - Early intervention targeting these areas may help prevent diabetes")
        report.append("")
        
        report.append("3. Important Considerations:")
        report.append("   - Feature importance indicates association, not causation")
        report.append("   - Results are specific to this model and dataset")
        report.append("   - Low importance doesn't mean the feature is clinically unimportant")
        report.append("   - Features may have multicollinearity effects")
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save to file
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Feature importance report saved to: {save_path}")
        print("\n" + report_text)
        
        return report_text
    
    def comprehensive_analysis(self, X_train, X_test, y_train, y_test, 
                              random_state=42, top_n=15):
        """
        Run comprehensive feature importance analysis.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            random_state (int): Random seed
            top_n (int): Number of top features to display
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
        print("="*70)
        
        # Random Forest
        rf_importance, rf_model = self.analyze_random_forest_importance(X_train, y_train, random_state=random_state)
        self.plot_feature_importance(rf_importance, 'Random Forest Feature Importance', 
                                     top_n, 'outputs/rf_importance.png')
        
        # Gradient Boosting
        gb_importance, gb_model = self.analyze_gradient_boosting_importance(X_train, y_train, random_state=random_state)
        self.plot_feature_importance(gb_importance, 'Gradient Boosting Feature Importance', 
                                     top_n, 'outputs/gb_importance.png')
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            xgb_importance, xgb_model = self.analyze_xgboost_importance(X_train, y_train, random_state=random_state)
            if xgb_importance is not None:
                self.plot_feature_importance(xgb_importance, 'XGBoost Feature Importance', 
                                           top_n, 'outputs/xgb_importance.png')
        
        # Permutation importance (using Random Forest model)
        perm_importance = self.analyze_permutation_importance(rf_model, X_test, y_test, random_state=random_state)
        
        # Comparison plots
        self.plot_multiple_importance_comparison(top_n=10)
        self.plot_top_features_detailed(rf_importance, top_n=5)
        
        # Generate report
        self.generate_importance_report()
        
        print("\n" + "="*70)
        print("COMPREHENSIVE ANALYSIS COMPLETED")
        print("="*70)


# Example usage
if __name__ == "__main__":
    print("Feature Importance Analysis Module")
    print("This module should be used after data preprocessing.")
    print("\nExample usage:")
    print("""
    from feature_importance import FeatureImportanceAnalyzer
    from data_preprocessing import DataPreprocessor
    
    # Preprocess data
    preprocessor = DataPreprocessor('data/diabetes_dataset.csv')
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_pipeline('diabetes_status')
    
    # Analyze feature importance
    analyzer = FeatureImportanceAnalyzer(feature_names)
    analyzer.comprehensive_analysis(X_train, X_test, y_train, y_test)
    """)
