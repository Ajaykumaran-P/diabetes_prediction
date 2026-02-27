"""
Model Comparison Module for Diabetes Prediction Project
Compares multiple machine learning models for blood glucose prediction
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Note: XGBoost not installed. Install with: pip install xgboost")


class ModelComparison:
    """
    Compare multiple machine learning models for diabetes prediction.
    """
    
    def __init__(self):
        """
        Initialize models for comparison.
        """
        self.models = {}
        self.results = {}
        self.trained_models = {}
        
    def initialize_models(self, random_state=42):
        """
        Initialize all models for comparison.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=random_state),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=random_state),
            'Support Vector Machine': SVC(probability=True, random_state=random_state),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB()
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=random_state, eval_metric='logloss')
        
        print(f"✓ Initialized {len(self.models)} models for comparison")
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate all models.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labels
        """
        print("\n" + "="*70)
        print("TRAINING AND EVALUATING MODELS")
        print("="*70 + "\n")
        
        results_list = []
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except:
                roc_auc = 0.0
            
            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            results_list.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc
            })
            
            print(f"  ✓ Accuracy: {accuracy:.4f}")
        
        # Create results DataFrame
        self.results_df = pd.DataFrame(results_list)
        self.results_df = self.results_df.sort_values('Accuracy', ascending=False)
        
        print("\n" + "="*70)
        print("MODEL COMPARISON RESULTS")
        print("="*70)
        print(self.results_df.to_string(index=False))
        
        return self.results_df
    
    def plot_model_comparison(self, save_path='outputs/model_comparison.png'):
        """
        Create visualization comparing model performance.
        
        Args:
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Accuracy comparison
        ax1 = axes[0, 0]
        models = self.results_df['Model']
        accuracy = self.results_df['Accuracy']
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        bars = ax1.barh(models, accuracy, color=colors)
        ax1.set_xlabel('Accuracy Score', fontweight='bold')
        ax1.set_title('Accuracy Comparison', fontweight='bold')
        ax1.set_xlim([0, 1])
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center', fontsize=9)
        
        # Multiple metrics comparison
        ax2 = axes[0, 1]
        metrics_df = self.results_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
        metrics_df.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Multiple Metrics Comparison', fontweight='bold')
        ax2.set_ylabel('Score', fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.legend(loc='lower right')
        ax2.tick_params(axis='x', rotation=45)
        
        # ROC-AUC comparison
        ax3 = axes[1, 0]
        roc_auc = self.results_df['ROC-AUC']
        bars = ax3.barh(models, roc_auc, color=colors)
        ax3.set_xlabel('ROC-AUC Score', fontweight='bold')
        ax3.set_title('ROC-AUC Comparison', fontweight='bold')
        ax3.set_xlim([0, 1])
        for i, bar in enumerate(bars):
            width = bar.get_width()
            if width > 0:
                ax3.text(width, bar.get_y() + bar.get_height()/2, 
                        f'{width:.4f}', ha='left', va='center', fontsize=9)
        
        # Best model metrics breakdown
        ax4 = axes[1, 1]
        best_model = self.results_df.iloc[0]
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metrics_values = [best_model['Accuracy'], best_model['Precision'], 
                         best_model['Recall'], best_model['F1-Score'], best_model['ROC-AUC']]
        bars = ax4.bar(metrics_names, metrics_values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'])
        ax4.set_title(f'Best Model: {best_model["Model"]}', fontweight='bold')
        ax4.set_ylabel('Score', fontweight='bold')
        ax4.set_ylim([0, 1])
        ax4.tick_params(axis='x', rotation=45)
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Model comparison plot saved to: {save_path}")
        plt.close()
    
    def plot_confusion_matrices(self, save_path='outputs/confusion_matrices.png'):
        """
        Plot confusion matrices for all models.
        
        Args:
            save_path (str): Path to save the plot
        """
        n_models = len(self.results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold')
        
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            ax = axes[idx]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                       cbar=False, square=True)
            ax.set_title(f'{name}\nAccuracy: {result["accuracy"]:.4f}', fontweight='bold')
            ax.set_ylabel('True Label', fontweight='bold')
            ax.set_xlabel('Predicted Label', fontweight='bold')
        
        # Hide unused subplots
        for idx in range(len(self.results), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrices saved to: {save_path}")
        plt.close()
    
    def save_best_model(self, save_path='models/best_model.pkl'):
        """
        Save the best performing model.
        
        Args:
            save_path (str): Path to save the model
        """
        best_model_name = self.results_df.iloc[0]['Model']
        best_model = self.trained_models[best_model_name]
        
        joblib.dump(best_model, save_path)
        print(f"\n✓ Best model ({best_model_name}) saved to: {save_path}")
        
        return best_model_name, best_model
    
    def generate_classification_report(self, y_test, model_name=None):
        """
        Generate detailed classification report for a specific model.
        
        Args:
            y_test: True labels
            model_name (str): Name of the model (if None, uses best model)
        """
        if model_name is None:
            model_name = self.results_df.iloc[0]['Model']
        
        y_pred = self.results[model_name]['y_pred']
        
        print(f"\n{'='*70}")
        print(f"CLASSIFICATION REPORT: {model_name}")
        print('='*70)
        print(classification_report(y_test, y_pred))
        
        return classification_report(y_test, y_pred, output_dict=True)


# Example usage
if __name__ == "__main__":
    print("Model Comparison Module")
    print("This module should be used after data preprocessing.")
    print("\nExample usage:")
    print("""
    from model_comparison import ModelComparison
    from data_preprocessing import DataPreprocessor
    
    # Preprocess data
    preprocessor = DataPreprocessor('data/diabetes_dataset.csv')
    X_train, X_test, y_train, y_test, _ = preprocessor.preprocess_pipeline('diabetes_status')
    
    # Compare models
    comparison = ModelComparison()
    comparison.initialize_models()
    results = comparison.train_and_evaluate(X_train, X_test, y_train, y_test)
    comparison.plot_model_comparison()
    comparison.plot_confusion_matrices()
    comparison.save_best_model()
    """)
