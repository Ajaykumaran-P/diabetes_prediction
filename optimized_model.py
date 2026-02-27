"""
Optimized Random Forest Model for Blood Sugar Prediction
Pre-configured best model based on NIH-K43 dataset analysis
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')


class OptimizedSugarPredictor:
    """
    Optimized Random Forest model for predicting elevated blood glucose.
    Pre-configured with best hyperparameters for diabetes prediction.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the optimized predictor.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        
        # Optimized hyperparameters for blood sugar prediction
        self.best_params = {
            'n_estimators': 200,          # More trees for better stability
            'max_depth': 15,              # Prevent overfitting
            'min_samples_split': 10,      # Minimum samples to split
            'min_samples_leaf': 4,        # Minimum samples in leaf
            'max_features': 'sqrt',       # Features to consider at each split
            'bootstrap': True,            # Bootstrap sampling
            'class_weight': 'balanced',   # Handle imbalanced classes
            'random_state': random_state,
            'n_jobs': -1                  # Use all CPU cores
        }
        
        self.model = RandomForestClassifier(**self.best_params)
        self.feature_names = None
        self.is_trained = False
        
    def train(self, X_train, y_train, feature_names=None):
        """
        Train the optimized Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            feature_names (list): Names of features
        
        Returns:
            self: Trained model
        """
        print("\n" + "="*70)
        print("TRAINING OPTIMIZED RANDOM FOREST MODEL")
        print("="*70)
        print(f"\nTraining samples: {X_train.shape[0]}")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"\nHyperparameters:")
        for param, value in self.best_params.items():
            if param != 'n_jobs':
                print(f"  {param}: {value}")
        
        self.feature_names = feature_names
        
        # Train model
        print("\n⏳ Training model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        print("✓ Training completed successfully!")
        
        return self
    
    def evaluate(self, X_test, y_test, detailed=True):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            detailed (bool): Print detailed metrics
        
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            print("✗ Model not trained yet!")
            return None
        
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        if detailed:
            print(f"\n📊 Performance Metrics:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            
            print("\n📋 Classification Report:")
            print(classification_report(y_test, y_pred))
            
            print("\n📈 Confusion Matrix:")
            print(metrics['confusion_matrix'])
        
        return metrics
    
    def get_feature_importance(self, top_n=10):
        """
        Get feature importance rankings.
        
        Args:
            top_n (int): Number of top features to return
        
        Returns:
            DataFrame: Feature importance rankings
        """
        if not self.is_trained:
            print("✗ Model not trained yet!")
            return None
        
        importances = self.model.feature_importances_
        
        if self.feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        else:
            feature_names = self.feature_names
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("\n" + "="*70)
        print(f"TOP {top_n} MOST IMPORTANT FEATURES")
        print("="*70)
        print(importance_df.head(top_n).to_string(index=False))
        
        return importance_df
    
    def plot_feature_importance(self, top_n=15, save_path='outputs/optimized_feature_importance.png'):
        """
        Visualize feature importance.
        
        Args:
            top_n (int): Number of top features to display
            save_path (str): Path to save plot
        """
        importance_df = self.get_feature_importance(top_n=len(self.feature_names))
        
        if importance_df is None:
            return
        
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features)))
        
        bars = plt.barh(range(len(top_features)), top_features['Importance'], color=colors)
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.title('Feature Importance for Blood Sugar Prediction\n(Optimized Random Forest Model)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Highlight top 4 features (uric acid, age, systolic BP, BMI)
        for i in range(min(4, len(top_features))):
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Feature importance plot saved to: {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, X_test, y_test, save_path='outputs/optimized_confusion_matrix.png'):
        """
        Plot confusion matrix.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path (str): Path to save plot
        """
        if not self.is_trained:
            print("✗ Model not trained yet!")
            return
        
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix\n(Optimized Random Forest Model)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        # Add accuracy text
        accuracy = accuracy_score(y_test, y_pred)
        plt.text(0.5, -0.15, f'Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)',
                ha='center', transform=plt.gca().transAxes, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
        plt.close()
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict
        
        Returns:
            array: Predictions (0 or 1)
        """
        if not self.is_trained:
            print("✗ Model not trained yet!")
            return None
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict probabilities for new data.
        
        Args:
            X: Features to predict
        
        Returns:
            array: Prediction probabilities
        """
        if not self.is_trained:
            print("✗ Model not trained yet!")
            return None
        
        return self.model.predict_proba(X)
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Labels
            cv (int): Number of folds
        
        Returns:
            dict: Cross-validation scores
        """
        print("\n" + "="*70)
        print(f"CROSS-VALIDATION ({cv}-Fold)")
        print("="*70)
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        
        print(f"\nAccuracy scores for each fold:")
        for i, score in enumerate(scores, 1):
            print(f"  Fold {i}: {score:.4f}")
        
        print(f"\nMean Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
    
    def save_model(self, save_path='models/optimized_sugar_predictor.pkl'):
        """
        Save the trained model.
        
        Args:
            save_path (str): Path to save model
        """
        if not self.is_trained:
            print("✗ Model not trained yet!")
            return
        
        joblib.dump(self, save_path)
        print(f"\n✓ Model saved to: {save_path}")
    
    @staticmethod
    def load_model(load_path='models/optimized_sugar_predictor.pkl'):
        """
        Load a saved model.
        
        Args:
            load_path (str): Path to load model from
        
        Returns:
            OptimizedSugarPredictor: Loaded model
        """
        model = joblib.load(load_path)
        print(f"✓ Model loaded from: {load_path}")
        return model
    
    def generate_prediction_report(self, X_test, y_test, save_path='outputs/prediction_report.txt'):
        """
        Generate comprehensive prediction report.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path (str): Path to save report
        """
        if not self.is_trained:
            print("✗ Model not trained yet!")
            return
        
        metrics = self.evaluate(X_test, y_test, detailed=False)
        importance_df = self.get_feature_importance(top_n=10)
        
        report = []
        report.append("="*70)
        report.append("BLOOD SUGAR PREDICTION MODEL REPORT")
        report.append("Optimized Random Forest Classifier")
        report.append("="*70)
        report.append("")
        
        report.append("MODEL CONFIGURATION")
        report.append("-"*70)
        for param, value in self.best_params.items():
            if param != 'n_jobs':
                report.append(f"{param}: {value}")
        report.append("")
        
        report.append("PERFORMANCE METRICS")
        report.append("-"*70)
        report.append(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        report.append(f"Precision: {metrics['precision']:.4f}")
        report.append(f"Recall:    {metrics['recall']:.4f}")
        report.append(f"F1-Score:  {metrics['f1_score']:.4f}")
        report.append(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        report.append("")
        
        report.append("TOP 10 MOST IMPORTANT FEATURES")
        report.append("-"*70)
        report.append(importance_df.head(10).to_string(index=False))
        report.append("")
        
        report.append("KEY FINDINGS")
        report.append("-"*70)
        report.append("Based on the NIH-K43 dataset analysis:")
        report.append(f"1. {importance_df.iloc[0]['Feature']} is the most important predictor")
        report.append(f"2. {importance_df.iloc[1]['Feature']} is the second most important factor")
        report.append(f"3. {importance_df.iloc[2]['Feature']} ranks third in importance")
        report.append(f"4. {importance_df.iloc[3]['Feature']} is the fourth key indicator")
        report.append("")
        
        report.append("CLINICAL IMPLICATIONS")
        report.append("-"*70)
        report.append("• Early screening should focus on monitoring top predictive features")
        report.append("• Healthcare interventions should target modifiable risk factors")
        report.append("• Regular monitoring recommended for high-risk individuals")
        report.append("• Results support preventive healthcare strategies")
        report.append("")
        
        report.append("IMPORTANT NOTES")
        report.append("-"*70)
        report.append("• Feature importance indicates association, not causation")
        report.append("• Results are specific to the NIH-K43 dataset from Nigeria")
        report.append("• Clinical decisions should be made by qualified healthcare professionals")
        report.append("• Model should be regularly updated with new data")
        report.append("")
        
        report_text = "\n".join(report)
        
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Prediction report saved to: {save_path}")
        
        return report_text


# Example usage
if __name__ == "__main__":
    print("Optimized Random Forest Model for Blood Sugar Prediction")
    print("\nThis is the recommended model based on project analysis.")
    print("\nExample usage:")
    print("""
    from optimized_model import OptimizedSugarPredictor
    from data_preprocessing import DataPreprocessor
    
    # Preprocess data
    preprocessor = DataPreprocessor('data/diabetes_dataset.csv')
    X_train, X_test, y_train, y_test, features = preprocessor.preprocess_pipeline('diabetes_status')
    
    # Train optimized model
    predictor = OptimizedSugarPredictor()
    predictor.train(X_train, y_train, feature_names=features)
    
    # Evaluate
    predictor.evaluate(X_test, y_test)
    predictor.cross_validate(X_train, y_train)
    
    # Visualize
    predictor.plot_feature_importance()
    predictor.plot_confusion_matrix(X_test, y_test)
    
    # Save
    predictor.save_model()
    predictor.generate_prediction_report(X_test, y_test)
    
    # Make predictions on new data
    predictions = predictor.predict(new_data)
    probabilities = predictor.predict_proba(new_data)
    """)
