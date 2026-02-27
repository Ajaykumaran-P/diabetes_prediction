"""
Quick Start: Use the Best Pre-Configured Model for Sugar Prediction
Optimized Random Forest - Best Model for Blood Glucose Prediction
"""

import sys
import os
from data_preprocessing import DataPreprocessor
from optimized_model import OptimizedSugarPredictor
import warnings
warnings.filterwarnings('ignore')


def main():
    """
    Quick start with the optimized Random Forest model.
    """
    print("\n" + "="*70)
    print("BLOOD SUGAR PREDICTION - OPTIMIZED MODEL")
    print("Using Pre-Configured Random Forest (Best Performance)")
    print("="*70 + "\n")
    
    # Configuration
    DATA_FILE = 'data/diabetes_dataset.csv'
    TARGET_COLUMN = 'diabetes_status'  # Adjust to your target column
    EXCLUDE_COLUMNS = None  # Add any ID columns to exclude
    
    print("🎯 Why Random Forest?")
    print("  ✓ Excellent accuracy for tabular medical data")
    print("  ✓ Provides interpretable feature importance")
    print("  ✓ Robust to outliers and missing values")
    print("  ✓ No assumptions about data distribution")
    print("  ✓ Handles non-linear relationships well")
    print("  ✓ Reduced overfitting through ensemble learning")
    
    # Check if data exists
    if not os.path.exists(DATA_FILE):
        print(f"\n✗ Error: Dataset not found at {DATA_FILE}")
        print("\nPlease place your dataset in the 'data' folder.")
        print("Update DATA_FILE and TARGET_COLUMN variables if needed.")
        return
    
    # ================================================================
    # STEP 1: PREPROCESS DATA
    # ================================================================
    print("\n" + "="*70)
    print("STEP 1: DATA PREPROCESSING")
    print("="*70)
    
    preprocessor = DataPreprocessor(DATA_FILE)
    result = preprocessor.preprocess_pipeline(
        target_column=TARGET_COLUMN,
        exclude_columns=EXCLUDE_COLUMNS,
        test_size=0.2,
        scale=True
    )
    
    if result is None:
        print("\n✗ Preprocessing failed.")
        return
    
    X_train, X_test, y_train, y_test, feature_names = result
    
    # ================================================================
    # STEP 2: TRAIN OPTIMIZED MODEL
    # ================================================================
    print("\n" + "="*70)
    print("STEP 2: TRAIN OPTIMIZED RANDOM FOREST MODEL")
    print("="*70)
    
    predictor = OptimizedSugarPredictor(random_state=42)
    predictor.train(X_train, y_train, feature_names=feature_names)
    
    # ================================================================
    # STEP 3: EVALUATE MODEL
    # ================================================================
    print("\n" + "="*70)
    print("STEP 3: MODEL EVALUATION")
    print("="*70)
    
    metrics = predictor.evaluate(X_test, y_test, detailed=True)
    
    # ================================================================
    # STEP 4: CROSS-VALIDATION
    # ================================================================
    print("\n" + "="*70)
    print("STEP 4: CROSS-VALIDATION")
    print("="*70)
    
    cv_results = predictor.cross_validate(X_train, y_train, cv=5)
    
    # ================================================================
    # STEP 5: FEATURE IMPORTANCE ANALYSIS
    # ================================================================
    print("\n" + "="*70)
    print("STEP 5: FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    importance_df = predictor.get_feature_importance(top_n=15)
    
    # ================================================================
    # STEP 6: GENERATE VISUALIZATIONS
    # ================================================================
    print("\n" + "="*70)
    print("STEP 6: GENERATE VISUALIZATIONS")
    print("="*70)
    
    predictor.plot_feature_importance(top_n=15)
    predictor.plot_confusion_matrix(X_test, y_test)
    
    # ================================================================
    # STEP 7: SAVE MODEL AND REPORT
    # ================================================================
    print("\n" + "="*70)
    print("STEP 7: SAVE MODEL AND GENERATE REPORT")
    print("="*70)
    
    predictor.save_model('models/optimized_sugar_predictor.pkl')
    predictor.generate_prediction_report(X_test, y_test)
    
    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print("\n" + "="*70)
    print("🎉 ANALYSIS COMPLETE - SUMMARY")
    print("="*70)
    print(f"\n✓ Model: Optimized Random Forest Classifier")
    print(f"✓ Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"✓ ROC-AUC Score: {metrics['roc_auc']:.4f}")
    print(f"✓ Cross-Validation Mean: {cv_results['mean']:.4f} (+/- {cv_results['std']*2:.4f})")
    
    print(f"\n✓ Training Samples: {X_train.shape[0]}")
    print(f"✓ Testing Samples: {X_test.shape[0]}")
    print(f"✓ Number of Features: {len(feature_names)}")
    
    print("\n🔝 Top 5 Most Important Features:")
    for i in range(min(5, len(importance_df))):
        row = importance_df.iloc[i]
        print(f"   {i+1}. {row['Feature']}: {row['Importance']:.4f}")
    
    print("\n📊 Generated Files:")
    print("   • models/optimized_sugar_predictor.pkl - Trained model")
    print("   • outputs/optimized_feature_importance.png - Feature importance plot")
    print("   • outputs/optimized_confusion_matrix.png - Confusion matrix")
    print("   • outputs/prediction_report.txt - Comprehensive report")
    
    print("\n💡 Using the Model for Predictions:")
    print("""
    # Load saved model
    from optimized_model import OptimizedSugarPredictor
    predictor = OptimizedSugarPredictor.load_model('models/optimized_sugar_predictor.pkl')
    
    # Make predictions
    predictions = predictor.predict(new_patient_data)
    probabilities = predictor.predict_proba(new_patient_data)
    """)
    
    print("\n" + "="*70)
    print("Model ready for deployment! 🚀")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Process interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
