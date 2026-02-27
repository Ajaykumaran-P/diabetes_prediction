"""
Main Pipeline for Diabetes Prediction Project
NIH-K43 Community Screening Dataset
"""

import sys
import os
from data_preprocessing import DataPreprocessor
from model_comparison import ModelComparison
from feature_importance import FeatureImportanceAnalyzer
import warnings
warnings.filterwarnings('ignore')


def main():
    """
    Main pipeline for diabetes prediction analysis.
    """
    print("\n" + "="*70)
    print("DIABETES PREDICTION PROJECT")
    print("NIH-K43 Community Screening Dataset Analysis")
    print("="*70 + "\n")
    
    # Configuration
    DATA_FILE = 'data/diabetes_dataset.csv'  # Adjust this to your dataset filename
    TARGET_COLUMN = 'diabetes_status'  # Adjust this to your target column name
    EXCLUDE_COLUMNS = None  # Add any ID columns or columns to exclude
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    print("Configuration:")
    print(f"  Data file: {DATA_FILE}")
    print(f"  Target column: {TARGET_COLUMN}")
    print(f"  Test size: {TEST_SIZE}")
    print(f"  Random state: {RANDOM_STATE}")
    
    # Check if data file exists
    if not os.path.exists(DATA_FILE):
        print(f"\n✗ Error: Dataset not found at {DATA_FILE}")
        print("\nPlease ensure your dataset is in the 'data' folder.")
        print("Expected file: data/diabetes_dataset.csv")
        print("\nYou can modify the DATA_FILE and TARGET_COLUMN variables in main.py")
        print("to match your actual dataset structure.")
        return
    
    # ====================================================================
    # STEP 1: DATA PREPROCESSING
    # ====================================================================
    print("\n" + "="*70)
    print("STEP 1: DATA PREPROCESSING")
    print("="*70)
    
    preprocessor = DataPreprocessor(DATA_FILE)
    
    result = preprocessor.preprocess_pipeline(
        target_column=TARGET_COLUMN,
        exclude_columns=EXCLUDE_COLUMNS,
        test_size=TEST_SIZE,
        scale=True,
        missing_strategy='mean'
    )
    
    if result is None:
        print("\n✗ Preprocessing failed. Please check your data and configuration.")
        return
    
    X_train, X_test, y_train, y_test, feature_names = result
    
    # ====================================================================
    # STEP 2: MODEL COMPARISON
    # ====================================================================
    print("\n" + "="*70)
    print("STEP 2: MODEL COMPARISON")
    print("="*70)
    
    comparison = ModelComparison()
    comparison.initialize_models(random_state=RANDOM_STATE)
    
    results_df = comparison.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Generate visualizations
    comparison.plot_model_comparison()
    comparison.plot_confusion_matrices()
    
    # Save best model
    best_model_name, best_model = comparison.save_best_model()
    
    # Generate classification report for best model
    comparison.generate_classification_report(y_test)
    
    # ====================================================================
    # STEP 3: FEATURE IMPORTANCE ANALYSIS
    # ====================================================================
    print("\n" + "="*70)
    print("STEP 3: FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    analyzer = FeatureImportanceAnalyzer(feature_names)
    analyzer.comprehensive_analysis(
        X_train, X_test, y_train, y_test,
        random_state=RANDOM_STATE,
        top_n=15
    )
    
    # ====================================================================
    # FINAL SUMMARY
    # ====================================================================
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*70)
    print(f"\n✓ Best Model: {best_model_name}")
    print(f"✓ Best Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
    print(f"✓ Number of Features: {len(feature_names)}")
    print(f"✓ Training Samples: {X_train.shape[0]}")
    print(f"✓ Testing Samples: {X_test.shape[0]}")
    
    print("\n📊 Generated Outputs:")
    print("  - outputs/model_comparison.png")
    print("  - outputs/confusion_matrices.png")
    print("  - outputs/rf_importance.png")
    print("  - outputs/gb_importance.png")
    print("  - outputs/xgb_importance.png (if XGBoost installed)")
    print("  - outputs/importance_comparison.png")
    print("  - outputs/top_features_detailed.png")
    print("  - outputs/feature_importance_report.txt")
    print("  - models/best_model.pkl")
    
    print("\n🎯 Key Findings:")
    if 'Random Forest' in analyzer.importance_results:
        top_5_features = analyzer.importance_results['Random Forest'].head(5)
        print("\nTop 5 Most Important Features:")
        for idx, row in top_5_features.iterrows():
            print(f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f}")
    
    print("\n" + "="*70)
    print("Thank you for using the Diabetes Prediction Pipeline!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Analysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
