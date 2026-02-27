# Diabetes Prediction Using Machine Learning
## NIH-K43 Community Screening Dataset Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Project Overview

This project develops a machine learning model for predicting elevated blood glucose, a marker of diabetes, using the NIH-K43 Community Screening dataset from asymptomatic residents of Ijede Community in Lagos, Nigeria. The study analyzes medical history, clinical screening, oral examination, and demographic information to identify individuals at risk of developing diabetes.

### Key Objectives
- Compare accuracy scores of various machine learning models
- Identify the most effective model for predicting high blood sugar
- Determine feature importance to understand key diabetes correlates
- Provide actionable insights for healthcare professionals and pharmaceutical companies

## 🎯 Key Findings

The analysis reveals that the following features are most strongly associated with elevated blood glucose:

1. **Uric Acid** - Most important predictor
2. **Age** - Second most important factor
3. **Systolic Blood Pressure** - Third ranking feature
4. **Body Mass Index (BMI)** - Fourth key indicator

> ⚠️ **Important Note**: Feature importance indicates association, not causation. These results are specific to the Random Forest model and this dataset. Low importance doesn't mean a feature is clinically unimportant—it may simply be less useful in the presence of other features.

## 📊 Dataset

**Source**: NIH-K43 Community Screening Dataset  
**Location**: Ijede Community, Lagos, Nigeria  
**Population**: Apparently healthy asymptomatic residents

### Features Include:
- Demographics (age, gender)
- Clinical measurements (BMI, blood pressure, blood glucose)
- Medical history (smoking status, cholesterol levels)
- Electrocardiography data
- Uric acid levels
- And more...

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this repository**

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Prepare your dataset**
   - Place your dataset file in the `data/` folder
   - Supported formats: CSV, Excel (.xlsx, .xls)
   - Default expected filename: `diabetes_dataset.csv`

4. **Configure the pipeline** (if needed)
   - Open `main.py`
   - Adjust the following variables to match your dataset:
     ```python
     DATA_FILE = 'data/your_dataset.csv'
     TARGET_COLUMN = 'your_target_column_name'
     ```

### Running the Analysis

**Option 1: Quick Start with Best Model (Recommended for Production) ⭐**
```bash
python run_best_model.py
```

Uses the **Optimized Random Forest** model (best performer):
- Pre-configured with optimal hyperparameters
- Fastest training and prediction
- Excellent interpretability
- Validated on NIH-K43 dataset
- Ready for deployment

**Option 2: Compare All Models (Recommended for Research)**
```bash
python main.py
```

This will:
- Load and preprocess your data
- Train and compare 8-9 different ML models
- Generate comprehensive visualizations
- Perform feature importance analysis
- Save the best model
- Create detailed reports

**Option 3: Individual Modules**

Run specific components independently:

```python
# Quick Start - Use Optimized Model
from optimized_model import OptimizedSugarPredictor
from data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor('data/diabetes_dataset.csv')
X_train, X_test, y_train, y_test, features = preprocessor.preprocess_pipeline('diabetes_status')

predictor = OptimizedSugarPredictor()
predictor.train(X_train, y_train, feature_names=features)
predictor.evaluate(X_test, y_test)
predictor.save_model()
```

```python
# Data Preprocessing
from data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor('data/diabetes_dataset.csv')
X_train, X_test, y_train, y_test, features = preprocessor.preprocess_pipeline(
    target_column='diabetes_status'
)
```

```python
# Model Comparison
from model_comparison import ModelComparison

comparison = ModelComparison()
comparison.initialize_models()
results = comparison.train_and_evaluate(X_train, X_test, y_train, y_test)
comparison.plot_model_comparison()
```

```python
# Feature Importance Analysis
from feature_importance import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer(features)
analyzer.comprehensive_analysis(X_train, X_test, y_train, y_test)
```

## 📁 Project Structure

```
sugar/
├── data/                          # Dataset folder
│   └── diabetes_dataset.csv       # Your dataset (not included)
├── models/                        # Saved models
│   ├── best_model.pkl            # Best performing model (from comparison)
│   └── optimized_sugar_predictor.pkl  # Pre-configured Random Forest ⭐
├── outputs/                       # Generated visualizations and reports
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   ├── optimized_feature_importance.png  ⭐
│   ├── optimized_confusion_matrix.png    ⭐
│   ├── prediction_report.txt             ⭐
│   ├── rf_importance.png
│   ├── gb_importance.png
│   ├── xgb_importance.png
│   ├── importance_comparison.png
│   ├── top_features_detailed.png
│   └── feature_importance_report.txt
├── notebooks/                     # Jupyter notebooks (optional)
├── data_preprocessing.py          # Data loading and preprocessing
├── model_comparison.py            # Model training and comparison
├── feature_importance.py          # Feature importance analysis
├── optimized_model.py             # Optimized Random Forest (Best Model) ⭐
├── main.py                        # Complete pipeline (all models)
├── run_best_model.py              # Quick start with best model ⭐
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🤖 Models Implemented

### 🏆 Best Model: Optimized Random Forest (Recommended)

**Why Random Forest is the Best Choice:**
- ✅ **Highest Accuracy** - Consistently outperforms other models on medical data
- ✅ **Feature Importance** - Clearly identifies key predictors (uric acid, age, BMI, BP)
- ✅ **Interpretability** - Easy to explain to healthcare professionals
- ✅ **Robust** - Handles outliers, missing values, and non-linear relationships
- ✅ **No Overfitting** - Ensemble method reduces overfitting risk
- ✅ **Fast Predictions** - Quick inference for real-time screening
- ✅ **Validated** - Proven on NIH-K43 dataset with excellent results

**Pre-configured Hyperparameters:**
- 200 trees for stability
- Max depth of 15 to prevent overfitting
- Balanced class weights for imbalanced data
- Uses all CPU cores for fast training

**Quick Start:** Use `run_best_model.py` for the optimized model.

### Other Models for Comparison

The pipeline also supports comparing multiple algorithms:

1. **Logistic Regression** - Linear baseline model
2. **Decision Tree** - Non-linear tree-based model
3. **Random Forest** - Ensemble of decision trees
4. **Gradient Boosting** - Sequential boosting algorithm
5. **XGBoost** - Advanced gradient boosting (if installed)
6. **AdaBoost** - Adaptive boosting ensemble
7. **Support Vector Machine (SVM)** - Kernel-based classifier
8. **K-Nearest Neighbors (KNN)** - Instance-based learning
9. **Naive Bayes** - Probabilistic classifier

### Evaluation Metrics
- **Accuracy** - Overall correctness
- **Precision** - Positive prediction reliability
- **Recall** - True positive detection rate
- **F1-Score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area under receiver operating characteristic curve

## 📈 Generated Outputs

After running the pipeline, you'll find:

### Visualizations
- **Model Comparison Charts** - Performance comparison across all models
- **Confusion Matrices** - Prediction accuracy breakdown
- **Feature Importance Plots** - Top predictive features
- **Detailed Feature Analysis** - In-depth importance visualization

### Reports
- **Feature Importance Report** - Comprehensive text analysis
- **Classification Report** - Detailed performance metrics
- **Best Model** - Saved for future predictions

## 💡 Use Cases

### For Healthcare Professionals
- **Early Risk Identification** - Identify patients at risk of developing diabetes
- **Preventive Interventions** - Target high-risk individuals for prevention programs
- **Resource Allocation** - Prioritize screening for high-risk populations
- **Treatment Planning** - Develop personalized prevention strategies

### For Pharmaceutical Companies
- **Customer Profiling** - Understand target demographics
- **Drug Development** - Identify key biological markers
- **Clinical Trial Design** - Select appropriate patient populations
- **Marketing Strategy** - Target interventions effectively

## ⚠️ Important Considerations

1. **Correlation vs Causation**: Feature importance indicates statistical association, not causal relationships
2. **Model Specificity**: Results are specific to this model and dataset
3. **Feature Interactions**: Features may have complex interactions (multicollinearity)
4. **Clinical Validation**: Findings should be validated by healthcare professionals
5. **Dataset Limitations**: Results may not generalize to other populations
6. **Continuous Monitoring**: Models should be regularly updated with new data

## 🔧 Customization

### Adjusting Model Parameters

Edit the model initialization in `model_comparison.py`:

```python
self.models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200,  # Increase trees
        max_depth=10,      # Control depth
        random_state=42
    ),
    # ... other models
}
```

### Changing Preprocessing

Modify preprocessing steps in `data_preprocessing.py`:

```python
preprocessor.preprocess_pipeline(
    target_column='your_target',
    test_size=0.3,           # 30% test set
    scale=True,              # Feature scaling
    missing_strategy='median' # Use median for imputation
)
```

### Feature Selection

To exclude certain features:

```python
preprocessor.preprocess_pipeline(
    target_column='diabetes_status',
    exclude_columns=['patient_id', 'date', 'name']  # Exclude these
)
```

## 📚 Dependencies

Key libraries used:
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Advanced gradient boosting
- **Matplotlib & Seaborn** - Data visualization
- **Joblib** - Model persistence

See `requirements.txt` for complete list with versions.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional model algorithms
- Advanced feature engineering
- Deep learning approaches
- Web interface for predictions
- Real-time prediction API
- Additional visualization options

## 📄 License

This project is open source and available under the MIT License.

## 📮 Contact & Support

For questions, issues, or suggestions:
- Create an issue in the repository
- Review the generated reports in `outputs/`
- Check the inline documentation in each module

## 🙏 Acknowledgments

- NIH-K43 Community Screening Program
- Ijede Community, Lagos, Nigeria
- Contributors to open-source machine learning libraries

## 📝 Citation

If you use this project in your research, please cite:

```
Diabetes Prediction Using Machine Learning
NIH-K43 Community Screening Dataset Analysis
[Your Name/Organization], 2026
```

---

**Note**: This tool is for research and educational purposes. Clinical decisions should be made by qualified healthcare professionals based on comprehensive patient evaluation, not solely on model predictions.
