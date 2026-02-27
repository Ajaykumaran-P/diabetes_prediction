"""
Use Pre-trained Model for Blood Sugar Prediction
Load and make predictions with the pre-trained model
"""

import numpy as np
import pandas as pd
import joblib
from optimized_model import OptimizedSugarPredictor
import warnings
warnings.filterwarnings('ignore')


def load_pretrained_model():
    """Load the pre-trained model."""
    try:
        model = OptimizedSugarPredictor.load_model('models/pretrained_sugar_predictor.pkl')
        feature_info = joblib.load('models/feature_info.pkl')
        return model, feature_info
    except FileNotFoundError:
        print("✗ Pre-trained model not found!")
        print("Run: python create_pretrained_model.py first")
        return None, None


def create_sample_patient():
    """Create a sample patient for demonstration."""
    # Sample patient with moderate risk
    patient_data = {
        'uric_acid': 6.5,           # Slightly elevated
        'age': 52,                  # Middle-aged
        'systolic_bp': 135,         # Pre-hypertension
        'bmi': 29.5,                # Overweight
        'diastolic_bp': 85,
        'cholesterol': 210,
        'triglycerides': 160,
        'fasting_glucose': 105,     # Pre-diabetic range
        'hdl_cholesterol': 45,
        'ldl_cholesterol': 140,
        'waist_circumference': 95,
        'hip_circumference': 105,
        'smoking_status': 0,
        'family_history': 1,        # Has family history
        'physical_activity': 3,     # Low activity
        'heart_rate': 78,
        'creatinine': 1.1,
        'albumin': 4.2
    }
    return patient_data


def predict_patient(model, patient_data, feature_info):
    """
    Make prediction for a patient.
    
    Args:
        model: Trained model
        patient_data: Dictionary of patient features
        feature_info: Feature information
    """
    print("\n" + "="*70)
    print("PATIENT BLOOD SUGAR RISK PREDICTION")
    print("="*70)
    
    # Prepare data
    feature_names = feature_info['feature_names']
    X = np.array([[patient_data[f] for f in feature_names]])
    
    # Display patient data
    print("\n📋 Patient Data:")
    print("-" * 70)
    for feature in feature_names[:10]:  # Show top 10 features
        value = patient_data[feature]
        description = feature_info['feature_descriptions'][feature]
        print(f"  {feature:25s} {value:8.2f}  ({description.split(' - ')[1]})")
    
    # Make prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    risk_level = "HIGH RISK ⚠️" if prediction == 1 else "LOW RISK ✓"
    print(f"\n🎯 Prediction: {risk_level}")
    print(f"📊 Probability of Elevated Blood Sugar: {probability[1]*100:.1f}%")
    print(f"📊 Probability of Normal Blood Sugar: {probability[0]*100:.1f}%")
    
    # Risk interpretation
    print("\n💡 Interpretation:")
    if prediction == 1:
        print("  ⚠️  This patient shows HIGH RISK for elevated blood glucose")
        print("  Recommended Actions:")
        print("    • Schedule comprehensive diabetes screening")
        print("    • Review and modify lifestyle factors")
        print("    • Monitor blood glucose levels regularly")
        print("    • Consider dietary and exercise interventions")
        if patient_data['uric_acid'] > 6.0:
            print("    • Pay special attention to uric acid levels")
    else:
        print("  ✓  This patient shows LOW RISK for elevated blood glucose")
        print("  Recommended Actions:")
        print("    • Continue regular health monitoring")
        print("    • Maintain healthy lifestyle")
        print("    • Annual screening recommended")
    
    # Feature importance for this prediction
    print("\n📈 Key Risk Factors for This Patient:")
    importances = model.model.feature_importances_
    feature_importance = sorted(zip(feature_names, importances), 
                                key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(feature_importance[:5], 1):
        value = patient_data[feature]
        print(f"  {i}. {feature:20s} (value: {value:6.2f}, importance: {importance:.4f})")
    
    return prediction, probability


def batch_predict(model, patients_df, feature_info):
    """
    Predict for multiple patients.
    
    Args:
        model: Trained model
        patients_df: DataFrame with patient data
        feature_info: Feature information
    """
    feature_names = feature_info['feature_names']
    
    # Ensure all features are present
    for feature in feature_names:
        if feature not in patients_df.columns:
            print(f"✗ Missing feature: {feature}")
            return None
    
    X = patients_df[feature_names].values
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    results = pd.DataFrame({
        'Patient_ID': range(1, len(patients_df) + 1),
        'Prediction': ['High Risk' if p == 1 else 'Low Risk' for p in predictions],
        'Risk_Probability': probabilities * 100
    })
    
    return results


def main():
    """Main function to demonstrate the pre-trained model."""
    print("\n" + "="*70)
    print("BLOOD SUGAR PREDICTION - PRE-TRAINED MODEL")
    print("NIH-K43 Community Screening Dataset")
    print("="*70)
    
    # Load model
    print("\n⏳ Loading pre-trained model...")
    model, feature_info = load_pretrained_model()
    
    if model is None:
        return
    
    print("✓ Model loaded successfully!")
    print(f"✓ Features: {len(feature_info['feature_names'])}")
    
    # Display feature information
    print("\n" + "="*70)
    print("REQUIRED FEATURES FOR PREDICTION")
    print("="*70)
    print("\nThe model requires the following 18 features:")
    for i, feature in enumerate(feature_info['feature_names'], 1):
        desc = feature_info['feature_descriptions'][feature]
        print(f"{i:2d}. {desc}")
    
    # Create sample patient
    print("\n" + "="*70)
    print("DEMO: SAMPLE PATIENT PREDICTION")
    print("="*70)
    
    sample_patient = create_sample_patient()
    prediction, probability = predict_patient(model, sample_patient, feature_info)
    
    # Example: Predict for multiple patients
    print("\n" + "="*70)
    print("DEMO: BATCH PREDICTION (3 PATIENTS)")
    print("="*70)
    
    # Create sample patients with different risk profiles
    patients = []
    
    # Patient 1: Low risk
    patients.append({
        'uric_acid': 4.5, 'age': 35, 'systolic_bp': 115, 'bmi': 23.5,
        'diastolic_bp': 75, 'cholesterol': 180, 'triglycerides': 100,
        'fasting_glucose': 85, 'hdl_cholesterol': 60, 'ldl_cholesterol': 100,
        'waist_circumference': 80, 'hip_circumference': 95, 'smoking_status': 0,
        'family_history': 0, 'physical_activity': 7, 'heart_rate': 65,
        'creatinine': 0.9, 'albumin': 4.5
    })
    
    # Patient 2: Moderate risk
    patients.append({
        'uric_acid': 6.0, 'age': 48, 'systolic_bp': 130, 'bmi': 28.0,
        'diastolic_bp': 82, 'cholesterol': 205, 'triglycerides': 150,
        'fasting_glucose': 100, 'hdl_cholesterol': 48, 'ldl_cholesterol': 130,
        'waist_circumference': 92, 'hip_circumference': 102, 'smoking_status': 0,
        'family_history': 1, 'physical_activity': 4, 'heart_rate': 72,
        'creatinine': 1.0, 'albumin': 4.3
    })
    
    # Patient 3: High risk
    patients.append({
        'uric_acid': 7.5, 'age': 58, 'systolic_bp': 145, 'bmi': 32.5,
        'diastolic_bp': 92, 'cholesterol': 240, 'triglycerides': 200,
        'fasting_glucose': 115, 'hdl_cholesterol': 38, 'ldl_cholesterol': 160,
        'waist_circumference': 105, 'hip_circumference': 110, 'smoking_status': 1,
        'family_history': 1, 'physical_activity': 2, 'heart_rate': 82,
        'creatinine': 1.3, 'albumin': 4.0
    })
    
    patients_df = pd.DataFrame(patients)
    results = batch_predict(model, patients_df, feature_info)
    
    print("\n📊 Batch Prediction Results:")
    print(results.to_string(index=False))
    
    # Usage instructions
    print("\n" + "="*70)
    print("HOW TO USE THIS MODEL IN YOUR CODE")
    print("="*70)
    print("""
# 1. Load the model
from optimized_model import OptimizedSugarPredictor
import joblib

model = OptimizedSugarPredictor.load_model('models/pretrained_sugar_predictor.pkl')
feature_info = joblib.load('models/feature_info.pkl')

# 2. Prepare patient data (18 features required)
patient = {
    'uric_acid': 6.5,
    'age': 52,
    'systolic_bp': 135,
    'bmi': 29.5,
    # ... (all 18 features)
}

# 3. Make prediction
import numpy as np
feature_names = feature_info['feature_names']
X = np.array([[patient[f] for f in feature_names]])

prediction = model.predict(X)[0]  # 0 = low risk, 1 = high risk
probability = model.predict_proba(X)[0][1]  # Probability of high risk

print(f"Risk: {'High' if prediction == 1 else 'Low'}")
print(f"Probability: {probability*100:.1f}%")
    """)
    
    print("\n" + "="*70)
    print("✓ Demo completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
