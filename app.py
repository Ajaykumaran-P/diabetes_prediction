"""
Flask Web Application for Blood Sugar Prediction
Pre-trained Random Forest Model
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from optimized_model import OptimizedSugarPredictor
import os

app = Flask(__name__)

# Use absolute paths so it works both locally and on Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'pretrained_sugar_predictor.pkl')
FEATURE_INFO_PATH = os.path.join(BASE_DIR, 'models', 'feature_info.pkl')

# Auto-create model if not found (important for Render cold deploys)
if not os.path.exists(MODEL_PATH):
    print("⏳ Model not found — building pre-trained model...")
    try:
        import subprocess, sys
        subprocess.run([sys.executable, os.path.join(BASE_DIR, 'create_pretrained_model.py')], check=True)
        print("✓ Model built successfully!")
    except Exception as build_err:
        print(f"✗ Failed to build model: {build_err}")

try:
    model = OptimizedSugarPredictor.load_model(MODEL_PATH)
    feature_info = joblib.load(FEATURE_INFO_PATH)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None
    feature_info = None


@app.route('/')
def index():
    """Home page with prediction form."""
    if model is None:
        return render_template('error.html', 
                             error="Model not found. Please run create_pretrained_model.py first.")
    
    return render_template('index.html', 
                         features=feature_info['feature_descriptions'])


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get form data
        patient_data = {}
        feature_names = feature_info['feature_names']
        
        for feature in feature_names:
            value = request.form.get(feature)
            if value is None or value == '':
                return jsonify({'error': f'Missing value for {feature}'}), 400
            patient_data[feature] = float(value)
        
        # Prepare data for prediction
        X = np.array([[patient_data[f] for f in feature_names]])
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        # Get feature importance
        importances = model.model.feature_importances_
        top_features = sorted(zip(feature_names, importances, 
                                 [patient_data[f] for f in feature_names]),
                            key=lambda x: x[1], reverse=True)[:5]
        
        # Prepare response
        result = {
            'prediction': 'High Risk' if prediction == 1 else 'Low Risk',
            'risk_class': 'high' if prediction == 1 else 'low',
            'high_risk_probability': float(probability[1] * 100),
            'low_risk_probability': float(probability[0] * 100),
            'top_features': [
                {
                    'name': name.replace('_', ' ').title(),
                    'value': float(value),
                    'importance': float(importance * 100)
                }
                for name, importance, value in top_features
            ],
            'recommendations': get_recommendations(prediction, patient_data)
        }
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for JSON predictions."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        feature_names = feature_info['feature_names']
        
        # Validate all features present
        missing_features = [f for f in feature_names if f not in data]
        if missing_features:
            return jsonify({
                'error': f'Missing features: {", ".join(missing_features)}'
            }), 400
        
        # Prepare data
        X = np.array([[data[f] for f in feature_names]])
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        result = {
            'prediction': int(prediction),
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
            'probability': {
                'high_risk': float(probability[1]),
                'low_risk': float(probability[0])
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch', methods=['GET', 'POST'])
def batch_predict():
    """Batch prediction page."""
    if request.method == 'GET':
        return render_template('batch.html')
    
    # Handle batch prediction (CSV upload)
    # Implementation for CSV file upload
    return jsonify({'message': 'Batch prediction feature coming soon!'})


def get_recommendations(prediction, patient_data):
    """Generate personalized recommendations."""
    recommendations = []
    
    if prediction == 1:  # High Risk
        recommendations.append("⚠️ Schedule comprehensive diabetes screening immediately")
        recommendations.append("📋 Consult with healthcare professional for detailed evaluation")
        
        if patient_data['uric_acid'] > 6.0:
            recommendations.append("🔬 Monitor and manage elevated uric acid levels")
        
        if patient_data['bmi'] > 28:
            recommendations.append("🏃 Weight management program recommended (current BMI: {:.1f})".format(patient_data['bmi']))
        
        if patient_data['systolic_bp'] > 130:
            recommendations.append("💓 Blood pressure management needed (current: {:.0f} mmHg)".format(patient_data['systolic_bp']))
        
        if patient_data['physical_activity'] < 5:
            recommendations.append("🚴 Increase physical activity level")
        
        recommendations.append("🍎 Consider dietary modifications and lifestyle changes")
        recommendations.append("📊 Regular blood glucose monitoring essential")
    else:  # Low Risk
        recommendations.append("✅ Maintain current healthy lifestyle")
        recommendations.append("📅 Continue annual health screenings")
        recommendations.append("💪 Stay physically active")
        recommendations.append("🥗 Maintain balanced diet")
        
        if patient_data['age'] > 45:
            recommendations.append("👨‍⚕️ Regular checkups recommended due to age")
    
    return recommendations


@app.route('/info')
def info():
    """Information about the model."""
    return render_template('info.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))

    if not os.path.exists(MODEL_PATH):
        print("\n" + "="*70)
        print("⚠️  PRE-TRAINED MODEL NOT FOUND")
        print("="*70)
        print("\nPlease run the following command first:")
        print("  python create_pretrained_model.py")
        print("\nThis will create the required model files.")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("🌐 BLOOD SUGAR PREDICTION WEB APP")
        print("="*70)
        print("\n✓ Model loaded successfully")
        print(f"✓ Server starting on port {port}...")
        print("\n📱 Open your browser and go to:")
        print(f"   http://localhost:{port}")
        print("\n🔌 API Endpoint:")
        print(f"   POST http://localhost:{port}/api/predict")
        print("\nPress CTRL+C to stop the server")
        print("="*70 + "\n")

        app.run(debug=False, host='0.0.0.0', port=port)
