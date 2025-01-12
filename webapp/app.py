from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

# Load model and scaler
model_path = os.path.join(os.path.dirname(__file__), '../models/heart_disease_model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), '../models/scaler.pkl')
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    result = 'Disease' if prediction[0] == 1 else 'No Disease'
    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)
