import joblib

class Predictor:
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, input_features):
        scaled_features = self.scaler.transform([input_features])
        return self.model.predict(scaled_features)