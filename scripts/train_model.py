# import sys
# import os
# import joblib

# # Add the project root directory to the Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from src.data_preprocessing import DataPreprocessor
# from src.model import ModelTrainer

# # Load and preprocess data
# file_path = './data/raw/cleveland.data'  # Updated to use cleveland.data
# preprocessor = DataPreprocessor(file_path)
# data = preprocessor.load_data()
# X_train, X_test, y_train, y_test, scaler = preprocessor.preprocess_data()

# # Train and evaluate model
# trainer = ModelTrainer(model_type='logistic')
# model = trainer.build_model()
# model = trainer.train(X_train, y_train)
# acc, report = trainer.evaluate(X_test, y_test)

# # Save model and scaler
# os.makedirs('./models', exist_ok=True)
# joblib.dump(model, './models/heart_disease_model.pkl')
# joblib.dump(scaler, './models/scaler.pkl')

# print(f"Model Accuracy: {acc}")
# print("Classification Report:")
# print(report)
import sys
import numpy as np
import os
import joblib
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor
from src.model import ModelTrainer

# Ensure directories exist
os.makedirs('./data/processed', exist_ok=True)
os.makedirs('./models', exist_ok=True)
os.makedirs('./results', exist_ok=True)

# Visualization functions
def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks(range(len(classes)), classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_precision_recall_curve(y_true, y_scores, output_path):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(output_path)
    plt.close()

def plot_feature_importance(model, feature_names, output_path):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Load and preprocess data
file_path = './data/raw/cleveland.data'
preprocessor = DataPreprocessor(file_path)
data = preprocessor.load_data()
X_train, X_test, y_train, y_test, scaler = preprocessor.preprocess_data()

# train and evaluate model
trainer = ModelTrainer(model_type='logistic')
model = trainer.build_model()
model = trainer.train(X_train, y_train)
acc, report = trainer.evaluate(X_test, y_test)

joblib.dump(model, './models/heart_disease_model.pkl')
joblib.dump(scaler, './models/scaler.pkl')

y_pred = model.predict(X_test)
plot_confusion_matrix(
    y_test, y_pred,
    classes=["No Disease", "Disease"],
    output_path="./results/confusion_matrix.png"
)

print(f"Model Accuracy: {acc}")
print("Classification Report:")
print(report)
