# Heart Disease Prediction with Machine Learning 🫀
This project is an end-to-end machine learning pipeline aimed at predicting the likelihood of heart disease based on clinical data. 

## Overview
Cardiovascular diseases have long been a leading cause of mortality worldwide. In this project, I aim to use machine learning techniques to predict the presence of heart disease based on key health indicators. The dataset used is the Cleveland Heart Disease dataset taken from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/45/heart+disease).

## Data Description
The Cleveland dataset contains 14 attributes and one target column indicating the presence or absence of heart disease. Key features include:

- Age: Age in years.
- Sex: Gender (1 = male, 0 = female).
- Chest Pain Type (cp): 0 to 3.
- Resting Blood Pressure (trestbps): mm Hg.
- Cholesterol (chol): Serum cholesterol in mg/dl.
- Fasting Blood Sugar (fbs): >120 mg/dl (1 = true, 0 = false).
- Resting ECG (restecg): 0 to 2.
- Max Heart Rate (thalach): Achieved.
- Exercise-Induced Angina (exang): 1 = yes, 0 = no.
- ST Depression (oldpeak): Exercise-induced.
- Slope of ST Segment (slope): 0 to 2.
- CA: Number of major vessels colored by fluoroscopy (0–3).
- Thalassemia (thal): 0 to 3.

## Data Preprocessing

The data processing includes handling missing values by replacing these with NaN and imputed with column medians, scaling which is standardized by StandardScaler, as well as train-test spliting the data (80% training, 20% testing). The processed data is then saved in data/processed/cleaned_cleveland.csv.

## Model Training and Evaluation
The ModelTrainer class supports Logistic Regression and Random Forest Classifier.
### Classification Report
<p align="center">
  <img width="496" alt="Screenshot 2025-01-13 at 1 13 53 AM" src="https://github.com/user-attachments/assets/65d44189-1229-4b48-91f4-92bbd9b4f16b" />
</p>

### Confusion Matrix
<p align="center">
  <img width="542" alt="Screenshot 2025-01-13 at 1 14 49 AM" src="https://github.com/user-attachments/assets/88beed49-8c52-4725-8270-4000f6ef02d8" />
</p>

### Precision Recall Curve
<p align="center">
  <img width="639" alt="Screenshot 2025-01-13 at 1 16 14 AM" src="https://github.com/user-attachments/assets/c5781ef4-07fb-428d-a85f-df13d496fc18" />
</p>

The model achieved an accuracy of 96%. The precision recall curve confirms the robustness of the model in predicting positive cases. However, the confusion matrix reveals that the model struggles with negative cases due to data imbalance. This is one of the main future improvements that I am taking into consideration. 
## Future Improvements
### Data Augmentation
The current dataset suffers from a significant imbalance between the positive and negative classes. This impacts the model's ability to correctly classify the minority class, as evidenced by the low precision and recall for the "No Disease" class. A possible solution would be using data augmentation techniques, such as oversampling using SMOTE (Synthetic Minority Oversampling Technique) or generating synthetic data using GANs (Generative Adversarial Networks). 

### Model Experimentation
While the project primarily uses logistic regression and random forest models, experimenting with other algorithms such as Support Vector Machines (SVM) can be effective for high-dimensional data and yield better classification boundaries. Gradient Boosting models, such as XGBoost or LightGBM, often outperform traditional models by leveraging ensemble techniques to reduce bias and variance. Hyperparameter tuning via Grid Search or Bayesian Optimization should accompany these experiments to maximize each model's potential.

### Explainability
As machine learning models are increasingly used in high-stakes domains like healthcare, interpretability becomes critical. Tools like SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) provide insights into individual predictions by quantifying the contribution of each feature, which gives the model more transparency and trustworthiness. 

## Web Application
In this project, I also deployed an app that makes use of the model, where users can input clinical data and receive predictions. You can check out the app by:
1. Navigate to the `webapp/` directory.
2. Run `app.py`:
 ```bash
 python webapp/app.py

