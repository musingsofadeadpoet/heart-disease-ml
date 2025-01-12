import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        cleaned_rows = []
        with open(self.file_path, 'r', encoding='latin1', errors='replace') as f:
            lines = f.readlines()
        for line in lines:
            values = line.strip().split()
            # Ensure consistent number of columns
            if len(values) < len(column_names):
                values.extend([np.nan] * (len(column_names) - len(values)))  # Pad missing values with NaN
            elif len(values) > len(column_names):
                values = values[:len(column_names)]  # Truncate extra values
            # Replace placeholders like '-9'
            values = [np.nan if v == '-9' else v for v in values]
            cleaned_rows.append(values)
        # Create DataFrame
        self.data = pd.DataFrame(cleaned_rows, columns=column_names)
        self.data = self.data.apply(pd.to_numeric, errors='coerce')  # Convert to numeric
        print("Missing values per column before imputation:")
        print(self.data.isna().sum())  # Log missing values per column
        
        # Impute missing values
        for column in self.data.columns:
            if self.data[column].dtype in ['float64', 'int64']:
                self.data[column].fillna(self.data[column].median(), inplace=True)
            else:
                self.data[column].fillna(self.data[column].mode()[0], inplace=True)

        print(f"Data shape after imputing missing values: {self.data.shape}")
        self.data.to_csv('./data/processed/cleaned_cleveland.csv', index=False, encoding='utf-8')
        return self.data

    def preprocess_data(self):
        # Feature-target split
        X = self.data.drop('target', axis=1)
        y = self.data['target'].apply(lambda x: 1 if x > 0 else 0)
        
        print(f"Dataset size before train-test split: {X.shape[0]} rows")
        # Check for minimum data size
        if X.shape[0] < 5:
            raise ValueError("Insufficient data after preprocessing. Please check the dataset.")

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, scaler