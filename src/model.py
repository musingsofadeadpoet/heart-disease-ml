from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class ModelTrainer:
    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.model = None

    def build_model(self):
        if self.model_type == 'logistic':
            self.model = LogisticRegression()
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=42)
        else:
            raise ValueError("Unsupported model type")
        return self.model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return acc, report