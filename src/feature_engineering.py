class FeatureEngineer:
    def __init__(self, data):
        self.data = data

    def select_features(self):
        correlation_matrix = self.data.corr()
        relevant_features = correlation_matrix['target'].abs().sort_values(ascending=False).index
        return self.data[relevant_features]