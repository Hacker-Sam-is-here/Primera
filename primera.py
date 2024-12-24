import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class MLFramework:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = None

    def load_data(self):
        """Load data from a CSV file."""
        self.data = pd.read_csv(self.file_path)
        return self.data

    def preprocess_data(self):
        """Preprocess the data and convert categorical data to numerical values."""
        for column in self.data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            self.data[column] = le.fit_transform(self.data[column])
        return self.data

    def feature_selection(self, target_column):
        """Select relevant features for training."""
        # Select features based on the provided target column
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        return X, y

    def train_model(self, X, y):
        """Train the machine learning model."""
        self.model = RandomForestClassifier()
        self.model.fit(X, y)

    def save_model(self, filename):
        """Save the trained model to a file."""
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self, filename):
        """Load the trained model from a file."""
        with open(filename, 'rb') as file:
            self.model = pickle.load(file)

    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)

    def evaluate_model(self, X, y):
        """Evaluate the model's performance."""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
