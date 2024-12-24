import pandas as pd
from primera import MLFramework
from sklearn.model_selection import train_test_split

# Load the dataset and preprocess it
framework = MLFramework('psychological_state_dataset.csv')
framework.load_data()
framework.preprocess_data()

# Feature selection
X, y = framework.feature_selection('Psychological State')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
framework.train_model(X_train, y_train)

# Save the trained model
framework.save_model('psychological_state_model.pkl')

# Load the model for prediction
framework.load_model('psychological_state_model.pkl')

# Define input parameters for prediction
input_data = {
    'ID': [1],  # Example value
    'Time': [521.0],  # Example numerical value
    'HRV (ms)': [50],  # Example value
    'GSR (μS)': [1.2],  # Example value
    'EEG Power Bands': [0.5],  # Wrapped in a list to maintain structure
    'Blood Pressure (mmHg)': [120],  # Example value
    'Oxygen Saturation (%)': [98],  # Example value
    'Heart Rate (BPM)': [75],  # Example value
    'Ambient Noise (dB)': [30],  # Example value
    'Cognitive Load': [5],  # Example value
    'Mood State': [1],  # Example value (encoded)
    # 'Psychological State': [1],  # Example value (Anxious)
    'Respiration Rate (BPM)': [16],  # Example value
    'Skin Temp (°C)': [36.5],  # Example value
    'Focus Duration (s)': [120],  # Example value
    'Task Type': [1],  # Example value (encoded)
    'Age': [25],  # Example value
    'Gender': [0],  # Example value (encoded)
    'Educational Level': [2],  # Example value (encoded)
    'Study Major': [1]  # Example value (encoded)
}

# Create a DataFrame for the input data
input_df = pd.DataFrame(input_data)

# Predict the psychological state for the input parameters
single_prediction = framework.predict(input_df)
state_mapping = {0: 'Stressed', 1: 'Anxious', 2: 'Relaxed', 3: 'Focused'}
predicted_state = state_mapping[single_prediction[0]]

# Output the predicted psychological state
print(predicted_state)
