from primera import MLFramework
import pandas as pd

# Initialize the model
model = MLFramework('updated_pollution_dataset.csv')

# Load and preprocess the data
model.load_data()
model.preprocess_data()
X, y = model.feature_selection('Air Quality')

# Train the model
model.train_model(X, y)

# Save the trained model
model.save_model('pollution_model.pkl')

# Single input for prediction
single_input = pd.DataFrame([[29.8, 59.1, 5.2, 17.9, 18.9, 9.2, 1.72, 6.3, 319]], columns=X.columns)

# Make prediction
prediction = model.predict(single_input)

# Map prediction to air quality label
label_mapping = {0: 'Good', 1: 'Moderate', 2: 'Poor', 3: 'Hazardous'}
predicted_label = label_mapping[prediction[0]]

# Print the result
print('Predicted Air Quality:', predicted_label)
