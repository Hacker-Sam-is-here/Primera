# Primera
A Machine Learning Framework for Training Models from CSV Files

## Description
This project is a machine learning framework that provides tools for loading datasets, preprocessing data, training machine learning models, and making predictions. It utilizes the Random Forest Classifier from the scikit-learn library to perform classification tasks.

## Installation Instructions
To set up the project, ensure you have Python installed, then install the required packages using pip:

```bash
pip install pandas scikit-learn
```

## Usage
1. Import the framework:
   ```python
   from primera import MLFramework
   ```

2. Create an instance of the framework:
   ```python
   ml = MLFramework('path_to_your_dataset.csv')
   ```

3. Load the data:
   ```python
   data = ml.load_data()
   ```

4. Preprocess the data:
   ```python
   processed_data = ml.preprocess_data()
   ```

5. Select features and target:
   ```python
   X, y = ml.feature_selection('target_column_name')
   ```

6. Train the model:
   ```python
   ml.train_model(X, y)
   ```

7. Save the model:
   ```python
   ml.save_model('model_filename.pkl')
   ```

8. Load the model:
   ```python
   ml.load_model('model_filename.pkl')
   ```

9. Make predictions:
   ```python
   predictions = ml.predict(X)
   ```

10. Evaluate the model:
    ```python
    accuracy = ml.evaluate_model(X, y)
    print(f'Accuracy: {accuracy}')
    ```

## Datasets
This project uses datasets in CSV format. Ensure that your dataset is structured correctly for the framework to process it.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License.
