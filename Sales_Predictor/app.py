import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import traceback # Import traceback module for detailed error printing

# Setup Flask app
app = Flask(__name__)

# Define the base directory of the application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the path to the trained model file
model_path = os.path.join(BASE_DIR, "best_model_xgboost.pkl")

# Load the trained model
try:
    model = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Set model to None if loading fails

# Define the list of expected features for the model
# This should match the columns of X_train used during training
# Based on the traceback, the model was trained with these features:
expected_features = ['TV', 'Radio', 'Newspaper', 'Total_Advertising', 'TV_Radio_Interaction']


@app.route('/')
def home():
    """Renders the index.html page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST']) # Specify that this route accepts POST requests
def predict():
    """Handles prediction requests."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        tv = float(data['tv'])
        radio = float(data['radio'])
        newspaper = float(data['newspaper'])

        # Create a DataFrame with the new data and perform feature engineering
        # Only create features that the loaded model was trained on
        new_data_df = pd.DataFrame({
            "TV": [tv],
            "Radio": [radio],
            "Newspaper": [newspaper]
        })

        # Apply only the feature engineering steps done before training the saved model
        new_data_df["Total_Advertising"] = new_data_df["TV"] + new_data_df["Radio"] + new_data_df["Newspaper"]
        new_data_df["TV_Radio_Interaction"] = new_data_df["TV"] * new_data_df["Radio"]
        # Do NOT create Radio_Newspaper_Interaction, TV_Newspaper_Interaction, or Newspaper_log here


        # Ensure column order matches the training data
        # This is crucial for the model to receive features in the expected order
        # Use the expected_features list to reindex the DataFrame
        new_data_df = new_data_df[expected_features]

        # *** Debugging: Print columns before prediction ***
        print("Columns of DataFrame before prediction:", new_data_df.columns)
        print("Expected features:", expected_features)
        # **************************************************

        # Make prediction using the loaded model
        # The loaded model (Pipeline) should handle scaling and prediction
        predicted_sales_transformed = model.predict(new_data_df)

        # Inverse transform the prediction if the target variable was transformed
        # In this case, we applied a square root transformation to 'Sales'
        predicted_sales = predicted_sales_transformed**2

        # Convert numpy float to standard Python float for JSON serialization
        predicted_sales_python_float = float(predicted_sales[0])


        return jsonify({'predicted_sales': predicted_sales_python_float})

    except KeyError as e:
        # This would happen if the incoming JSON is missing tv, radio, or newspaper keys
        traceback.print_exc() # Print traceback to console
        return jsonify({'error': f'Missing expected data in request: {e}'}), 400
    except ValueError as e:
        # This would happen if tv, radio, or newspaper values cannot be converted to float
        traceback.print_exc() # Print traceback to console
        return jsonify({'error': f'Invalid data format: {e}. Please ensure values are numbers.'}), 400
    except Exception as e:
        # Catch any other exceptions and print traceback
        traceback.print_exc() # Print traceback to console
        return jsonify({'error': f'An internal error occurred during prediction: {e}'}), 500


if __name__ == '__main__':
    # Run the Flask development server
    # In a production environment, you would use a production-ready server like Gunicorn or uWSGI
    app.run(debug=True)