from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)  # Only define Flask app once

# Step 1: Load the pre-trained churn model
try:
    model = joblib.load('churn_model.pkl')  # Ensure this path is correct
except FileNotFoundError:
    model = None
    print("Model file not found. Please ensure the trained model is saved as 'churn_model.pkl'.")


# Define a simple route to check if the server is running
@app.route("/")
def home():
    return "Hello World"


# Step 2: Define a route for predicting churn
@app.route("/predict_churn", methods=['POST'])
def predict_churn():
    if model is None:
        return jsonify({"error": "Model not loaded. Please ensure the model file exists."}), 500

    try:
        # Step 3: Extract data from the request
        data = request.json
        features = [[data['Total_Spend'], data['Num_of_Purchases'], data['Last_Purchase_Days_Ago']]]

        # Step 4: Predict churn using the model
        prediction = model.predict(features)
        return jsonify({'churn': int(prediction[0])})
    except KeyError as e:
        return jsonify({"error": f"Missing required input data: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Ensure this is uncommented and correct
