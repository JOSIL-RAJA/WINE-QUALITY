from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('wine_quality_model.pkl', 'rb'))

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Make a prediction
    prediction = model.predict(final_features)
    
    # Return the result
    return render_template('index.html', prediction_text=f'Predicted Wine Quality: {prediction[0]}')

# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080,debug=True)
