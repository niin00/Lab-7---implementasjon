from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the trained model
model = joblib.load('iris_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Create a numpy array with the input data
        data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Make prediction
        prediction = model.predict(data)
        iris_species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

        # Get the predicted species name
        predicted_species = iris_species[prediction[0]]

        return f'<h1>Predicted Species: {predicted_species}</h1>'

    except Exception as e:
        return f'<h1>Error: {str(e)}</h1>'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
