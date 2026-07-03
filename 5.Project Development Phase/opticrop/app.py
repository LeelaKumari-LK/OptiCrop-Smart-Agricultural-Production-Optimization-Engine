import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/findyourcrop')
def findyourcrop():
    return render_template('findyourcrop.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]

    input_df = pd.DataFrame([int_features], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    input_df['K'] = np.log1p(input_df['K'])

    features_scaled = scaler.transform(input_df)
    prediction = model.predict(features_scaled)
    output = prediction[0]

    return render_template('findyourcrop.html', prediction_text='Best crop for given conditions is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)