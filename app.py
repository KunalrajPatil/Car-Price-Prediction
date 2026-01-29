from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Load dataset
car = pd.read_csv('cleaned car.csv')

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())

    # company -> models mapping
    company_model_map = {}
    for company in companies:
        company_model_map[company] = sorted(
            car[car['company'] == company]['name'].unique()
        )

    return render_template(
        'index.html',
        companies=companies,
        years=years,
        fuel_types=fuel_types,
        company_model_map=company_model_map
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        df = pd.DataFrame([[
            data['car_models'],
            data['company'],
            int(data['year']),
            int(data['kilo_driven']),
            data['fuel_type']
        ]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

        prediction = model.predict(df)[0]
        return jsonify(price=round(prediction, 2))

    except Exception as e:
        return jsonify(error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
