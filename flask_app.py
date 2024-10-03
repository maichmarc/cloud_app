# Using flask app to make an api
# import necessary libraries
from flask import Flask, jsonify, request, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# create a Flask app
app = Flask(__name__)


@app.route('/')  # , methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def price_predict():
    model = pickle.load(open('model.pickle', 'rb'))
    age = request.form.get('age')
    sex = request.form.get('sex')
    bmi = request.form.get('bmi')
    children = request.form.get('children')
    smoker = request.form.get('smoker')
    region = request.form.get('region')
    encoder = LabelEncoder()

    test_df = pd.DataFrame({'age': [age], 'sex': [sex], 'bmi': [bmi], 'children': [children], 'smoker': [smoker],
                            'region': [region]})
    test_df['sex'] = encoder.fit_transform(test_df['sex'])
    test_df['smoker'] = encoder.fit_transform(test_df['smoker'])
    test_df['region'] = encoder.fit_transform(test_df['region'])
    pred_price = model.predict(test_df)
    output = round(pred_price[0], 2)
    return render_template('index.html', prediction_text='Charges should be ${}'.format(output))


if __name__ == '__main__':
    app.run(debug = True)