import json
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
smote_model=pickle.load(open('smote_rf.pkl', 'rb'))
oneHotEncoder = pickle.load(open('oneHotEncoder.pkl', 'rb'))
standardScaler=pickle.load(open('StandardScaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data=request.json['data']
    category_features = ['CITY', 'EDUCATION_LEVEL', 'GENDER', 'MARITAL_STATUS', 'OCCUPATION']
    # transform json data into dataframe
    testdata = pd.DataFrame([data])

    feature_arr = oneHotEncoder.transform(testdata[category_features]).toarray()
    feature_labels = oneHotEncoder.categories_
    feature_labels = np.concatenate(feature_labels)
    encoded_df = pd.DataFrame(feature_arr, columns=feature_labels)
    testdata = pd.concat([testdata, encoded_df], axis=1)
    testdata.drop(category_features, axis=1, inplace=True)

    testdata['AGE'] = standardScaler.transform(testdata[['AGE']])

    output = smote_model.predict(testdata)
    output_serializable = int(output[0])
    return jsonify(output_serializable)

@app.route('/predict', methods=['POST'])
def predict():
    form_values = request.form.values()
    testdata = pd.DataFrame([list(form_values)], columns=request.form.keys())
    print("************************")
    print(testdata)
    print("************************")
    category_features = ['CITY', 'EDUCATION_LEVEL', 'GENDER', 'MARITAL_STATUS', 'OCCUPATION']
    # transform json data into dataframe

    feature_arr = oneHotEncoder.transform(testdata[category_features]).toarray()
    feature_labels = oneHotEncoder.categories_
    feature_labels = np.concatenate(feature_labels)
    encoded_df = pd.DataFrame(feature_arr, columns=feature_labels)
    testdata = pd.concat([testdata, encoded_df], axis=1)
    testdata.drop(category_features, axis=1, inplace=True)

    testdata['AGE'] = standardScaler.transform(testdata[['AGE']])

    output = smote_model.predict(testdata)
    output_serializable = int(output[0])
    return render_template("home.html", prediction_text="The donor prediction is {}".format(output_serializable))

if __name__ == "__main__":
    app.run(debug=True, port=5001)