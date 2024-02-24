import json
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
smote_model=pickle.load(open('smote_rf', 'rb'))
oneHotEncoder = pickle.load(open('oneHotEncoder', 'rb'))
standardScaler=pickle.load(open('StandardScaler', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data=request.json['data']
    category_features = ['CITY', 'EDUCATION_LEVEL', 'GENDER', 'MARITAL_STATUS', 'OCCUPATION']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    # new_data = oneHotEncoder.transform(np.array(list(data[category_features].values())).reshape(1,-1))
    feature_arr = oneHotEncoder.transform(list(data[category_features].values())).toarray().reshape(1,-1)
    feature_labels = oneHotEncoder.categories_
    feature_labels = np.concatenate(feature_labels)
    encoded_df = pd.DataFrame(feature_arr, columns=feature_labels)
    testdata = pd.concat([data, encoded_df], axis=1)
    testdata.drop(category_features, axis=1, inplace=True)
    testdata['AGE'] = standardScaler.transform(testdata[['AGE']])
    output = smote_model.predict(testdata)
    print(output[0])
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)