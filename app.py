import json
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template, redirect, send_from_directory
import numpy as np
import pandas as pd
import pymongo

# mongodb+srv://DonorPredictionApp:9WADs0jcTC4tYg41@donorpredictioncluster0.2rauyga.mongodb.net/

app = Flask(__name__)


client = pymongo.MongoClient("mongodb+srv://DonorPredictionApp:9WADs0jcTC4tYg41@donorpredictioncluster0.2rauyga.mongodb.net/")
db = client['testSet']


# Load the model
smote_model=pickle.load(open('smote_rf.pkl', 'rb'))
oneHotEncoder = pickle.load(open('oneHotEncoder.pkl', 'rb'))
standardScaler=pickle.load(open('StandardScaler.pkl', 'rb'))

@app.route('/',  methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/multiple_prediction')
def multiple_prediction():
    return render_template('multiple_prediction.html')

@app.route('/single_prediction')
def single_prediction():
    return render_template('single_prediction.html')

@app.route('/styles.css')
def styles():
    return send_from_directory('static', 'styles.css')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data=request.json['data']
    category_features = ['CITY', 'EDUCATION_LEVEL', 'GENDER', 'MARITAL_STATUS', 'OCCUPATION']
    # transform json data into dataframe
    testdata = pd.DataFrame([data])
    # processing the testing data
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

@app.route('/single_prediction_result', methods=['POST'])
def single_prediction_result():
    form_values = request.form.values()
    testdata = pd.DataFrame([list(form_values)], columns=request.form.keys())
    category_features = ['CITY', 'EDUCATION_LEVEL', 'GENDER', 'MARITAL_STATUS', 'OCCUPATION']
    feature_arr = oneHotEncoder.transform(testdata[category_features]).toarray()
    feature_labels = oneHotEncoder.categories_
    feature_labels = np.concatenate(feature_labels)
    encoded_df = pd.DataFrame(feature_arr, columns=feature_labels)
    testdata = pd.concat([testdata, encoded_df], axis=1)
    testdata.drop(category_features, axis=1, inplace=True)

    testdata['AGE'] = standardScaler.transform(testdata[['AGE']])

    output = smote_model.predict(testdata)
    output_serializable = int(output[0])
    return render_template("single_prediction.html", prediction_text="This person is a potential donor") if output_serializable == 1 else render_template("single_prediction.html", prediction_text="This person is not a potential donor")

@app.route('/multiple_prediction_result', methods=['POST'])
def multiple_prediction_result():
    if 'CSV File' not in request.files:
            return 'No file part'
    csv_file = request.files['CSV File']
    filename = csv_file.filename
    filename_without_extension = filename.rsplit('.', 1)[0]
    testdata0 = pd.read_csv(csv_file)
    testdata = testdata0.copy()
    testdata['AGE'].fillna(testdata['AGE'].mean(), inplace=True)
    category_features = ['CITY', 'EDUCATION_LEVEL', 'GENDER', 'MARITAL_STATUS', 'OCCUPATION']
    feature_arr = oneHotEncoder.transform(testdata[category_features]).toarray()
    feature_labels = oneHotEncoder.categories_
    feature_labels = np.concatenate(feature_labels)
    encoded_df = pd.DataFrame(feature_arr, columns=feature_labels)
    testdata = pd.concat([testdata, encoded_df], axis=1)
    testdata.drop(category_features, axis=1, inplace=True)

    testdata['AGE'] = standardScaler.transform(testdata[['AGE']])

    output = smote_model.predict(testdata)
    output_df = pd.DataFrame(output, columns=['CLASS'])
    combined_df = pd.concat([testdata0, output_df], axis=1)
    combined_df.to_csv('predicted_data.csv', index=False)


    last_document = db[filename_without_extension].find_one(sort=[("_id", pymongo.DESCENDING)])
    last_id = last_document["_id"] if last_document else 0
    
    data_dict = combined_df.to_dict(orient='records')
    for i, doc in enumerate(data_dict, start=1):
        doc["_id"] = last_id + i
    collection = db[filename_without_extension]
    collection.insert_many(data_dict)
    
    prediction_str = str(output)

    return render_template("multiple_prediction.html", prediction_text="The prediction result is: \n {}".format(prediction_str))

if __name__ == "__main__":
    app.run(debug=True, port=5001)
