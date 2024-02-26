import json
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template, send_from_directory
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
smote_model=pickle.load(open('smote_rf.pkl', 'rb'))
oneHotEncoder = pickle.load(open('oneHotEncoder.pkl', 'rb'))
standardScaler=pickle.load(open('StandardScaler.pkl', 'rb'))


# @app.route('/')：这个装饰器定义了根路径的处理函数 home()。当用户访问根路径时，将渲染名为 home.html 的 HTML 模板，并将其发送给客户端。
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/styles.css')
def styles():
    return send_from_directory('static', 'styles.css')

# @app.route('/predict_api', methods=['POST'])：这个装饰器定义了 /predict_api 路由的处理函数 predict_api()。
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

# @app.route('/predict', methods=['POST'])：这个装饰器定义了 /predict 路由的处理函数 predict(), 最后，处理函数将预测结果渲染到一个 HTML 模板中，并将该模板发送给客户端。
@app.route('/predict', methods=['POST'])
def predict():
    form_values = request.form.values()
    testdata = pd.DataFrame([list(form_values)], columns=request.form.keys())
    # print("************************")
    # print(testdata)
    # print("************************")
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
    # render_template 函数被用于渲染名为 home.html 的模板文件，并将 prediction_text 变量传递给模板。
    return render_template("home.html", prediction_text="The donor prediction is {}".format(output_serializable))

# if __name__ == "__main__":：这个条件语句检查脚本是否直接运行，而不是作为模块导入。如果脚本直接运行，那么调用 app.run() 启动 Flask 应用程序，并在调试模式下运行（debug=True）。
# 在 Flask 应用中，通常不需要显式地编写 main 函数。当您运行 Flask 应用时，应用会自动检测并启动。
#     在这种情况下，如果 __name__ 变量的值是 "__main__"，那么 Flask 应用会自动开始运行。
if __name__ == "__main__":
    app.run(debug=True, port=5001)