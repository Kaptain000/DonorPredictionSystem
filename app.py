import json
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template, redirect, send_from_directory
import numpy as np
import pandas as pd


# 使用 Flask 框架创建一个 Flask 应用程序对象
# __name__ 是一个特殊变量，表示当前 Python 模块的名称
app = Flask(__name__)

# Load the model
smote_model=pickle.load(open('smote_rf.pkl', 'rb'))
oneHotEncoder = pickle.load(open('oneHotEncoder.pkl', 'rb'))
standardScaler=pickle.load(open('StandardScaler.pkl', 'rb'))


# @app.route('/')：这个装饰器定义了当用户访问应用程序的根路径 '/' 时，Flask 将调用 home() 函数，并渲染名为 'home.html' 的模板文件
@app.route('/',  methods=['GET', 'POST'])
def home():
    # Flask 应用程序默认会在 templates 文件夹中查找模板文件，因此您不需要在调用 render_template 函数时提供完整的路径
    # render_template('home.html') 是 Flask 中的一个函数，用于渲染模板文件 'home.html'，并将其作为 HTTP 响应返回给客户端
    return render_template('home.html')

@app.route('/csv_predict')
def csv_predict():
    return render_template('csv_predict.html')

@app.route('/form_predict')
def form_predict():
    return render_template('form_predict.html')


@app.route('/styles.css')
def styles():
    # send_from_directory('static', 'styles.css') 是 Flask 中的一个函数，用于从指定的目录中发送文件。在这里，它从名为 'static' 的目录中发送名为 'styles.css' 的文件。
    return send_from_directory('static', 'styles.css')

# @app.route('/predict_api', methods=['POST'])：这个装饰器定义了 /predict_api 路由的处理函数 predict_api()。
@app.route('/predict_api', methods=['POST'])
def predict_api():
    # 从请求中获取 JSON 格式的数据，并将其中名为 'data' 的字段值赋给名为 data 的变量。
    data=request.json['data']
    category_features = ['CITY', 'EDUCATION_LEVEL', 'GENDER', 'MARITAL_STATUS', 'OCCUPATION']
    # transform json data into dataframe
    testdata = pd.DataFrame([data])
    feature_arr = oneHotEncoder.transform(testdata[category_features]).toarray()
    # oneHotEncoder.categories_得到的feature_labels是这样的：
    #     Feature 1: ['Blue' 'Green' 'Red']
    #     Feature 2: ['Large' 'Medium' 'Small']
    # np.concatenate(feature_labels)后为：
    #     ['Blue' 'Green' 'Red' 'Large' 'Medium' 'Small']
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
@app.route('/form_predict_result', methods=['POST'])
def form_predict_result():
    # 通过 request.form.values() 可以获取这些表单数据的值：
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
    # render_template 函数被用于渲染名为 home.html 的模板文件，并将 prediction_text 变量传递给模板。
    return render_template("form_predict.html", prediction_text="This person is a potential donor") if output_serializable == 1 else render_template("form_predict.html", prediction_text="This person is not a potential donor")

@app.route('/csv_predict_result', methods=['POST'])
def csv_predict_result():
    # 通过 request.form.values() 可以获取这些表单数据的值：
    if 'CSV File' not in request.files:
            return 'No file part'
    csv_file = request.files['CSV File']
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
    prediction_str = str(output)

    # render_template 函数被用于渲染名为 home.html 的模板文件，并将 prediction_text 变量传递给模板。
    return render_template("csv_predict.html", prediction_text="The prediction result is {}".format(prediction_str))

# if __name__ == "__main__":：这个条件语句检查脚本是否直接运行，而不是作为模块导入。如果脚本直接运行，那么调用 app.run() 启动 Flask 应用程序，并在调试模式下运行（debug=True）。
# 在 Flask 应用程序中，通常不需要显式定义一个 main() 函数，因为 Flask 应用程序对象（即 app）会被创建并在需要的时候运行。
if __name__ == "__main__":
    app.run(debug=True, port=5001)