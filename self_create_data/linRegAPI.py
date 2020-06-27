import numpy as np
import pandas as pd
from flask import Flask, request
from flask import jsonify, render_template, make_response
import pickle

app =Flask(__name__)
# load the saved model
model_linreg = pickle.load(open('model_linReg.pkl', 'rb'))
# load the one_hot encodinga and label encoding
label_en_loaded = pickle.load(open("le.obj",'rb'))
with open("one_hot.pkl",'rb') as f:
    oneHot_en_loaded = pickle.load(f)

@app.route('/index', methods=['GET'])
def index():
    return 'Hello Linear Regression is started'

# Note: we will optimize the code on later classes: just run the lengthy process for now and list append below
@app.route('/pred', methods=['POST', 'GET'])
def pred():
    data =[]
    req = request.get_json()
    data.append(req['sales1'])
    data.append(req['sales2'])
    data.append(req['sales3'])
    data.append(req['sales4'])
    # encoding the demand level passing from req
    data.append(label_en_loaded.transform([req['deman_level']]))
    data.extend(oneHot_en_loaded.transform([[req['best_seller']]]).toarray()[0])
    pred_result = model_linreg.predict([data])
    output = round(pred_result[0],2)
    print(f'The predicted value is {output}')
    return jsonify({'The Predicted Value:': output})

if __name__ == '__main__':
    app.run(debug=True, threaded=False)