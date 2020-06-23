import numpy as np
import pandas as pd
from flask import Flask, request
from flask import jsonify, render_template, make_response
import pickle

app =Flask(__name__)
# load the saved model
model_linreg = pickle.load(open('model_linReg.pkl', 'rb'))

@app.route('/index')
def index():
    return 'Hello regression is started'

def pred():
    pass

if __name__ == '__main__':
    app.run(debug=True, threaded=False)