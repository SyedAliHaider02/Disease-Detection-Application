from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open('heart1.pkl', 'rb'))
modell = pickle.load(open('kidney1.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def homepage():
    return render_template('homepage.html')


@app.route('/home')
def man():
    return render_template('front.html')


@app.route('/self')
def mann():
    return render_template('kidney_front.html')


@app.route('/predict', methods=['GET', 'POST'])
def home():
    data1 = request.form['age']
    data2 = request.form['gender']
    data3 = request.form['cp']
    data4 = request.form['tresbps']
    data5 = request.form['chol']
    data6 = request.form['fbs']
    data7 = request.form['restecg']
    data8 = request.form['thalach']
    data9 = request.form['exang']
    data10 = request.form['oldpeak']
    data11 = request.form['slope']
    data12 = request.form['ca']
    data13 = request.form['thal']
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


@app.route('/pred', methods=['GET', 'POST'])
def self():
    dat1 = request.form['id']
    dat2 = request.form['age']
    dat3 = request.form['bp']
    dat4 = request.form['sg']
    dat5 = request.form['al']
    dat6 = request.form['su']
    dat7 = request.form['rbc']
    dat8 = request.form['pc']
    dat9 = request.form['pcc']
    dat10 = request.form['ba']
    dat11 = request.form['bgr']
    dat12 = request.form['bu']
    dat13 = request.form['sc']
    dat14 = request.form['sd']
    dat15 = request.form['pot']
    dat16 = request.form['hemo']
    dat17 = request.form['pcv']
    dat18 = request.form['wc']
    dat19 = request.form['rc']
    dat20 = request.form['htn']
    dat21 = request.form['dm']
    dat22 = request.form['cad']
    dat23 = request.form['appet']
    dat24 = request.form['pe']
    dat25 = request.form['ane']
    arr1 = np.array([[dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8, dat9, dat10, dat11, dat12, dat13, dat14,dat15, dat16, dat17, dat18, dat19, dat20, dat21, dat22, dat23, dat24, dat25]])
    predict = modell.predict(arr1)
    return render_template('home.html', dataa=predict)


if __name__ == "__main__":
    app.run(debug=True)
