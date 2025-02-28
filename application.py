from flask import Flask,request,app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd

application=Flask(__name__)
app=application

scaler=pickle.load(open('C:/Users/Piyush Naula/OneDrive/Desktop/Python/Machine Learning/Diabetes Prediction/model/standardscaler.pkl','rb'))
model=pickle.load(open('C:/Users/Piyush Naula/OneDrive/Desktop/Python/Machine Learning/Diabetes Prediction/model/modelforprediction.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=''
    if request.method=='POST':
        Pregnancies=int(request.form.get('Pregnancies'))
        Glucose=int(request.form.get('Glucose'))
        BloodPressure=int(request.form.get('BloodPressure'))
        SkinThickness=int(request.form.get('SkinThickness'))
        Insulin=int(request.form.get('Insulin'))
        BMI=int(request.form.get('BMI'))
        DiabetesPedigreeFunction=int(request.form.get('DiabetesPedigreeFunction'))
        Age=int(request.form.get('Age'))

        new_data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict=model.predict(new_data)

        if predict[0]==1:
            result='Diabetic'
        else:
            result='Non-Diabetic'
        
        return render_template('single_prediction.html',result=result)
    else:
        return render_template('home.html')
    
if __name__=='__main__':
    app.run(host='0.0.0.0')