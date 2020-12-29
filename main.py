import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
import os

app = Flask('__name__')

def load_model():
    return pickle.load(open('loan_model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')
   
@app.route('/predict', methods=['POST'])
def predict():
    labels=['Approved','Rejected']
    features=[float(x) for x in request.form.values()]
    values=[np.array(features)]
    model=load_model()
    prediction=model.predict(values)
    result=labels[prediction[0]]
    return render_template('index.html', output='The loan request is {}'.format(result))

if __name__=="__main__":
   # port=int(os.environ.get('PORT',5000))
   app.run(debug=True)

