from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model/loan_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = 1 if request.form['gender'] == 'Male' else 0
    married = 1 if request.form['married'] == 'Yes' else 0
    education = 1 if request.form['education'] == 'Graduate' else 0
    income = float(request.form['income'])
    loan = float(request.form['loan'])
    credit = int(request.form['credit'])

    features = np.array([[gender, married, education, income, loan, credit]])
    pred = model.predict(features)[0]
    result = "Approved ✅" if pred == 1 else "Not Approved ❌"

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
