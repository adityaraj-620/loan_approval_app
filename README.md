# 🏦 Loan Approval Prediction App

This project is a Machine Learning web application that predicts whether a loan application will be approved or rejected based on applicant data such as income, credit history, loan amount, and other factors.

---

## 📌 Features

- Predict loan approval status using a trained ML classification model.
- User-friendly web interface built with Flask.
- Takes multiple input features relevant to loan applications.
- Provides fast and reliable predictions.
- Visualizes prediction results clearly to the user.

---

## 🛠️ Tech Stack

- Python
- Flask (Web Framework)
- Scikit-learn (Machine Learning)
- Pandas, NumPy (Data processing)
- HTML, CSS, Bootstrap (Frontend)
- Pickle (Model serialization)

---

## 📁 Project Structure
loan-approval-app/
├── app.py # Flask application
├── model/
│ └── loan_model.pkl # Trained ML model
├── requirements.txt # Python dependencies
├── static/
│ └── style.css # CSS styles
├── templates/
│ ├── index.html # Main form page
│ └── result.html # Prediction result page
├── loan_data.csv # Dataset used for training (optional)
├── loan_data_large.csv # Larger dataset (optional)
├── train_model.py # Script to train and save model
└── README.md # Project documentation
