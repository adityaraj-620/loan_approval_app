import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

df = pd.read_csv('loan_data_large.csv')

# Encode categorical data
le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Loan_Status']:
    df[col] = le.fit_transform(df[col])

X = df[['Gender', 'Married', 'Education', 'ApplicantIncome', 'LoanAmount', 'Credit_History']]
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

os.makedirs('model', exist_ok=True)
with open('model/loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)
