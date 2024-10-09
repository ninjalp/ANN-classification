import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

 
model = tf.keras.models.load_model('model_kredi_riks_yeni_model.h5')

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)   

 
st.title('Risk Score Tahmini')

 
credit_score = st.number_input('Kredi Puanı', min_value=300, max_value=850, value=600)
debt_to_income_ratio = st.number_input('Borç/Gelir Oranı', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
monthly_debt_payments = st.number_input('Aylık Borç Ödemeleri', min_value=0, value=400)
monthly_income = st.number_input('Aylık Gelir', min_value=100, value=2700)
employment_status = st.selectbox('İstihdam Durumu', ['Employed', 'Self-Employed', 'Unemployed'])
loan_amount = st.number_input('Kredi Miktarı', min_value=1000, value=15000)
loan_duration = st.number_input('Kredi Süresi (Ay)', min_value=1, value=48)
previous_loan_defaults = st.selectbox('Önceki Kredi Temerrüt Durumu', ['Yes', 'No'])
payment_history = st.number_input('Ödeme Geçmişi', min_value=0, value=26)
 
employment_status_map = {'Employed': 0, 'Self-Employed': 1, 'Unemployed': 2}
previous_loan_defaults = 1 if previous_loan_defaults == 'Yes' else 0

 
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'DebtToIncomeRatio': [debt_to_income_ratio],
    'MonthlyDebtPayments': [monthly_debt_payments],
    'MonthlyIncome': [monthly_income],
    'EmploymentStatus': [employment_status_map[employment_status]],
    'LoanAmount': [loan_amount],
    'LoanDuration': [loan_duration],
    'PreviousLoanDefaults': [previous_loan_defaults],
    'PaymentHistory': [payment_history]
})

 
input_data_scaled = scaler.transform(input_data)

 
prediction = model.predict(input_data_scaled)
risk_score = prediction[0][0]

 
st.write(f'Tahmini Risk Skoru: {risk_score:.2f}')

if risk_score > 70:
    st.write('Yüksek Risk')
else:
    st.write('Düşük Risk')
