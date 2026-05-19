import streamlit as st
import numpy as np
import joblib

model = joblib.load('churn_stacked_lr_model.pkl')
scaler = joblib.load('churn_scale.pkl')

st.set_page_config(page_title='Customer Churn Predictor', layout='centered')
st.title('📞 Telco Customer Churn Prediction')
st.markdown('Predict whether a customer is likely to churn or not.')

stenure = st.number_input('Tenure (Months)', min_value=0, max_value=100, value=12)
MonthlyCharges = st.number_input('Monthly Charges', min_value=0.0, value=70.0)
TotalCharges = st.number_input('Total Charges', min_value=0.0, value=1000.0)

Partner = st.selectbox('Partner', ['Yes', 'No'])
Dependents = st.selectbox('Dependents', ['Yes', 'No'])
PaperlessBilling = st.selectbox('Paperless Billing', ['Yes', 'No'])
MultipleLines = st.selectbox('Multiple Lines', ['Yes', 'No'])

InternetService = st.selectbox('Internet Service', ['Fiber optic', 'DSL', 'No'])
OnlineSecurity = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
OnlineBackup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
DeviceProtection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
TechSupport = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
StreamingTV = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
StreamingMovies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])

Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
PaymentMethod = st.selectbox(
    'Payment Method',
    [
        'Electronic check',
        'Credit card (automatic)',
        'Mailed check',
        'Bank transfer (automatic)'
    ]
)

InternetService_Fiber_optic = 1 if InternetService == 'Fiber optic' else 0
InternetService_No = 1 if InternetService == 'No' else 0

Contract_Two_year = 1 if Contract == 'Two year' else 0
Contract_One_year = 1 if Contract == 'One year' else 0

PaymentMethod_Electronic_check = 1 if PaymentMethod == 'Electronic check' else 0
PaymentMethod_Credit_card = 1 if PaymentMethod == 'Credit card (automatic)' else 0
PaymentMethod_Mailed_check = 1 if PaymentMethod == 'Mailed check' else 0

DeviceProtection_No_internet_service = 1 if DeviceProtection == 'No internet service' else 0
TechSupport_No_internet_service = 1 if TechSupport == 'No internet service' else 0
StreamingMovies_No_internet_service = 1 if StreamingMovies == 'No internet service' else 0
OnlineBackup_No_internet_service = 1 if OnlineBackup == 'No internet service' else 0
OnlineSecurity_No_internet_service = 1 if OnlineSecurity == 'No internet service' else 0
StreamingTV_No_internet_service = 1 if StreamingTV == 'No internet service' else 0

PaperlessBilling_Yes = 1 if PaperlessBilling == 'Yes' else 0

OnlineSecurity_Yes = 1 if OnlineSecurity == 'Yes' else 0
TechSupport_Yes = 1 if TechSupport == 'Yes' else 0
Dependents_Yes = 1 if Dependents == 'Yes' else 0
Partner_Yes = 1 if Partner == 'Yes' else 0
OnlineBackup_Yes = 1 if OnlineBackup == 'Yes' else 0
DeviceProtection_Yes = 1 if DeviceProtection == 'Yes' else 0
StreamingTV_Yes = 1 if StreamingTV == 'Yes' else 0
StreamingMovies_Yes = 1 if StreamingMovies == 'Yes' else 0
MultipleLines_Yes = 1 if MultipleLines == 'Yes' else 0

full_features = np.array([[ 
    stenure,
    TotalCharges,
    MonthlyCharges,
    InternetService_Fiber_optic,
    Contract_Two_year,
    PaymentMethod_Electronic_check,
    DeviceProtection_No_internet_service,
    TechSupport_No_internet_service,
    StreamingMovies_No_internet_service,
    OnlineBackup_No_internet_service,
    OnlineSecurity_No_internet_service,
    InternetService_No,
    StreamingTV_No_internet_service,
    PaperlessBilling_Yes,
    Contract_One_year,
    OnlineSecurity_Yes,
    TechSupport_Yes,
    Dependents_Yes,
    Partner_Yes,
    PaymentMethod_Credit_card,
    PaymentMethod_Mailed_check,
    OnlineBackup_Yes,
    DeviceProtection_Yes,
    StreamingTV_Yes,
    StreamingMovies_Yes,
    MultipleLines_Yes
]])

final_features = scaler.transform(full_features)

if st.button('Predict Churn'):

    prediction = model.predict(final_features)[0]
    probability = model.predict_proba(final_features)[0][1]

    if prediction == 1:
        st.error(f'Customer is likely to churn ⚠️\n\nProbability: {probability:.2f}')
    else:
        st.success(f'Customer is not likely to churn ✅\n\nProbability: {probability:.2f}')