import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the machine learning model and encode
model = joblib.load('XGB_churn.pkl')
gender_encode= joblib.load('gender_encode.pkl')
oneHot_encode_geo=joblib.load('geo_encode.pkl')
HasCrCard_encode = joblib.load('cr_encode.pkl')
IsActiveMember_encode = joblib.load('active_encode.pkl')

def main():
    st.title("Customer Churn")

    creditscore = st.number_input("Credit Score", 0, 1000)
    gender = st.radio("Gender", ["Male", "Female"])
    age = st.number_input("Age", 0, 100)
    tenure = st.number_input("Length of time customer has worked with bank (in years)", 0, 20)
    balance = st.number_input("Current Balance", 0, 300000)
    numofprod = st.number_input("Number of products customer has", 0, 4)
    hascrcard = st.radio("Customer has credit card", ["Yes", "No"])
    isactive = st.radio("Customer is an active member", ["Yes", "No"])
    salary = st.number_input("Estimated salary of customer", 0, 300000)
    geo = st.radio("Where customer lives", ["France", "Germany", "Spain"])

    data = {'CreditScore': int(creditscore), 'Gender': gender, 'Age': int(age), 'Tenure': int(tenure), 'Balance': int(balance), 'NumOfProducts': int(numofprod),
            'HasCrCard': hascrcard, 'IsActiveMember': isactive, 'EstimatedSalary': int(salary), 'Geography': geo}
    
    df=pd.DataFrame([list(data.values())], columns=['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
       'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography'])
    
    df=df.replace(gender_encode)
    df=df.replace(HasCrCard_encode)
    df=df.replace(IsActiveMember_encode)
    cat_geo=df[['Geography']]
    cat_enc_geo=pd.DataFrame(oneHot_encode_geo.transform(cat_geo).toarray(),columns=oneHot_encode_geo.get_feature_names_out())
    df=pd.concat([df,cat_enc_geo], axis=1)
    df=df.drop(['Geography'],axis=1)

    if st.button('Make Prediction'):
        features=df      
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
