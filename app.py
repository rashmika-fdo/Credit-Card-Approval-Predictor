import streamlit as st
import pandas as pd
from joblib import load
import numpy as np


st.title('Credit Card Approval Predictor')
st.write('Please enter the below details')

Annual_income = st.text_input('Enter the annual income')
gender_options = ['Male', 'Female']
GENDER = st.selectbox('Select the Gender', gender_options)

car_opt = ['Yes', 'No']
Car_Owner = st.selectbox('Car Owner?', car_opt)

property_opt = ['Yes', 'No']
Propert_Owner = st.selectbox('Property Owner?', property_opt)

marital_opt = ['Married', 'Single / not married', 'Civil marriage', 'Separated','Widow']
Marital_status = st.selectbox('Marital Status', marital_opt)

edu_opt = ['Higher education', 'Secondary / secondary special','Lower secondary', 'Incomplete higher', 'Academic degree']
EDUCATION = st.selectbox('Education Level', edu_opt)

from sklearn import preprocessing
labelencoder = preprocessing.LabelEncoder()

gen = labelencoder.fit_transform([GENDER])
car = labelencoder.fit_transform([Car_Owner])
prop = labelencoder.fit_transform([Propert_Owner])
inc = labelencoder.fit_transform([Annual_income])
mar = labelencoder.fit_transform([Marital_status])
edu = labelencoder.fit_transform([EDUCATION])

input = [gen,car,prop,inc,mar,edu]

# Convert to 2D array for prediction
input_scaled = np.array(input).reshape(1, -1)



model = load('ada_model.joblib')

pred = model.predict(input_scaled)

if pred==0:
    st.success('Credit Approved')
else:
    st.error('Credit Rejected')




