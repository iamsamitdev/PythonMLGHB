# import libraries
import streamlit as st
from joblib import load

# load the model from disk
model = load('titanic_model.joblib')

# Create Streamlit Web App
st.title('Titanic Survival Prediction')

# Sidebar with menu
st.sidebar.title('Menu')

# Menu options
menu = ['Home', 'Predition']

# Sidebar selection
st.sidebar.selectbox('', menu)

# Input with sliders
age = st.slider('Age', 0.42, 80.0, 30.0)
sibsp = st.slider('SibSp', 0, 8, 0)
parch = st.slider('Parch', 0, 9, 0)
fare = st.slider('Fare', 0.0, 512.30, 32.20)

# Add prediction button
predict_button = st.button('Predict')

# Prediction logic
if predict_button:
    # รับค่าจาก input มาเก็บไว้ในตัวแปรแบบ list
    input_data = [[age, sibsp, parch, fare]]

    # ทำนายผล
    prediction = model.predict(input_data)

    # หาค่าความน่าจะเป็น
    predict_proba = model.predict_proba(input_data)

    # แสดงผลลัพธ์ที่ได้
    st.subheader('Prediction')
    if prediction[0] == 1:
        st.write('Survived')
    else:
        st.write('Not Survived')

    # แสดงค่าความน่าจะเป็น
    st.subheader('Prediction Probability:')
    st.write(f'Survived: {predict_proba[0][1]:.2f}')
    st.write(f'Did not survive: {predict_proba[0][0]:.2f}')