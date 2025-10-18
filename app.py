import streamlit as st
import pandas as pd
import numpy as np
import pickle


encoder=pickle.load(open('models/encoder.pkl','rb'))
scaling=pickle.load(open('models/scaling.pkl','rb'))   
model_gbc=pickle.load(open('models/model_gbc.pkl','rb'))

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
  input_df =pd.DataFrame(
      [[N, P, K, temperature, humidity, ph, rainfall,]],
                         columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
  
  input_scaled=scaling.transform(input_df.values)

  prediction_encoded=model_gbc.predict(input_scaled)
  prediction_encoded=np.ravel(prediction_encoded)
  prediction=encoder.inverse_transform(prediction_encoded)

  return prediction[0]

st.title('Crop Recommendation System')
st.markdown("This app predicts the most suitable crop to cultivate based on soil and weather conditions.")

col1, col2 = st.columns(2)

with col1:
    N =st.number_input('Nitrogen content in soil (N)', min_value=0, max_value=140, value=10, step=1)
    P=st.number_input('Phosphorus content in soil (P)', min_value=0, max_value=145, value=10, step=1)
    K=st.number_input('Potassium content in soil (K)', min_value=0, max_value=205, value=10, step=1)
    ph=st.number_input('pH value of the soil', min_value=3.5, max_value=9.5, value=6.5, step=0.1)

with col2:
    temperature=st.number_input('Temperature (°C)', min_value=8.0, max_value=43.0, value=20.0, step=0.1)
    humidity=st.number_input('Humidity (%)', min_value=14.0, max_value=100.0, value=50.0, step=0.1)
    rainfall=st.number_input('Rainfall (mm)', min_value=20.0, max_value=300.0, value=100.0, step=0.1)

if st.button('Predict Crop'):
    result=predict_crop(N, P, K, ph, temperature, humidity, rainfall)
    st.success(f'The most suitable crop to cultivate is: {result}') 