import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

#load the model
model = pickle.load(open('D:/deploy ML kualitas udara/forecast_model.sav','rb'))

#load dataset to plot alongside predictions
df = pd.read_excel("D:/deploy ML Suhu/CO2 dataset.xlsx")
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index(['Year'], inplace=True)

st.title('Forecasting Suhu Rata-Rata')
year = st.slider("Tentukan Tahun",1,30,step=1)

pred = model.forecast(year)
pred = pd.DataFrame(pred, columns=['CO2'])

if st.button("Predict"):
    col1, col2 = st.columns([2,3])
    with col1:
        st.dataframe(pred)
    with col2:
        fig, ax = plt.subplots()
        df['CO2'].plot(style='--', color='gray', legend=True, label='known')
        pred['CO2'].plot(color='b', legend=True, label='prediction')
        st.pyplot(fig)