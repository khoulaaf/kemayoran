#Impor Library
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("Prediksi Cuaca Kemayoran")

st.subheader("Menampilkan Data Iklim Kemayoran 2018-2022")
df = pd.read_csv("https://raw.githubusercontent.com/khoulaaf/kemayoran/main/iklim_kemayoran_18_23.csv")
st.dataframe(df)

df['Tanggal'] = pd.to_datetime(df['Tanggal'])
df.set_index('Tanggal', inplace= True)

st.subheader("Suhu Rata-Rata")
fig = plt.figure(figsize = (12, 6))
plt.plot(df.Tavg, 'b')
plt.legend()
st.pyplot(fig)

st.subheader("Prediksi Suhu Rata-Rata Kemayoran")

#Membagi data menjadi data train dan data test
data_training= pd.DataFrame(df['Tavg'][0:int(len(df)*0.75)])
data_testing = pd.DataFrame(df['Tavg'][int(len(df)*0.25): int(len(df))])

print("training data: ",data_training.shape)
print("testing data: ", data_testing.shape)

#Scaling data menggunakan min max scaler (0,1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

#Implementasi model lstm
model = load_model("keras_model.h5")
