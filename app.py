#Impor Library
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

#Menampilkan Judul
st.title('Prediksi Cuaca Kemayoran')

#Membaca Dataset
df = pickle.load(open('iklim_kemayoran_18_22.sav', 'rb'))

st.subheader('Data from 2018-2022')
#df= df.reset_index()
st.write(df.tail(10))
st.write(df.describe())