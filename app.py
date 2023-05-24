#Impor Library
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("Prediksi Cuaca Kemayoran")

st.subheader("Menampilkan Data Iklim Kemayoran 2018-2023")
df = pd.read_csv("https://raw.githubusercontent.com/khoulaaf/kemayoran/main/iklim_kemayoran_18_23.csv")
st.dataframe(df)

df=df.dropna(axis=0)

df['Tanggal'] = pd.to_datetime(df['Tanggal'])
df.set_index('Tanggal', inplace= True)

st.subheader("Suhu Rata-Rata")
fig = plt.figure(figsize = (12, 6))
plt.plot(df.Tavg, 'b')
plt.legend()
st.pyplot(fig)

n_cols = 1
dataset = df['Tavg']
dataset = pd.DataFrame(dataset)
data = dataset.values

st.subheader("Prediksi Suhu Rata-Rata Kemayoran")

#Scaling data menggunakan min max scaler (0,1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range= (0, 1))
scaled_data = scaler.fit_transform(np.array(data))

#Split data menjadi data train dan data test
train_size = int(len(data) * 0.75)
test_size = len(data) - train_size
print("Train Size :",train_size,"Test Size :",test_size)

train_data = scaled_data[0:train_size, :]

x_train = []
y_train = []
time_steps = 50
n_cols = 1

for i in range(time_steps, len(scaled_data)):
    x_train.append(scaled_data[i-time_steps:i, :n_cols])
    y_train.append(scaled_data[i, :n_cols])
    if i<=time_steps:
        print('X_train: ', x_train)
        print('y_train:' , y_train)

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], n_cols))

#Implementasi Model LSTM 
#dengan 3 layer LSTM dengan layer pertama sebesar 200 neuron, 
#layer kedua sebesar 10 neuron dan layer ketiga sebesar 50 neuron 
#serta layer dense dengan layer pertama sebesar 20 neuron, 
#layer kedua sebesar 10 neuron, dan layer ketiga sebesar 1 neuron. 
#Optimizer menggunakan optimizer adam, untuk loss terdapat mean_absolute_error.
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
model = Sequential()
model.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(50, 1)))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(LSTM(50, activation='relu', return_sequences=False))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics= "mean_absolute_error")
print(model.summary())

history = model.fit(x_train, y_train, epochs= 100, batch_size= 32)

# Membuat data test dengan 50 time-steps and 1 output
time_steps = 50
test_data = scaled_data[train_size - time_steps:, :]

x_test = []
y_test = []
n_cols = 1

for i in range(time_steps, len(test_data)):
    x_test.append(test_data[i-time_steps:i, 0:n_cols])
    y_test.append(test_data[i, 0:n_cols])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], n_cols))

predictions = model.predict(x_test)

predictions = scaler.inverse_transform(predictions)

y_test = scaler.inverse_transform(y_test)

RMSE = np.sqrt(np.mean( y_test - predictions )**2).round(2)

train = dataset.iloc[:train_size , 0:1]
test = dataset.iloc[train_size: , 0:1]
test['Predictions'] = predictions

#Visualisasi Prediksi
fig2 = plt.figure(figsize= (12, 6))
plt.xlabel('Tahun', fontsize=12)
plt.ylabel('Suhu', fontsize=12)
plt.plot(train['Tavg'])
plt.plot(test['Tavg'])
plt.plot(test['Predictions'])
plt.legend(['Train', 'Test', 'Predictions'])
st.pyplot(fig2)

#Prediksi vs Aktual
fig3 = plt.figure(figsize = (12, 6))
plt.plot(preds_acts['Predictions'])
plt.plot(preds_acts['Actuals'])
plt.legend(['Predictions', 'Actuals'])
st.pyplot(fig3)

preds_acts = pd.DataFrame(data={'Prediksi':predictions.flatten()})
preds_acts
