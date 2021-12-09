import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.models import load_model

start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Price Predictor')

user_input = st.text_input('Enter Stock Ticker', 'TSLA')
df = data.DataReader(user_input, 'yahoo', start, end)

# Display Data
st.subheader('Data from 2010 - 2019')
st.write(df.describe())

# Plot Graphs
st.subheader('Closing Stock Price vs Time')
fig1 = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig1)

st.subheader('50 Days Moving Average & 200 Days Moving Average')
movingAverage50 = df.Close.rolling(50).mean()
movingAverage200 = df.Close.rolling(200).mean()
fig2 = plt.figure(figsize = (12,6))
plt.plot(df.Close, label="Stock Price")
plt.plot(movingAverage50, 'r', label="50 Days Moving Average")
plt.plot(movingAverage200, 'g', label="50 Days Moving Average")
plt.legend()
st.pyplot(fig2)

# Splitting the data into training and testing sets

training_data = pd.DataFrame(df['Close'][:int(len(df)*0.70)])
testing_data = pd.DataFrame(df['Close'][int(len(df)*0.70):])

scaler = MinMaxScaler(feature_range=(0,1))
training_data_array = scaler.fit_transform(training_data)

# Load the model
model = load_model('keras_model.h5')

# Making Predictions
past_100_days = training_data.tail(100)
final_df = past_100_days.append(testing_data, ignore_index=True)
input_data = scaler.fit_transform(final_df)

X_testing = []
y_testing = []

for i in range(100, input_data.shape[0]):
    X_testing.append(input_data[i-100:i])
    y_testing.append(input_data[i, 0])
    
X_testing, y_testing = np.array(X_testing), np.array(y_testing)
y_predicted = model.predict(X_testing)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_testing = y_testing * scale_factor

# Plot the predictions
st.subheader('Predicted Stock Price vs Original Stock Price')
fig3 = plt.figure(figsize=(12,6))
plt.plot(y_testing, 'b', label='Original Stock Price')
plt.plot(y_predicted, 'r', label='Predicted Stock Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)