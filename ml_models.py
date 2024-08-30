import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, LSTM

def predict_stock_price_lstm(data, days=30):
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

   
    training_data_len = int(np.ceil( len(scaled_data) * .95 ))
    train_data = scaled_data[0:training_data_len, :]
    x_train, y_train = [], []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

   
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

   
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

   
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    
    last_60_days = scaled_data[-60:]
    X_future = []
    for i in range(days):
        X_future.append(last_60_days[-60:])
        prediction = model.predict(np.array(X_future).reshape(1, 60, 1))
        last_60_days = np.append(last_60_days, prediction)
        X_future = []

    future_predictions = scaler.inverse_transform(last_60_days[-days:].reshape(-1, 1))

    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days)
    predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_predictions.flatten()})

    return predictions_df