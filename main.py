import datetime
import numpy as np
import pandas_datareader as web
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


def main():
    # loading data
    today = datetime.datetime.today()
    load_time = str(today.year) + '-' + str(today.month) + '-' + str(today.day - 2)
    print(load_time)
    df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end=load_time)

    # visualize data
    plt.figure(figsize=(16, 8))
    plt.title('Close prise history')
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Prise USD ($)', fontsize=18)
    plt.show()

    # fit scaler
    data = df.filter(['Close'])
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # creating train and test data
    train_data = scaled_data[0:2000, :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # building model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # training model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # creating test data
    test_data = scaled_data[2000 - 60:, :]
    x_test = []
    y_test = dataset[2000:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # calculate root mean sqr error
    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
    print(rmse)

    # save model
    model.save('stock_price_predictor_' + str(rmse) + '.h5')

    # visualize train result
    train = data[:2000]
    valid = data[2000:]
    valid.loc[:, 'Predictions'] = predictions

    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Prise USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()

    # getting prediction for tomorrow
    load_time = str(today.year) + '-' + str(today.month) + '-' + str(today.day - 1)
    df = web.DataReader('AAPL', data_source='yahoo', start='2013-01-06', end=load_time)
    new_df = df.filter(['Close'])
    last_60_days = new_df[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    print(pred_price)


if __name__ == '__main__':
    main()
