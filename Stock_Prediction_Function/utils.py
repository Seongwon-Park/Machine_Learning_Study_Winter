# Import Library
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import EarlyStopping

''' 94번째 줄의 모델이 저장될 공간의 PATH 를 변경해야함 '''
''' 나중에 가장 성능이 좋은 모델을 이용하여 예측하기 위해 저장 '''
''' 모델의 이름은 RMSE 와 그 값으로 이루어져 있음 '''

# ** Future works ** #
# 1. 그래서 내일 가격이 얼마인데?
# -> mid 라는 변수 속에서 전날에 대한 비율만 알 수 있으므로, 우리는 실제 가격을 알 수 없는게 아닐까?
# 즉, Target attribute 를 바꿔서 진행해야 하지 않을까? (ex. 종가)

# 2. RMSE 가 낮다고 무조건 잘 예측된 값일까?

# 3. Layer 의 층의 개수는 몇개가 적당할까?
# -> 위 경우, for 문을 이용하기는 어려우며,일일히 시도해봐야 할 것 같음

# 4. Drop Out 의 비율을 어떻게 하는 것이 적당할까?
# -> for 문에 넣고 돌려버릴까?


# Load the Dataset
def load_dataset():
    df = pd.read_csv('005930.KS.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['Year'] = df['Date'].dt.year
    return df


# Preprocessing the Dataset
def preprocessing_dataset(df):
    df.dropna(axis=0, how="any", inplace=True)
    high_prices = df['High'].values
    low_prices = df['Low'].values
    mid_prices = (high_prices + low_prices) / 2
    return mid_prices


# Normalize the Dataset
def normalize_dataset(mid_prices, window_size):
    seq_len = window_size
    sequence_length = seq_len + 1
    result = []
    for index in range(len(mid_prices) - sequence_length):
        result.append(mid_prices[index: index + sequence_length])
    normalized_data = []
    for window in result:
        normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalized_data.append(normalized_window)
    result = np.array(normalized_data)
    return result


# Split Train and Test Dataset
def split_dataset(result):
    row = int(round(result.shape[0] * 0.9))
    train = result[:row, :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = train[:, -1]
    x_test = result[row:, :-1]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_test = result[row:, -1]
    return x_train, x_test, y_train, y_test


# Build a Sequential model
def build_model(model_number, window_size, activation_type, optimizer_type, batch, x_train, x_test, y_train, y_test):

    # 모델의 call back 함수 설정 (early stopping)
    early_stop = EarlyStopping(monitor='val_loss', patience=5)

    # 모델 생성 및 훈련
    model = Sequential()
    model.add(LSTM(units=window_size, activation=activation_type, return_sequences=True, input_shape=(window_size, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=window_size+14, activation=activation_type, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer=optimizer_type)
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              batch_size=batch, epochs=50, callbacks=[early_stop])
    pred = model.predict(x_test)
    rmse = np.sqrt(np.mean(pred-y_test)**2)

    # 모델 저장
    model.save('/Users/parkseongwon/PycharmProjects/ML_Study/model/No_{0}_rmse_{1}.h5'
               .format(model_number, round(rmse, 4)))
    return pred, rmse


# Plot and Save Graph
def plot_graph(model_number, rmse, pred, y_test, window_size, activation_type, optimizer_type, batch):
    fig = plt.figure(facecolor='white', figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.plot(y_test, label='True')
    ax.plot(pred, label='Prediction')
    ax.legend()
    plt.title('No.{0} RMSE = {1}'.format(model_number, rmse))
    plt.xlabel('time')
    plt.ylabel('price')
    try:
        plt.savefig('image/NO.{0}_{1}_{2}_{3}_{4}.png'
                    .format(model_number, window_size, activation_type, optimizer_type, batch))
    except FileNotFoundError:
        print('image 폴더가 존재하지 않습니다. 폴더를 생성합니다.')
        os.mkdir('./image')
        plt.savefig('image/{0}_{1}_{2}_{3}.png'.format(window_size, activation_type, optimizer_type, batch))
    plt.clf()
