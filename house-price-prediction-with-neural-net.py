#モジュールの読み込み
from __future__ import print_function

import pandas as pd
from pandas import Series,DataFrame

#from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from keras.optimizers import Adam

from sklearn.metrics import mean_squared_error

path = "../input/kc_house_data.csv"
#CSVファイルの読み込み
data_set = pd.read_csv(path)

#説明変数(except price and date)
x = DataFrame(data_set.drop(["price","date"],axis=1))


#目的変数(price)
y = DataFrame(data_set["price"])

# normalize x and y with sklearn.preprocessing
x = preprocessing.scale(x)
y = preprocessing.scale(y)

#説明変数・目的変数をそれぞれ訓練データ・テストデータに分割
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)

#データの整形
x_train = x_train.astype(np.float)
x_test = x_test.astype(np.float)

#ニューラルネットワークの実装①
model = Sequential()

model.add(Dense(50, activation='relu', input_shape=(19,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(50, activation='relu', input_shape=(50,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(50, activation='relu', input_shape=(50,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(1))

model.summary()
print("\n")

#ニューラルネットワークの実装②
model.compile(loss='mean_squared_error',optimizer=Adam(),metrics=['mse'])

#ニューラルネットワークの学習
history = model.fit(x_train, y_train,batch_size=200, epochs=1000, verbose=1, validation_data=(x_test, y_test))

#RMSE用にMSEを算出
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("KERAS REG MSE : %.2f" % (mse))

# show its root mean square error
mse = mean_squared_error(y_test, y_pred)
print("KERAS REG RMSE : %.2f" % (mse ** 0.5))

#ニューラルネットワークの推論
score = model.evaluate(x_test,y_test,verbose=1)
print("\n")
print("Test loss:",score[0])
#print("Test accuracy:",score[1])

def plot_history(history):
    # 損失の履歴をプロット
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()

# 学習履歴をプロット
plot_history(history)