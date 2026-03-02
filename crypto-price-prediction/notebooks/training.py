import pandas as pd
data=pd.read_csv("BTC-USD.csv")
data=data[['Close']]
data.dropna(inplace=True)

from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
scaled_data=sc.fit_transform(data)

training_size=int(len(scaled_data)*0.8)
train_data=scaled_data[:training_size]
test_data=scaled_data[training_size-60:]

import numpy as np
x_train=[]
y_train=[]

for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i])
    y_train.append(train_data[i])



x_train,y_train=np.array(x_train), np.array(y_train)


x_test=[]
y_test=[]

for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i])
    y_test.append(test_data[i])



x_test,y_test=np.array(x_test), np.array(y_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

model=Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train,y_train, epochs=20, batch_size=32)
model.save("lstm_model.h5")

predicted_price=model.predict(x_test)
predicted_price=sc.inverse_transform(predicted_price)
y_true=sc.inverse_transform(y_test.reshape(-1,1))

print("Next Day Predicted Price: ",predicted_price[0][0])

from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

mae=mean_absolute_error(y_true, predicted_price)
rmse=math.sqrt(mean_squared_error(y_true, predicted_price))

print("MAE: ",mae)
print("RMSE: ",rmse)