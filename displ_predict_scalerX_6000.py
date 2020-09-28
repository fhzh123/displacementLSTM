
import os

os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout
import time #helper libraries
from scipy.io import loadmat
from keras.models import model_from_json

import keras.backend.tensorflow_backend as K
K.set_session
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# file is downloaded from finance.yahoo.com, 1.1.1997-1.1.2017
# training data = 1.1.1997 - 1.1.2007
# test data = 1.1.2007 - 1.1.2017
#input_file="DIS.csv"
#input_file_X="X.csv"
#input_file_Y="Y.csv"
input_file="ACCSTRAIN_DISPL_test.mat"

## convert an array of values into a dataset matrix
#def create_dataset(dataset, look_back=1):
#	dataX, dataY = [], []
#	for i in range(len(dataset)-look_back-1):
#		a = dataset[i:(i+look_back), 0]
#		dataX.append(a)
#		dataY.append(dataset[i + look_back, 0])
#	return np.array(dataX), np.array(dataY)
 
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		dataX.append(dataset[i:(i+look_back), 0:2])
		dataY.append(dataset[i:(i+look_back), 2])
	return np.array(dataX), np.array(dataY)

# fix random seed for reproducibility
np.random.seed(5)

# load the dataset
#df = read_csv(input_file, header=None, index_col=None, delimiter=',')

df_temp = loadmat(input_file)
df = df_temp['ACCSTRAIN_DISPL']
###########################
#df = loadmat(input_file)
#print(df['Y_test'])
#dff = df['X_test']
###########################

print(df.shape)

# take close price column[5]
#all_y = df[5].values
#X = df.values
#dataset=X.reshape(-1, 1)
#dataset=X

dataset = df
'''
print(type(dataset))
print(dataset.shape)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(np.reshape(dataset, (-1, 3)))

print(dataset.shape)
'''
# split into train and test sets, 50% test data, 50% training data
train_size = int(len(dataset) * 1)
#test_size = len(dataset) - train_size
#train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
train = dataset[0:train_size,:]

# reshape into X=t and Y=t+1, timestep 240
look_back = 6000
trainX, trainY = create_dataset(train, look_back)
#testX, testY = create_dataset(test, look_back)
'''
print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)
'''

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 2, trainX.shape[1]))
#testX = np.reshape(testX, (testX.shape[0], 2, testX.shape[1]))
'''
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 2))
#testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 2))

'''
print(trainX.shape)
print(testX.shape)
'''

# create and fit the LSTM network, optimizer=adam, 25 neurons, dropout 0.1
model = Sequential()
model.add(LSTM(64, input_shape=(2, look_back)))
#model.add(LSTM(64, input_shape=(look_back, 2)))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(trainX, trainY, epochs=10, batch_size=look_back, verbose=1)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

'''

json_file = open('model_balance_timestep=1024_scalerX_neurons=64.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_balance_timestep=1024_scalerX_neurons=64.h5")

# make predictions
trainPredict = loaded_model.predict(trainX)
#testPredict = model.predict(testX)

print(trainPredict.shape)
#print(testPredict.shape)

'''
# invert predictions
trainPredict = displ_scaler.inverse_transform(trainPredict)
trainY = displ_scaler.inverse_transform([trainY])
testPredict = displ_scaler.inverse_transform(testPredict)
testY = displ_scaler.inverse_transform([testY])
'''

'''
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
'''

'''
# shift train predictions for plotting
#trainPredictPlot = np.empty_like(dataset)
trainPredictPlot = np.empty_like(np.reshape(dataset[:,2], (-1, 1)))
trainPredictPlot[:, :] = np.nan
trainPredictPlot[:train_size, :] = trainPredict

# shift test predictions for plotting
#testPredictPlot = np.empty_like(dataset)
testPredictPlot = np.empty_like(np.reshape(dataset[:,2], (-1, 1)))
testPredictPlot[:, :] = np.nan
#testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
#testPredictPlot[len(trainPredict):len(dataset)+look_back, :] = testPredict
testPredictPlot[train_size:len(dataset)+1, :] = testPredict

# plot baseline and predictions
plt.plot(displ_scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.legend(['train', 'predict'], loc='upper left')
#print('testPrices:')
#print(test_size)
#testPrices=displ_scaler.inverse_transform(dataset[test_size+look_back:])
testPrices=displ_scaler.inverse_transform(np.reshape(dataset[train_size:len(dataset)+1:, 2], (-1, 1)))
#print(testPrices)

#print('testPredictions:')
#print(testPredict)

print(testPredict.shape)
print(testPrices.shape)
print(testPredictPlot.shape)
'''

Ref=dataset[:, 2]
# export prediction and actual prices
df = pd.DataFrame(data={"Predicted_Displ": np.around(list(trainPredict.reshape(-1)), decimals=2)})
df.to_csv("Predicted_Displ.csv", sep=';', index=None)
#df = pd.DataFrame(data={"Predicted_Displ_test": np.around(list(testPredict.reshape(-1)), decimals=2)})
#df.to_csv("Predicted_Displ_test.csv", sep=';', index=None)
df = pd.DataFrame(data={"Ref_Displ": np.around(list(Ref.reshape(-1)), decimals=2)})
df.to_csv("Ref_Displ.csv", sep=';', index=None)


# plot the actual price, prediction in test data=red line, actual price=blue line
#plt.plot(testPredictPlot)
#plt.show()

