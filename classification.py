from keras import Sequential
from keras.layers import Dense,Input
from keras.utils import to_categorical as tc
import pandas as pd
import os
import matplotlib.pyplot as plt
from keras.datasets import mnist

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(xTrain,yTrain),(xTest,yTest) = mnist.load_data()

numPixels = xTrain.shape[1]*xTrain.shape[2]

xTrain = (xTrain.reshape(xTrain.shape[0],numPixels).astype('float32'))/255
xTest = (xTest.reshape(xTest.shape[0],numPixels).astype('float32'))/255

yTrain,yTest = tc(yTrain),tc(yTest)


model = Sequential()
model.add(Input(shape=(numPixels,)))
model.add(Dense(numPixels,activation='relu'))
model.add(Dense(100,activation='relu'))
ncols = yTest.shape[1] # type: ignore
model.add(Dense(ncols,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(xTrain,yTrain,epochs=10)

# scores = model.evaluate(xTest,yTest,verbose=0)

# print(f"Accuracy: {scores[1]}\nError: {1-scores[1]}")

model.save("identifyNumbers.keras")