import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers import Input,Dense
from keras.utils import to_categorical

from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras.datasets import mnist

(xTrain,yTrain),(xTest,yTest) = mnist.load_data()

xTrain = (xTrain.reshape(xTrain.shape[0],28,28,1).astype('float32'))/255
xTest = (xTest.reshape(xTest.shape[0],28,28,1).astype('float32'))/255

yTrain = to_categorical(yTrain)
yTest = to_categorical(yTest)
numCol = yTest.shape[1] # type: ignore

def cnn():
    model = Sequential()
    model.add(Input(shape=(28,28,1)))
    
    model.add(Conv2D(16,(5,5),strides=(1,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(8,(2,2),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(numCol,activation='softmax'))

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model = cnn()
model.fit(xTrain,yTrain,validation_data=(xTest,yTest),epochs = 10,batch_size=200,verbose=2) # type: ignore

scores = model.evaluate(xTest,yTest,verbose=0)#type: ignore

print(f"Accuracy:{scores[1]}\n Error: {100-scores[1]*100}")

model.save(".\\handwritingRecognition\\cnnIdentify.keras")