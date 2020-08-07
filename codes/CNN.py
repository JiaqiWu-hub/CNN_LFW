import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
import numpy as np
import matplotlib.pyplot as plt

class CNNmodel():
    def __init__(self, batchsize, epochs,classnumber,optimizer):
        self.model = None
        self.batchsize = batchsize
        self.epochs = epochs
        self.classnumber = classnumber
        self.optimizer = optimizer

    def build_model(self):
        self.model = Sequential()
        # Convolution layer and pooling layer
        self.model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding='same', activation='relu', input_shape=(200, 200, 3)))
        self.model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding='same'))
        self.model.add(Dropout(0.25))
        # Convolution layer and pooling layer
        self.model.add(Conv2D(filters = 64, kernel_size = (3, 3),padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        self.model.add(Dropout(0.25))
        # flatten and connection
        self.model.add(Flatten())
        self.model.add(Dense(512,activation='relu'))
        # softmax multi-classification
        self.model.add(Dense(self.classnumber,activation='softmax'))
        self.model.summary()

    def get_traindata(self,images, labels):
        self.images = images
        self.labels = labels
    def get_testdata(self,images_test, labels_test):
        self.images_test = images_test
        self.labels_test = labels_test

    def train_model(self):
        self.model.compile(optimizer=self.optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
        self.history = self.model.fit(self.images, self.labels, epochs=self.epochs, batch_size=self.batchsize)


    def evaluate(self):
        print('\nTesting---------------')
        loss, accuracy = self.model.evaluate(self.images_test, self.labels_test)
        print('test loss;', loss)
        print('test accuracy:', accuracy)
        return loss,accuracy

    def save_model(self,filename):
        self.model.save(filename)
        print('Model saved')

    def load_model(self,filename):
        self.model = keras.models.load_model(filename)
        print('Model loaded')
    def save_para(self, accuracy_name,loss_name):
        np_acc = np.array(self.history.history['accuracy'])
        np_loss = np.array(self.history.history['loss'])
        np.savetxt(accuracy_name, np_acc)
        np.savetxt(loss_name, np_loss)