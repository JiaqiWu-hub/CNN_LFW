import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import CNN
import matplotlib.pyplot as plt
import time


'''----------pick the dataset----------'''
# the directory of processed images
output_dir = 'D:\\UF course files\\pattern recognition\\project\\facenet\\lfw1'
images = []
labels = []
names = []
label = 1
threshold = 100 # the threshold of picking the dataset
for name in os.listdir(output_dir):
    class_dir = output_dir + '\\' + name
    if len(os.listdir(class_dir))>threshold:
        for file in os.listdir(class_dir):
                file_path = class_dir + '\\' + file
                img = cv2.imread(file_path)
                images.append(img)
                labels.append(label)
        names.append(name)
        label = label+1
images = np.array(images)
labels = np.array(labels)
# split the dataset to training data and testing data
images_train,images_test,labels_train,labels_test = train_test_split(images[0:1000,:,:,:],labels[0:1000],test_size=0.1)
class_number = len(names)+1
images_train = images_train.astype('float32')
images_test =images_test.astype('float32')
images_train = images_train/255.0
images_test = images_test/255.0
# the labels are converted to binary class matrices
labels_test_b = keras.utils.to_categorical(labels_test, class_number)
labels_train_b = keras.utils.to_categorical(labels_train, class_number)

'''-----------training and testing--------------'''
'''RMSprop'''
#train
start_time = time.time()
optimizer = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
model = CNN.CNNmodel(batchsize=32,epochs=100, classnumber=class_number,optimizer = optimizer)
model.build_model()
model.get_traindata(images_train,labels_train_b)
model.train_model()
train_time = time.time() - start_time
print("RMSprop train_time:%s seconds" % (train_time))
model.save_model('RMSprop3')
# save the loss and accuracy of training
model.save_para('TrainAcc_RMS3','TrainLoss_RMS3')

#test
start_time = time.time()
model.get_testdata(images_test,labels_test_b)
test_loss,test_acc = model.evaluate()
test_time = time.time() - start_time
print("RMSprop test_time:%s seconds" % (test_time))
np.savetxt('Test_RMS3', [test_loss,test_acc])

'''SGD'''
#train
start_time = time.time()
optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model = CNN.CNNmodel(batchsize=32,epochs=100, classnumber=class_number,optimizer = optimizer)
model.build_model()
model.get_traindata(images_train,labels_train_b)
model.train_model()
train_time = time.time() - start_time
print("SGD train_time:%s seconds" % (train_time))
model.save_model('SGD3')
# save the loss and accuracy of training
model.save_para('TrainAcc_SGD3','TrainLoss_SGD3')

#test
start_time = time.time()
model.get_testdata(images_test,labels_test_b)
test_loss,test_acc = model.evaluate()
test_time = time.time() - start_time
print("SGD test_time:%s seconds" % (test_time))
np.savetxt('Test_SGD3', [test_loss,test_acc])

'''Adam'''
#train
start_time = time.time()
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model = CNN.CNNmodel(batchsize=32,epochs=100, classnumber=class_number,optimizer = optimizer)
model.build_model()
model.get_traindata(images_train,labels_train_b)
model.train_model()
train_time = time.time() - start_time
print("adam train_time:%s seconds" % (train_time))
model.save_model('adam3')
# save the loss and accuracy of training
model.save_para('TrainAcc_adam3','TrainLoss_adam3')

#test
start_time = time.time()
model.get_testdata(images_test,labels_test_b)
test_loss,test_acc = model.evaluate()
test_time = time.time() - start_time
print("adam test_time:%s seconds" % (test_time))
np.savetxt('Test_adam3', [test_loss,test_acc])