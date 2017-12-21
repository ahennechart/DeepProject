from __future__ import print_function
import os
import keras
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import load_model
import numpy as np
from keras import backend as K
from LoadCK import buildDataSetCK
from LoadCK import ShuffleDataSet
from sklearn.metrics import confusion_matrix
from ConfusionMatrixBuild import plot_confusion_matrix
import itertools
import matplotlib.pyplot as plt
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.utils import np_utils

os.environ['KERAS_BACKEND'] = 'tensorflow'

batch_size = 32
# 7 classes for CK+
num_classes = 7
epochs = 35
#k choice for the k-fold
k =5

# input image dimensions

img_rows, img_cols = 32, 32

# Read and save dataset 

'''
DataSet = buildDataSetCK([img_rows, img_cols])
k_folders = ShuffleDataSet(DataSet,k,num_classes)
np.save('ck5fold.npy', k_folders)
'''

# Load dataset (if already saved)

k_folders=np.load('ck5fold.npy')

# Initialise output

result=[]
y=[]
mean = 0.

# Load Folds 

for n in range(k):


    trainSet = []

    # Select Test Folder and Train Folder

    for i in range(k):
        if (i == n):
            testSet = k_folders[n]
        else:
            trainSet.append(k_folders[i])

    # Initialise TrainSet

    x_train = []
    y_train = []

    # Build TrainSet

    for j in range(len(trainSet)):
        for i in range(len(trainSet[j])):
            x_train.append(trainSet[j][i][0])
            y_train.append(trainSet[j][i][1])

    # Initialise TestSet

    x_test = []
    y_test = []

    # Build TestSet by selecting only one image by sequence

    for i in range(0, len(testSet), 3):
        x_test.append(testSet[i][0])
        y_test.append(testSet[i][1])    

    for i in range(len(x_train)):
        x_train[i] = np.array(x_train[i])
    for i in range(len(x_test)):
        x_test[i] = np.array(x_test[i])

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    
    print(x_train.shape)

    y.extend(y_test)
    
    if (K.image_data_format() == 'channels_first'):
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('fold =', n + 1, '/', k)
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print(x_test[0].shape)
    
    #imgplot = plt.imshow(np.squeeze(x_test[0], axis=2))
    #plt.show()
    
    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    # Create model

    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer='Adadelta',
                metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    mean = mean + score[1]
    #y_pred = model.predict(x_test)
    y_pred = model.predict_classes(x_test)
    result.extend(y_pred)
    print(model.summary())

mean = mean/k

print('Total Test accuracy:', mean)
print (model.layers[2].output)

# Compute confusion matrix

cnf_matrix = confusion_matrix(y, result)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Angry','Contempt','Disgust','Fear','Happy','Sadness','Surprise'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()
