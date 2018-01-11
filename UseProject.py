import os
import glob
import keras
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
import numpy as np
from keras import backend as K
from sklearn.metrics import confusion_matrix
from ConfusionMatrixBuild import plot_confusion_matrix
import itertools
import matplotlib.pyplot as plt
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.utils import np_utils
from keras.objectives import categorical_crossentropy
from WeightedCrossentropy import weighted_categorical_crossentropy
from LoadCK import buildDataSetCK
from LoadCK import ShuffleDataSet
from LoadCK import BuildDemoSet
from Cropping import faceCropping as fc
import cv2

batch_size = 32
# 7 classes for CK+
num_classes = 7
epochs = 100
#k choice for the k-fold
k = 10
img_rows, img_cols = 96, 96

#Dataset = BuildDemoSet([img_rows, img_cols], 'c:/Users/Alix/Desktop/DemoEd')
weights=[1., 1., 1., 1., 1., 1., 1.]
Dataset2=fc('c:/Users/Alix/Desktop/DemoEd/Alix1.jpg')
Dataset = cv2.resize(Dataset2, (img_rows, img_cols), interpolation = cv2.INTER_AREA)
'''
for i in range(len(Dataset)):
    Dataset[i] = np.array(Dataset[i])
'''
#Dataset = np.array(Dataset)


if (K.image_data_format() == 'channels_first'):
    Dataset = Dataset.reshape(1, img_rows, img_cols, 1)
    input_shape = (1, img_rows, img_cols)
else:
    Dataset = Dataset.reshape(1, img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


model = Sequential()
model.add(Conv2D(96, kernel_size=(5, 5),
                activation='relu',
                input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
#model.add(Dense(32, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(loss=weighted_categorical_crossentropy(weights),
            optimizer='RMSProp',
            metrics=['accuracy'])
results=[]

for i in range (k):
    model.load_weights('model_k=%s_dim=%s_epochs=%s'%(i, img_rows, epochs), by_name=False)
    pred=model.predict(Dataset)
    results.append(pred[0])

#print(results)

result=[]



for j in range (num_classes):
    mean_n=0
    for n in range (k):
        mean_n += results[n][j]
    mean_n = mean_n/k
    result.append(mean_n)

#print (results)
print (result)
print (['Angry','Contempt','Disgust','Fear','Happy','Sadness','Surprise'])
cv2.imshow('img', Dataset2)
cv2.waitKey()
