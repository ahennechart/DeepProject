# Deep Project

In this Depp Learning project, we work on the facial expression recognition and try to optimise it. 
It base on a CNN algorithm which is trained and tested on the database CK+. We obtain a validation accuracy of 86% more or less 2%.


## Usage and dependencies 

### Usage : 

The database is saved for three different sizes for a k fold = 5:

- if you want to work with a size of (32, 32) you have to use 'ck5fold.npy'
- if you rather work with a size of (64, 64) you have to use 'ck5fold64.npy'
- you can also work with a size of (96, 96) you have to use 'ck5fold96x96.npy'

If you want to recreate your own database with a different treatment on, first you will have to download Ck+ database.
Then, change the path directory in LoadCK.py. 
And finally, uncomment in CNN.py the part "Read and save dataset" and put as comment "Load dataset".

### Dependencies :

The code run under Python 3.5. You will need numpy, keras 2.0.9, opencv(-python as cv2), itertools, sklearn and finally os. 


## Function : 

### LoadCK.py

buildDataSetCK(size)

```
Permits to build the dataset from CK+ with the correct path and at the dimension choosen 
```

ShuffleDataSet(mat,k,num_classe)

```
Permits to shuffle the dataset into the different folders, thus we equilibrate the dataset to ensure us to have data on each class for the trainning as the testing
```

### Cropping.py

faceCropping(imgpath)

```
Permits to detect faces in the input picture, and then to resize the picture it. 
```

#### ConfusionMatrixBuild.py

plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues) 

```
Show the confusion matrix at the end of the run
```


## Architecture 

on passe par deux conv, un max pooling, en sortie on a 7 neuronnes

```
Schéma de toute l'archi
```


## Data Base and treatment

Our CNN is trained and tested on the CK+ database. 

For the training, we keep only the last 3 pictures,  they will be the most expressive ones. 
On the other hand, we only test one picture to not disturted the result.

Once CK+ load, we realise a face detection and we crop the picture to resize it to the right dimension (most of the time 32, 32 for us)
When the treatment of the database is done, we realise a k fold method, which permits us to be sure of not testing on what was trained.


## Results

matrice + graphe de  loss et acc validation


## Authors

* **Alix Hennechart** - *Initial work* -

* **Thomas Gosset** - *Initial work* -


## Contributors

* **Anis Kacem**


