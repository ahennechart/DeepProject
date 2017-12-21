# Deep Project

DEEP learning keras, facial expression recognition
It is a CNN algorithm which is trained and tested on the database CK+. We obtain a validation accuracy of XX.XX% more or less X%.


## Usage and dependencies 

### Usage : 

The database is saved for three different sizes for a k fold = 5 :
        - if you want to work with a size of (32, 32) you have to use 'ck5fold.npy'
        - if you rather work with a size of (64, 64) you have to use 'ck5fold64.npy'
        - you can also work with a size of (96, 96) you have to use 'ck5fold96x96.npy'

If you want to recreate your own database with a different treatment on, first you will have to download Ck+ database.
                    link pour dl CK+ 
Then, change the path directory in LoadCK.py. 
And finally, uncomment in CNN.py the part "Read and save dataset" and put as comment "Load dataset".

### Dependencies :

The code run under Python 3.5. You will need numpy, keras x.XXXX, opencv(-python as cv2), itertools, sklearn and finally os. 


## Architecture 

on passe par deux conv, un max pooling, en sortie on a 7 neuronnes


```
Schéma de toute l'archi
```


## Data Base and treatment

Our CNN is trained and tested on the CK+ database. Once CK+ load, we realise a face detection and then we crop the picture to resize it to the right dimension (most of the time 32, 32 for us)
When the treatment of the database is done, we realise a k fold method, which permits us to be sure of not testing on what was trained. 


## Results

matrice + graphe de  loss et acc validation


## Authors

* **Alix Hennechart** - *Initial work* -

* **Thomas Gosset** - *Initial work* -

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.


## Contributors

* **Anis Kacem**


