'''

@author: T.Gosset & A.Hennehart

Helped by A.Kacem
'''
from __future__ import print_function
import os
import glob
import cv2 as cv
import Cropping as cf
from Cropping import faceCropping as fc
import random as rd

path = 'c:/Users/Admin/Desktop/CK+/'

def buildDataSetCK(size):
    
    fold1 =os.listdir(path + 'Emotion')

    print (glob.glob(path + 'Emotion'))

    mat= []
    x = []
    y = []

    i = 0 
    dim = (size[0],size[1])
    while i < len(fold1) :

        fold2= os.listdir(path + 'Emotion/' + fold1[i])
        j = 0

        while j < len(fold2) :
            if len(os.listdir(path + 'Emotion/'+ fold1[i] + '/' + fold2[j] + '/')) > 0 :
                fold3 = glob.glob(path +'cohn-kanade-images/'+ fold1[i] + '/' + fold2[j] + '/*.png')
                k = len(fold3)
                txt = os.listdir(path + 'Emotion/'+ fold1[i] + '/' + fold2[j] + '/')
                y = open(path + 'Emotion/'+ fold1[i] + '/' + fold2[j] + '/' +txt[0])
                label = int(float(y.read()))-1
                seq = []
                for n in range (3):    
                    img  = fold3[k-1]
                    img2 = fc(img)
                    resized = cv.resize(img2, dim, interpolation = cv.INTER_AREA)
                    seq.append(resized)
                    k-= 1
                mat.append([seq, label]) 
            

            j+= 1
        
        i+= 1

    return (mat)


def ShuffleDataSet(mat,k,num_classe):

    sortedSeq=[]

    for i in range(num_classe):
        sortedSeq.append([])

    for data in mat:
        n = data[1]
        sortedSeq[n].append(data)

    for i in range(num_classe):
        rd.shuffle(sortedSeq[i])
    
    k_folder=[]

    for i in range (k):
        k_folder.append([])

    for i in range(num_classe):
        j=0
        left = len(sortedSeq[i])
        for data in (sortedSeq[i]):
            seq=[]
            for n in range (3):
                seq=[[data[0][0],data[1]],[data[0][1],data[1]],[data[0][2],data[1]]]
            if left > k:
                if j >= k:
                    j=0
                k_folder[j].extend(seq)
                j=j+1
                left-=1
            else:
                l=rd.randint(0,k-1)
                k_folder[l].extend(seq)


    return (k_folder)          

