import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import sys
from tqdm import tqdm
sys.path.insert(0, 'machine_learning')
sys.path.insert(0, 'utils')
from machine_learning import MachineLearning

## Constant

HOG_URL = "./Descriptors/hog/"
LBP_URL = "./Descriptors/lbp/"
SAVE = "./Descriptors/Save/"
SAMPLE_IMAGES ="./SampleImages/"
RESULT = "./Results/"
UTILS = "./utils/"
HISTOGRAM = "Histogram/"
NEGATIVE = "Negative/"
POSITIVE = "Positive/"
SCALER ="Scaler"
MODEL = "Model"
LINEAR = "linear"
RBF = "rbf"
RF = "rf"
JPG = ".jpg"
PNG = ".png"
SAV = ".sav"
CSV = ".csv"

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print "Error not flags: -c -h rbf rf linear || -l -h rbf rf linear"
    else:
        if sys.argv[1] == '-c':
            if sys.argv[2] == '-h':              
                print "------Getting Negative Samples from File------"
                XN = mergeX(positive=0)
                rowsN, colsN = XN.shape
                YN = np.zeros(shape=(rowsN, 1), dtype=int)
                print "------Getting Positive Samples from File------"
                XP = mergeX(positive=1)
                rowsP, colsP = XP.shape
                YP = np.ones(shape=(rowsP, 1), dtype=int)
                X = np.vstack((XP, XN))
                y = np.vstack((YP, YN))
                y = y.ravel()
        
                print "---------------Start The Model----------------"
                ML = MachineLearning(X,y)
                
                if sys.argv[3] == 'rbf':
                    modelFile = SAVE+MODEL+RBF+SAV
                    scalerFile = SAVE+SCALER+RBF+SAV
                    classifier,cm, standardScaler = ML.svm(kernel='rbf')
                    
                if sys.argv[3] == 'rf':
                    modelFile = SAVE+MODEL+RF+SAV
                    scalerFile = SAVE+SCALER+RF+SAV
                    classifier,cm, standardScaler = ML.randomForest(N=100,theads=3)
                    
                if sys.argv[3] == 'linear':
                    modelFile = SAVE+MODEL+LINEAR+SAV
                    scalerFile = SAVE+SCALER+LINEAR+SAV
                    X_train,X_test,y_train,y_test,y_pred,classifier,cm,standardScaler = ML.linearSvm(Ce=0.01)
        
                print "-----------------Save The Model---------------"
                pickle.dump(classifier, open(modelFile, 'wb'))
                pickle.dump(standardScaler, open(scalerFile, 'wb'))
        
                print "-----------------Train The Model--------------"
                train(classifier, standardScaler, std=1)
            
            if sys.argv[2] == 'l':
                print "------NOT YET!------"
    
        if sys.argv[1] == '-l':
            if sys.argv[2] == '-h': 
                if sys.argv[3] == 'rbf':
                    modelFile = SAVE+MODEL+RBF+SAV
                    scalerFile = SAVE+SCALER+RBF+SAV
                if sys.argv[3] == 'rf':
                    modelFile = SAVE+MODEL+RF+SAV
                    scalerFile = SAVE+SCALER+RF+SAV
                if sys.argv[3] == 'linear':
                    modelFile = SAVE+MODEL+LINEAR+SAV
                    scalerFile = SAVE+SCALER+LINEAR+SAV
                classifier = pickle.load(open(modelFile, 'rb'))
                standardScaler = pickle.load(open(scalerFile, 'rb'))
                train(classifier, standardScaler, std=1)
    
        