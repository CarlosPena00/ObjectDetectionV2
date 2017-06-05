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
sys.path.insert(0, 'Descriptors/hog')
from machine_learning import MachineLearning
from utils import Utils
from hog import HOG
# Constant

HOG_URL = "./Descriptors/hog/"
LBP_URL = "./Descriptors/lbp/"
SAVE = "./Descriptors/Save/"
SAMPLE_IMAGES = "./SampleImages/"
IMG_URL = "sample01.jpg"
RESULT = "./Results/"
UTILS = "./utils/"
HISTOGRAM = "Histogram/"
NEGATIVE = "Negative/"
POSITIVE = "Positive/"
SCALER = "Scaler"
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
        Ut = Utils()
        if sys.argv[1] == '-c':
            if sys.argv[2] == '-h':

                print "------Getting Negative Samples from File------"
                XN = Ut.mergeCSV(positive=False)
                rowsN, colsN = XN.shape
                YN = np.zeros(shape=(rowsN, 1), dtype=int)
                print "------Getting Positive Samples from File------"
                XP = Ut.mergeCSV(positive=True)
                rowsP, colsP = XP.shape
                YP = np.ones(shape=(rowsP, 1), dtype=int)
                X = np.vstack((XP, XN))
                y = np.vstack((YP, YN))
                y = y.ravel()

                print "---------------Start The Model----------------"
                ML = MachineLearning(X, y)

                if sys.argv[3] == 'rbf':
                    modelFile = SAVE + RBF + MODEL + SAV
                    scalerFile = SAVE + RBF + SCALER + SAV
                    classifier, cm, standardScaler = ML.svm()

                if sys.argv[3] == 'rf':
                    modelFile = SAVE + RF + MODEL + SAV
                    scalerFile = SAVE + RF + SCALER + SAV
                    classifier, cm, standardScaler = ML.randomForest(
                        N=100, theads=3)

                if sys.argv[3] == 'linear':
                    modelFile = SAVE + LINEAR + MODEL + SAV
                    scalerFile = SAVE + LINEAR + SCALER + SAV
                    classifier, cm, standardScaler = ML.linearSvm(
                        Ce=0.01)

                print "-----------------Save The Model---------------"
                pickle.dump(classifier, open(modelFile, 'wb'))
                pickle.dump(standardScaler, open(scalerFile, 'wb'))

                print "-----------------Train The Model--------------"
                Ut.faceDetect(classifier, standardScaler, IMG_URL)
                #train(classifier, standardScaler, std=1)

            if sys.argv[2] == 'l':
                print "------NOT YET!------"

        if sys.argv[1] == '-l':
            if sys.argv[2] == '-h':
                if sys.argv[3] == 'rbf':
                    modelFile = SAVE + RBF + MODEL + SAV
                    scalerFile = SAVE + RBF + SCALER + SAV
                if sys.argv[3] == 'rf':
                    modelFile = SAVE + RF + MODEL + SAV
                    scalerFile = SAVE + RF + SCALER + SAV
                if sys.argv[3] == 'linear':
                    modelFile = SAVE + LINEAR + MODEL + SAV
                    scalerFile = SAVE + LINEAR + SCALER + SAV
                classifier = pickle.load(open(modelFile, 'rb'))
                standardScaler = pickle.load(open(scalerFile, 'rb'))
                Ut.faceDetect(classifier, standardScaler, IMG_URL)
                
                #train(classifier, standardScaler, std=1)
