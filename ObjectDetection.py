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
IMG_URL = "sample03.jpg"
RESULT = "./Results/"
UTILS = "./utils/"
HISTOGRAM = "Histogram/"
NEGATIVE = "Negative/"
POSITIVE = "Positive/"
SCALER = "Scaler"
MODEL = "Model"
LINEAR = "linear"
RBF = "rbf"
PCA_URL = "PCA"
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
                Descriptor = "HOG"
            if sys.argv[2] == '-l':
                Descriptor = "LBP"
            print "------Getting Negative Samples from File------"
            XN = Ut.mergeCSV(positive=False, descriptor=Descriptor)
            rowsN, colsN = XN.shape
            YN = np.zeros(shape=(rowsN, 1), dtype=int)
            print "------Getting Positive Samples from File------"
            XP = Ut.mergeCSV(positive=True, descriptor=Descriptor)
            rowsP, colsP = XP.shape
            YP = np.ones(shape=(rowsP, 1), dtype=int)
            X = np.vstack((XP, XN))
            y = np.vstack((YP, YN))
            y = y.ravel()

            print "---------------Start The Model----------------"
            ML = MachineLearning(X, y)
            PCA = ML.PCA()

            if sys.argv[3] == 'rbf':
                modelFile = SAVE + Descriptor + RBF + MODEL + SAV
                scalerFile = SAVE + Descriptor + RBF + SCALER + SAV
                pcaFile = SAVE + Descriptor + RBF + PCA_URL + SAV
                classifier, cm, standardScaler = ML.svm()

            if sys.argv[3] == 'rf':
                modelFile = SAVE + Descriptor + RF + MODEL + SAV
                scalerFile = SAVE + Descriptor + RF + SCALER + SAV
                pcaFile = SAVE + Descriptor + RF + PCA_URL + SAV
                classifier, cm, standardScaler = ML.randomForest(
                    N=100, theads=3)

            if sys.argv[3] == 'linear':
                modelFile = SAVE + Descriptor + LINEAR + MODEL + SAV
                scalerFile = SAVE + Descriptor + LINEAR + SCALER + SAV
                pcaFile = SAVE + Descriptor + LINEAR + PCA_URL + SAV
                classifier, cm, standardScaler = ML.linearSvm(
                    Ce=0.01)

            print "-----------------Save The Model---------------"
            pickle.dump(classifier, open(modelFile, 'wb'))
            pickle.dump(standardScaler, open(scalerFile, 'wb'))
            pickle.dump(PCA, open(pcaFile, 'wb'))

            print "-----------------Train The Model--------------"
            Ut.faceDetect(classifier, standardScaler, IMG_URL, PCA)
            #train(classifier, standardScaler, std=1)

            if sys.argv[2] == 'l':
                print "------NOT YET!------"

        if sys.argv[1] == '-l':
            if sys.argv[2] == '-h':
                if sys.argv[3] == 'rbf':
                    modelFile = SAVE + Descriptor + RBF + MODEL + SAV
                    scalerFile = SAVE + Descriptor + RBF + SCALER + SAV
                    pcaFile = SAVE + Descriptor + RBF + PCA_URL + SAV
                if sys.argv[3] == 'rf':
                    modelFile = SAVE + Descriptor + RF + MODEL + SAV
                    scalerFile = SAVE + Descriptor + RF + SCALER + SAV
                    pcaFile = SAVE + Descriptor + RF + PCA_URL + SAV
                if sys.argv[3] == 'linear':
                    modelFile = SAVE + Descriptor + LINEAR + MODEL + SAV
                    scalerFile = SAVE + Descriptor + LINEAR + SCALER + SAV
                    pcaFile = SAVE + Descriptor + LINEAR + PCA_URL + SAV
                classifier = pickle.load(open(modelFile, 'rb'))
                standardScaler = pickle.load(open(scalerFile, 'rb'))
                PCA = pickle.load(open(pcaFile, 'rb'))

                Ut.faceDetect(classifier, standardScaler, IMG_URL, PCA)
                
                #train(classifier, standardScaler, std=1)
