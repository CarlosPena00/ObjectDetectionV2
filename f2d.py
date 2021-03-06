# -*- coding: utf-8 -*-

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'Descriptors/hog')
sys.path.insert(0, 'Descriptors/lbp')
sys.path.insert(0, 'utils')
from lbp import LBP
from hog import HOG
from utils import Utils
import time
IMSIZE = 76
# Constant

HOG_URL = "./Descriptors/hog/"
LBP_URL = "./Descriptors/lbp/"
SAVE = "./Descriptors/Save/"
SAMPLE_IMAGES = "./SampleImages/"
RESULT = "./Results/"
UTILS = "./utils/"
DATA_URL = "./Data/"
NEGATIVE_FOLDER = "negativeFolder.csv"
POSITIVE_FOLDER = "positiveFolder.csv"
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
NUMBER_OF_DIMS = 1764


class F2D:
    'Apply the selected Descriptors into the selected foldes'

    def __init__(self, positive=True, argMin=0, argMax=100, blur=False, openCV=True):
        if positive is True:
            self.csvFolder = POSITIVE_FOLDER
            self.folder = POSITIVE
        else:
            self.csvFolder = NEGATIVE_FOLDER
            self.folder = NEGATIVE
        data = pd.read_csv(UTILS + self.csvFolder)
        self.listOfFolder = data.iloc[:, :].values
        self.argMin = max(argMin, 0)
        self.argMax = min(argMax, self.listOfFolder.shape[0])
        self.blur = blur
        self.openCV = openCV

    def savePlotOfHistogram(self, hist, globalHist, fileName='hist'):
        if self.descriptor == 'HOG':
            url = HOG_URL + HISTOGRAM
        if self.descriptor == 'LBP':
            url = LBP_URL + HISTOGRAM
        hog = hist.ravel()
        histOfHist = np.zeros(shape=9, dtype=float)
        for z in range(0, 9):
            histOfHist[z] = sum(hog[i] for i in range(len(hog)) if i % 9 == z)
            globalHist[z] += histOfHist[z]
        plt.plot(histOfHist)
        plt.savefig(url + fileName + ".png")
        plt.close()

    def transform(self, descriptor='HOG', histogram=False):
        self.descriptor = descriptor
        self.hog = HOG()
        self.lbp = LBP()
        self.Ut = Utils()
        self.histogram = histogram
        for i in range(self.argMin, self.argMax):
            self.folderName = self.listOfFolder[i][0]
            self.typeOfFolder = "." + self.listOfFolder[i][1]
            self.cut = self.listOfFolder[i][2]
            self.numberOfImgs = self.listOfFolder[i][3]
            self.saveFile = self.folderName[:-1] + CSV
            print "----- Start Folder: " + self.folderName + " -----"
            time.sleep(2)
            if descriptor == 'HOG':
                self.url = HOG_URL
                self.dims = NUMBER_OF_DIMS
                self.color = 1
                self.fmt = "%f"
            elif descriptor == 'LBP':
                self.url = LBP_URL
                self.dims = 5776
                self.color = 0
                self.fmt = "%d"
            self.__fold2features()
            
    def __getFeatures(self, img):
        if self.descriptor == 'HOG':
            return self.hog.getOpenCV(img)
        if self.descriptor == 'LBP':
            histLBP = local_binary_pattern(img,8,1)
            return histLBP.ravel()
            
        
                
    def __fold2features(self):
        lista = np.empty([1, self.dims])
        for f in tqdm(range(1, self.numberOfImgs + 1)):
            src = cv2.imread(DATA_URL + self.folderName +
                             str(f) + self.typeOfFolder, self.color)
            rows, cols = src.shape[0], src.shape[1]
            if rows > 1 and cols > 1:
                if self.blur:
                    src = cv2.pyrUp(cv2.pyrDown(src))
                    rows, cols = src.shape[0], src.shape[1]
                if self.cut:
                    maxRows = rows / IMSIZE
                    maxCols = cols / IMSIZE
                else:
                    maxRows = 1
                    maxCols = 1
                for j in range(0, maxRows):
                    for i in range(0, maxCols):
                        if self.cut is True:
                            roi, xMin, xMax, yMin, yMax = Utils.getRoi(
                                src, j, i, px=IMSIZE)
                        else:
                            roi = src
                        rowsR, colsR = roi.shape[0], roi.shape[1]
                        if rowsR < 1 or colsR < 1:
                            print "f2d.py fold2Hog 88 ERRO roi.shape > (1,1)"
                            continue
                        if rowsR != IMSIZE or colsR != IMSIZE:
                            roi = cv2.resize(roi, (IMSIZE, IMSIZE))
                            rowsR, colsR = roi.shape[0], roi.shape[1]

                        hist = self.__getFeatures(roi)
                        lista = np.vstack((lista, hist))
            else:
                print "f2d.py fold2Hog 99 ERRO roi.shape > (1,1)"
                hist = np.zeros(NUMBER_OF_DIMS)
                lista = np.vstack((lista, hist))
        # globalHistMean = globalHist/float(f)
        # Save plot of Histogram
        X = np.delete(lista, (0), axis=0)
        np.savetxt(self.url + self.folder + self.folderName[:-1] + CSV,
                   X, delimiter=',', fmt=self.fmt)


from skimage.feature import local_binary_pattern
import cv2 #12499
if __name__ == "__main__":
    f2d = F2D(positive=False, argMin=1, argMax=2)
    #f2d.transform(descriptor='HOG')
    f2d.transform(descriptor='LBP')
    #src = cv2.imread("./SampleImages/sample01.jpg",0)
    #print src.shape
    #lbp = local_binary_pattern(src,8,1)