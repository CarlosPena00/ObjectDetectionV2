import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.insert(0, './../Descriptors/hog')
from hog import HOG

HOG_URL = "./Descriptors/hog/"
LBP_URL = "./Descriptors/lbp/"
SAVE = "./Descriptors/Save/"
SAMPLE_IMAGES = "./SampleImages/"
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

negativeList = "utils/negativeFiles"
positiveList = "utils/positiveFiles"
NUMBER_OF_DIMS = 1764
IMSIZE = 76

class Utils:
    'Commoun function'
    def mergeCSV(self, positive=True, descriptor="HOG", usePCA=True):
        self.usePCA = usePCA
        if descriptor == "HOG":
            csvFiles = HOG_URL
            numberOfFeatures = NUMBER_OF_DIMS 
        if descriptor == "LBP":
            csvFiles = LBP_URL
            numberOfFeatures = 5776
            
        if positive is True:
            csvFiles += POSITIVE
            print csvFiles
            csvList = positiveList + CSV
        else:
            csvFiles += NEGATIVE
            csvList = negativeList + CSV
        dataList = pd.read_csv(csvList).iloc[:, :].values
        xList = np.empty([1, numberOfFeatures], dtype="float32")
        for folder in dataList:
            print "---- Start To Get Folder " + csvFiles + folder[0] + " ----"
            dataset = pd.read_csv(csvFiles + folder[0] + CSV, dtype="float32")
            #dataset2 = pd.read_csv(LBP_URL + negPos + folder[0] + CSV, dtype="float32")
            xNew = dataset.iloc[:, :].values
            xList = np.vstack((xList, xNew))
        xList = np.delete(xList, (0), axis=0)
        return xList

    def getRoi(self, src, idX, idY, px=8, dy=0, dx=0):
        xMin = dx + (idX * px)
        xMax = xMin + px
        yMin = dy + (idY * px)
        yMax = yMin + px
        if len(src.shape) == 3:
            return src[xMin:xMax, yMin:yMax, :], xMin, xMax, yMin, yMax
        if len(src.shape) == 2:
            return src[xMin:xMax, yMin:yMax], xMin, xMax, yMin, yMax
            
    def getBlock(self, src, idX, idY, px=16):
        xMin = idX * (px / 2)
        xMax = xMin + px
        yMin = idY * (px / 2)
        yMax = yMin + px
        return src[xMin:xMax, yMin:yMax, :]
   


if __name__ == "__main__":
    ut = Utils()
    hg = HOG()

   # ut.mergeCSV(True)
