import pandas as pd
import numpy as np


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

negativeList = "utils/negativeFiles"
positiveList = "utils/positiveFiles"
NUMBER_OF_DIMS = 1764

class Utils:
    'Commoun function'
    
    def mergeCSV(self,positive=True, descriptor="HOG"):
        if descriptor == "HOG":
            csvFiles = HOG_URL
        if descriptor == "LBP":
            csvFiles = LBP_URL
        if positive == True:
            csvFiles += POSITIVE
            print csvFiles
            csvList = positiveList + CSV
        else:
            csvFiles += NEGATIVE
            csvList = negativeList + CSV
        dataList = pd.read_csv(csvList).iloc[:,:].values
        xList = np.empty([1, NUMBER_OF_DIMS], dtype="float32")
        for folder in dataList:
            print "----- Start To Get Folder " + csvFiles + folder[0] + " -----"
            dataset = pd.read_csv(csvFiles + folder[0] + CSV, dtype="float32")
            xNew = dataset.iloc[:, :].values
            xList = np.vstack((xList, xNew))
        xList = np.delete(xList, (0), axis=0)
        return xList



if __name__ == "__main__":
    ut = Utils()
    
   #ut.mergeCSV(True)
    