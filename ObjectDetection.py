import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import sys
import time
from tqdm import tqdm
sys.path.insert(0, 'machine_learning')
sys.path.insert(0, 'utils')
sys.path.insert(0, 'Descriptors/hog')
sys.path.insert(0, 'Descriptors/lbp')
from machine_learning import MachineLearning
from utils import Utils
from hog import HOG
from lbp import LBP


from skimage.feature import local_binary_pattern

# Constant

HOG_URL = "./Descriptors/hog/"
LBP_URL = "./Descriptors/lbp/"
SAVE = "./Descriptors/Save/"
SAMPLE_IMAGES = "./SampleImages/"
IMG_URL = "sample11.jpg"
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
usePCA = False
NUMBER_OF_DIMS_HOG = 1764
IMSIZE = 76


class ObjectDetection:
    def __init__(self, classifier, stdScaler, hogPCA='', lbpPCA=''):
        self.classifier = classifier
        self.stdScaler = stdScaler
        if type(hogPCA) != str and type(lbpPCA) != str :
            self.usePCA = True
            self.hogPCA = hogPCA
            self.lbpPCA = lbpPCA
        else:
            self.usePCA = False

        
    def run(self, imgUrl, upSample=True, pyr=False, descriptor='HOG'):
        'A face detector'
        self.url = imgUrl
        self.pyr = pyr
        self.descriptor = descriptor
        src = cv2.imread(SAMPLE_IMAGES + imgUrl)
        rows, cols = src.shape[0],src.shape[1]
        if upSample:  
            rows = int(rows * 1.5)
            cols = int(cols * 1.5)    
            srcUp = cv2.resize(src,(cols,rows))
        else:
            srcUp = src
        rows, cols = srcUp.shape[0], srcUp.shape[1]
        self.HOG = HOG()
        self.LBP = LBP()
        self.src = srcUp.copy()
        self.srcPrint = srcUp.copy()
        self.maxRows = rows/IMSIZE
        self.maxCols = cols/IMSIZE
        self.rects = []
        self.__DetectEachBlock()
        
        
    def __DetectEachBlock(self):
        iterator = 0
        enable = True
        Ut = Utils()
        while self.src.shape[0] > 64 and self.src.shape[1] > 64 and enable:
            iterator += 1
            for i in tqdm(range(0, self.maxRows)):
                for j in range(0, self.maxCols):
                    for dX in range(0, 3):
                        for dY in range(0, 3):
                            roi, xMin, xMax, yMin, yMax = Ut.getRoi(self.src, i, j, px=IMSIZE, dy=dX*20, dx=dY*20)
                            rows, cols = roi.shape[0], roi.shape[1]
                            if rows == 0 or cols == 0:
                                break
                            if rows != IMSIZE or cols != IMSIZE:
                                roi = cv2.resize(roi, (IMSIZE, IMSIZE))
                                rows, cols = roi.shape[0], roi.shape[1]    

                            if self.__predict(roi):
                                self.recs_aux = np.array([xMin * iterator, yMin * iterator, xMax * iterator, yMax * iterator]) 
                                self.rects.append(self.recs_aux)
            self.src = cv2.pyrDown(self.src)
            rows, cols = self.src.shape[0],self.src.shape[1]
            self.maxRows = rows/IMSIZE
            self.maxCols = cols/IMSIZE
            if not self.pyr:
                enable = False
        boxes = self.__No_MaxSuppresion(np.asarray(self.rects), 0.3)
        for bx in boxes:
            xMin, yMin, xMax, yMax = bx[0], bx[1], bx[2], bx[3]
            cv2.rectangle(self.srcPrint, (yMin, xMin), (yMax, xMax), (0, 255, 0))       
        cv2.imwrite( RESULT +"LBP_" + self.url , self.srcPrint)


    def __predict(self,src):
        if self.descriptor == 'HOG':
            feature = self.__getHog(src)
        if self.descriptor == 'LBP':
            feature = self.__getLbp(src)
        if self.descriptor =='HOGLBP':
            feature = self.__getHogLbp(src)
        return self.classifier.predict(feature)
        
    def __getHog(self, src):
        histG = self.HOG.getOpenCV(src)
        histGE = self.stdScaler.transform(histG.reshape(1, -1))        
        if self.usePCA:
            histGE = self.hogPCA.transform(histGE)
        return histGE
    
    def __getLbp(self, src):
        src = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        localBin = local_binary_pattern(src,8,1).ravel() 
        localBin = self.stdScaler.transform(localBin.reshape(1, -1))
        if self.usePCA:
            localBin = self.lbpPCA.transform(localBin)
        return localBin
    
    def __getHogLbp(self, src):
        hog = self.__getHog(src)
        lbp = self.__getLbp(src)
        return np.hstack((hog,lbp))
    
    
    def __No_MaxSuppresion(self, boxes, overlapThresh):
        if len(boxes) == 0:
            print "Boxes Vazio"
            return []
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        pick = []
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))
        return boxes[pick].astype("int")



if __name__ == "__main__":

    if len(sys.argv) <= 1:
        print "Error not flags: -c -h rbf rf linear || -l -h rbf rf linear"
    else:
        Ut = Utils()
        if sys.argv[1] == '-c':
            pcaFeature = 550
            if sys.argv[2] == '-h':
                Descriptor = "HOG"
            if sys.argv[2] == '-l':
                Descriptor = "LBP"
            if sys.argv[2] == '-hl':
                Descriptor = "HOGLBP"
                pcaFeature = 1100
                
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
            print "Samples load!"
            print "Positive vs Negative"
            print XP.shape[0],"--- VS ---",XN.shape[0]

            print "---------------Start The Model----------------"
            ML = MachineLearning(X, y)
            if usePCA:
                PCA = ML.PCA(n=pcaFeature)
            else: 
                PCA = ''
            
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
            if usePCA:
                pickle.dump(PCA, open(pcaFile, 'wb'))

            print "-----------------Train The Model--------------"
            time.sleep(1)
            detect = ObjectDetection(classifier, standardScaler, PCA)
            detect.run(IMG_URL)
            #train(classifier, standardScaler, std=1)


        if sys.argv[1] == '-l':
            if sys.argv[2] == '-h':
                Descriptor = "HOG"
                pcaFeature = 550
            if sys.argv[2] == '-l':
                Descriptor = "LBP"
                pcaFeature = 550                
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
            print modelFile
            classifier = pickle.load(open(modelFile, 'rb'))
            standardScaler = pickle.load(open(scalerFile, 'rb'))
            if usePCA:
                PCA = pickle.load(open(pcaFile, 'rb'))
            else:
                PCA = ''
            detect = ObjectDetection(classifier, standardScaler, PCA)
            detect.run(IMG_URL)
            #train(classifier, standardScaler, std=1)
