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
    def mergeCSV(self, positive=True, descriptor="HOG"):
        if descriptor == "HOG":
            csvFiles = HOG_URL
        if descriptor == "LBP":
            csvFiles = LBP_URL
        if positive is True:
            csvFiles += POSITIVE
            print csvFiles
            csvList = positiveList + CSV
        else:
            csvFiles += NEGATIVE
            csvList = negativeList + CSV
        dataList = pd.read_csv(csvList).iloc[:, :].values
        xList = np.empty([1, NUMBER_OF_DIMS], dtype="float32")
        for folder in dataList:
            print "---- Start To Get Folder " + csvFiles + folder[0] + " ----"
            dataset = pd.read_csv(csvFiles + folder[0] + CSV, dtype="float32")
            xNew = dataset.iloc[:, :].values
            xList = np.vstack((xList, xNew))
        xList = np.delete(xList, (0), axis=0)
        return xList

    def getRoi(self, src, idX, idY, px=8, dy=0, dx=0):
        xMin = dx + (idX * px)
        xMax = xMin + px
        yMin = dy + (idY * px)
        yMax = yMin + px
        return src[xMin:xMax, yMin:yMax, :], xMin, xMax, yMin, yMax
    
    def getBlock(self, src, idX, idY, px=16):
        xMin = idX * (px / 2)
        xMax = xMin + px
        yMin = idY * (px / 2)
        yMax = yMin + px
        return src[xMin:xMax, yMin:yMax, :]
    
    def faceDetect(self, classifier, stdScaler, imgUrl, upSample=True):
        'A face detector'
        self.classifier = classifier
        self.stdScaler = stdScaler
        self.url = imgUrl
        src = cv2.imread(SAMPLE_IMAGES + imgUrl)
        rows, cols = src.shape[0],src.shape[1]
        if upSample:  
            rows = int(rows * 1.5)
            cols = int(cols * 1.5)    
            srcUp = cv2.resize(src,(rows,cols)) #cv2.pyrDown(src)
        else:
            srcUp = src
        # srcUp = cv2.pyrUp( cv2.pyrDown(src))
        rows, cols = srcUp.shape[0],srcUp.shape[1]
        self.HOG = HOG()
        self.src = srcUp.copy()
        self.srcPrint = srcUp.copy()
        self.maxRows = rows/IMSIZE
        self.maxCols = cols/IMSIZE
        self.rects = []
        self.__DetectEachBlock()
        
        
    def __DetectEachBlock(self):
        iterator = 0
        while self.src.shape[0] > 64 and self.src.shape[1] > 64:
            iterator += 1
            for i in tqdm(range(0, self.maxRows)):
                for j in range(0, self.maxCols):
                    for dX in range(0, 3):
                        for dY in range(0, 3):
                            roi, xMin, xMax, yMin, yMax = self.getRoi(self.src, i, j, px=IMSIZE, dy=dX*20, dx=dY*20)
                            rows, cols = roi.shape[0], roi.shape[1]
                            if rows == 0 or cols == 0:
                                break
                            if rows != IMSIZE or cols != IMSIZE:
                                roi = cv2.resize(roi, (IMSIZE, IMSIZE))
                                rows, cols = roi.shape[0], roi.shape[1]    
                            histG = self.HOG.getOpenCV(roi)
                            histGE = self.stdScaler.transform(histG.reshape(1, -1))
                            if self.classifier.predict(histGE):
                                #cv2.rectangle(self.srcPrint, (yMin, xMin), (yMax, xMax), (0, 0, 255))
                                self.recs_aux = np.array([xMin * iterator, yMin * iterator, xMax * iterator, yMax * iterator]) 
                                self.rects.append(self.recs_aux)
                                #plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                                #cv2.imwrite("Img/"+"ID"+str(j*1000)+str(i*100)+str(dY*10)+str(dX)+"Foi"+".jpg",roi)
            self.src = cv2.pyrDown(self.src)
            rows, cols = self.src.shape[0],self.src.shape[1]
            self.maxRows = rows/IMSIZE
            self.maxCols = cols/IMSIZE
            
        boxes = self.__No_MaxSuppresion(np.asarray(self.rects), 0.3)
        for bx in boxes:
            xMin = bx[0]
            yMin = bx[1]
            xMax = bx[2]
            yMax = bx[3]
            cv2.rectangle(self.srcPrint, (yMin, xMin), (yMax, xMax), (0, 255, 0))
            #print "Box detectado"
        #cv2.imwrite("ID" + str(ID) + "Rect.jpg", src2)
        # print "Image Save On :" + RESULT + self.url
        cv2.imwrite(RESULT + self.url, self.srcPrint)
    #        print "The ID: " + str(ID)
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
    ut = Utils()
    hg = HOG()

   # ut.mergeCSV(True)
