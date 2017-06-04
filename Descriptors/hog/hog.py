import cv2
import numpy as np
import matplotlib.pyplot as plt

class HOG:
    
    def __init__(self, winSize=(64, 64), blockSize=(16,16), blockStride=(8,8), cellSize=(8,8),
                 nbins=9, derivAperture=1, winSigma=4.0, histogramNormType=0, L2HysThreshold=2.0000000000000001e-01,
                 gammaCorrection=0, nlevels = 64):
        self.winSize = winSize
        self.blockSize = blockSize
        self.blockStride = blockStride
        self.cellSize = cellSize
        self.nbins = nbins
        self.derivAperture = derivAperture
        self.winSigma = winSigma
        self.histogramNormType = histogramNormType
        self.L2HysThreshold = L2HysThreshold
        self.gammaCorrection = gammaCorrection
        self.nlevels = nlevels

    
    def getOpenCV(self, src):
        hog = cv2.HOGDescriptor(self.winSize, self.blockSize,
                            self.blockStride, self.cellSize, self.nbins,
                            self.derivAperture, self.winSigma,
                            self.histogramNormType, self.L2HysThreshold,
                            self.gammaCorrection, self.nlevels)
        winStride = (8, 8)
        padding = (8, 8)
        locations = ((10, 20), )
        self.hist = hog.compute(src, winStride, padding, locations)
        self.hist = np.transpose(self.hist).ravel()
        return self.hist
    
    
    
