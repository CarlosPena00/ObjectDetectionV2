import cv2
import numpy as np


class LBP:
    def getHistogramOfLBP(self, image):
        # Convert to Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # get rows and cols len of the image
        rows = image.shape[0]
        cols = image.shape[1]

        # initialize empty list of lbp features
        fullLBP = np.float32()
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                cellLBP = np.float32()
                pc = image[i, j]
                p0 = 1 if image[i - 1, j - 1] >= pc else 0
                p1 = 1 if image[i - 1, j] >= pc else 0
                p2 = 1 if image[i - 1, j + 1] >= pc else 0
                p3 = 1 if image[i, j + 1] >= pc else 0
                p4 = 1 if image[i + 1, j + 1] >= pc else 0
                p5 = 1 if image[i + 1, j] >= pc else 0
                p6 = 1 if image[i + 1, j - 1] >= pc else 0
                p7 = 1 if image[i, j - 1] >= pc else 0
                p0 = (int(p0))
                p1 = (int(p1) * (2 << 0))
                p2 = (int(p2) * (2 << 1))
                p3 = (int(p3) * (2 << 2))
                p4 = (int(p4) * (2 << 3))
                p5 = (int(p5) * (2 << 4))
                p6 = (int(p6) * (2 << 5))
                p7 = (int(p7) * (2 << 6))
                summ = p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7
                summ = int(summ)
                cellLBP = np.float32(summ)
                fullLBP = np.append(fullLBP, cellLBP)
        fullLBPMatrix = np.asmatrix(fullLBP)
        return fullLBPMatrix
