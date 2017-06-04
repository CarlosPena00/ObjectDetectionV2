import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.svm import LinearSVC


class MachineLearning:
     'Common base class for all ML methods'
     def __init__(self,X,y):
         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                 X, y, test_size=0.30, random_state=0)
         self.sc_X = StandardScaler()
         self.X_train = self.sc_X.fit_transform(self.X_train)
         self.X_test = self.sc_X.transform(self.X_test)
     
     def getClassifier(self):
         self.classifier.fit(self.X_train, self.y_train)
         self.y_pred = self.classifier.predict(self.X_test)
         self.cm = confusion_matrix(self.y_test, self.y_pred)
         print "Confuse Matrix"
         print self.cm
         print (self.cm[0, 0] + self.cm[1, 1]) / float(self.cm.sum())
         return self.classifier, self.cm, self.sc_X
       
     def svm(self, Kernel='rbf'):
         self.classifier = SVC(kernel=Kernel)  # Kernel cam be linear, poly, rbf, sigmoid
         return self.getClassifier();
     
     def linearSvm(self, Ce=0.01):
         self.classifier = LinearSVC(C=Ce)
         return self.getClassifier();
     
     def randomForest(self, N=100, theads=3):
         self.classifier = RandomForestClassifier(
                 n_estimators=N, n_jobs=theads, criterion='entropy')
         return self.getClassifier();
         