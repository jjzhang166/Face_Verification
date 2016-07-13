#!/usr/bin/env python
#coding=utf-8
import sys
import numpy as np
import cPickle as pickle
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn import preprocessing
# MaxAbsScaler is in sklearn version 0.17.1 or up

class JointBayesian:
    def __init__(self):
        self.pca = joblib.load("result/pca_model.m")
        self.scaler = joblib.load("result/scale_model.m")
        with open("result/A_con.pkl", "rb") as f:
            self.A = pickle.load(f)
        with open("result/G_con.pkl", "rb") as f:
            self.G = pickle.load(f)
              
    def Verify(self, x1, x2):
        x1.shape = (-1,1)
        x2.shape = (-1,1)
        ratio = np.dot(np.dot(np.transpose(x1),self.A),x1) + np.dot(np.dot(np.transpose(x2),self.A),x2) - 2*np.dot(np.dot(np.transpose(x1),self.G),x2)
        return float(ratio)
    
    def data_pre(self, data):
        data = np.sqrt(data)
        data = np.divide(data, np.repeat(np.sum(data, 1), data.shape[1]).reshape(data.shape[0], data.shape[1]))    
        return data
    
    def verification(self, person1, person2):
        data = np.vstack((person1, person2))
        print data.shape
        data  = self.scaler.transform(data)
        data = self.pca.transform(data)
        print data.shape
        ratio = self.Verify(data[0], data[1])
        print ratio
        return ratio
    

    
    
    
    
    
    
    
    
    
    
    
