# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 21:36:11 2021
@author: by Kevin

leave one out "LDA" for 2class only. 
"""

#reset  ##"variable"  /  #clear  ##"console"
import numpy as np

def Function_LOO_LDA(Feature , Class_1_Num , Class_2_Num):
    
    Feature_Num = np.size(Feature,1)
    Trial = np.size(Feature,0)
    O_Label = np.hstack([np.ones([1,Class_1_Num]),np.ones([1,Class_2_Num])*2])    # 原始資料的Label
    
    Mean_1_1 = np.zeros((Class_1_Num,Feature_Num))
    Mean_2_1 = np.zeros((Class_1_Num,Feature_Num))
    Cov_1_1 = np.zeros((Class_1_Num,Feature_Num,Feature_Num))
    Cov_2_1 = np.zeros((Class_1_Num,Feature_Num,Feature_Num))
    for i in range(Class_1_Num):
        Class_1 = Feature[0:Class_1_Num,:]
        Class_2 = Feature[Class_1_Num:Trial,:]
        # Class1的"Test"資料逐一更換
        Class_1 = np.delete(Class_1 , i , axis=0 )   # axis "0"為橫列，"1"為直行
        
        # array為將"1D"矩陣轉換為"2D"矩陣，然後在進行轉置
        Mean_1_1[i,:] = np.array([\
                        np.mean(Class_1, axis=0)])
        Mean_2_1[i,:] = np.array([\
                        np.mean(Class_2, axis=0)])
        Cov_1_1[i,:,:] = np.cov(Class_1.T)
        Cov_2_1[i,:,:] = np.cov(Class_2.T)
    Pi_1_1 = np.ones((Class_1_Num,1)).dot(((Class_1_Num-1)/(Trial-1)))
    Pi_2_1 = np.ones((Class_1_Num,1)).dot((Class_2_Num/(Trial-1)))
    
    
    Mean_1_2 = np.zeros((Class_2_Num,Feature_Num))
    Mean_2_2 = np.zeros((Class_2_Num,Feature_Num))
    Cov_1_2 = np.zeros((Class_2_Num,Feature_Num,Feature_Num))
    Cov_2_2 = np.zeros((Class_2_Num,Feature_Num,Feature_Num))
    for i in range(Class_2_Num):
        Class_1 = Feature[0:Class_1_Num,:]
        Class_2 = Feature[Class_1_Num:Trial,:]
        Class_2 = np.delete(Class_2 , i , axis=0 )   # axis "0"為橫列，"1"為直行
        
        # array為將"1D"矩陣轉換為"2D"矩陣，然後在進行轉置
        Mean_1_2[i,:] = np.array([\
                        np.mean(Class_1, axis=0)])
        Mean_2_2[i,:] = np.array([\
                        np.mean(Class_2, axis=0)])
        Cov_1_2[i,:,:] = np.cov(Class_1.T)
        Cov_2_2[i,:,:] = np.cov(Class_2.T)
    Pi_1_2 = np.ones((Class_2_Num,1)).dot((Class_1_Num/(Trial-1)))
    Pi_2_2 = np.ones((Class_2_Num,1)).dot(((Class_2_Num-1)/(Trial-1)))
    

    Mean_1 = np.vstack((Mean_1_1,Mean_1_2))
    Mean_2 = np.vstack((Mean_2_1,Mean_2_2))
    Cov_1 = np.vstack((Cov_1_1,Cov_1_2))
    Cov_2 = np.vstack((Cov_2_1,Cov_2_2))
    # "Class1"&"Class2"的事前機率(prior probability) 
    Pi_1 = np.vstack((Pi_1_1,Pi_1_2))
    Pi_2 = np.vstack((Pi_2_1,Pi_2_2))
    # "Class1"判成"Class2"的錯誤權重，若C12>C21，則D(x)會偏向"Class2"，
    # 即代表越重視第一類別(Class1 data越完整)
    C12 = 1
    C21 = 1
    
    # 2類別的共同共變異矩陣(Common Covariance )
    Cov = np.zeros((Trial,Feature_Num,Feature_Num))
    for i in range(Trial):
        Cov[i,:,:] = Cov_1[i,:,:]*Pi_1[i] + Cov_2[i,:,:]*Pi_2[i,:]
        
    Dx_LDA = np.zeros((1,Trial))
    for i in range(Trial): 
        Test = np.array([Feature[i,:]])
        # LDA決策函式
        Dx_LDA[:,i] = np.array([Mean_1[i,:]-Mean_2[i,:]]).dot(np.linalg.pinv(Cov[i,:,:])).dot(Test.T)-\
                      0.5 * np.array([Mean_1[i,:]-Mean_2[i,:]]).dot(np.linalg.pinv(Cov[i,:,:])).dot(np.array([Mean_1[i,:]+Mean_2[i,:]]).T)-\
                      np.log((C12*Pi_2[i])/(C21*Pi_1[i]))
        
    Label = 1 + (Dx_LDA<0)  
    Check = 0 + (Label==O_Label)     # 判斷與原始Label差異

    True_Positive = np.sum((Check[:,0:Class_1_Num]),1)
    False_Negative = Class_1_Num - True_Positive
    True_Negative = np.sum((Check[:,Class_1_Num:Trial]),1)
    False_Positive = Class_2_Num - True_Negative

    True_Positive_Rate = True_Positive / (True_Positive+False_Negative)
    True_Negative_Rate = True_Negative / (False_Positive+True_Negative)
    False_Positive_Rate = False_Positive / (False_Positive+True_Negative)
    # BL_CR = (TPR+TNR)/2

    # Classification_Rate = (True_Positive+True_Negative) / (True_Positive+False_Negative+True_Negative+False_Positive)
    # print('%.2f%%' % (Classification_Rate*100))
    
    Balance_Classification_Rate = (True_Positive_Rate+True_Negative_Rate)/2
    # print('%.2f%%' % (Balance_Classification_Rate*100))

    return   Balance_Classification_Rate  

    
    