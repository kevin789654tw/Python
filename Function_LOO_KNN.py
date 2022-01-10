# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 10:43:33 2021
@author: by Kevin

leave one out "K-NN" for 2class only. 
"""

#reset  ##"variable"  /  #clear  ##"console"
import numpy as np
import scipy.stats as st

def Function_LOO_KNN(Feature , Class_1_Num , Class_2_Num):
    
    Trial = np.size(Feature,0)
    O_Label = np.hstack([np.ones([1,Class_1_Num]),np.ones([1,Class_2_Num])*2])    # 原始資料的Label
    
    # 1筆當Test，其餘剩下當Training
    Distance = np.zeros((Trial,Trial))    # 給"Distance"空間
    for i in range(Trial):
        Feature_Transpose = Feature.T     # "T"為轉置
        Copy_Matrix = np.tile(Feature[i,:],(Trial,1)).T     # "tile"為repmat
        Distance[i,:] = np.sqrt(\
                        np.sum(\
                        np.power([Feature_Transpose-Copy_Matrix],2),1))     # "power"為平方
        
    Rank = np.argsort((Distance),axis=1)     # 格子裡的數字為原本資料的地方
    K_Rank = Rank[:,1:4]     # 不取第一排的自己，並將"K"取3
    
    K_Label = 1 + (K_Rank>Class_1_Num)     # boolean，Class_1的Label為1
    pass # st修改
    Vote = st.mode(K_Label.T,0)[0][0]     # "Mode"為投票
    Check = 0 + (Vote==O_Label)     # 判斷與原始Label差異
    
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
