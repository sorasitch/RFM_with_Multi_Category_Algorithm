# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:34:15 2023

@author: User
"""


import numpy as np
import pandas as pd
from os.path import join,exists
from os import  remove, makedirs

import time; 
import datetime
import sys
import os
import shutil
import csv
import json

# from googletrans import Translator #error
# from googletrans.client import Translator #fix

import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from mpl_toolkits.mplot3d import Axes3D 
from sklearn.preprocessing import OneHotEncoder,minmax_scale
from sklearn.cluster import DBSCAN 
import matplotlib.colors
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import SpectralClustering, spectral_clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn import mixture

import sys,os,time
# sys.exit()  

import matplotlib
matplotlib.rcParams["figure.dpi"] = 400
matplotlib.rcParams['font.size'] = 8
plt.rcParams['font.size'] = 6

#Translator
# translator = Translator()
# translator = Translator(service_urls=[
#       'translate.google.co.th',
#     ])


'''
*****************************************
Function
*****************************************
'''

def cleaning():
    
    sampleDB=pd.read_excel('sampleDB1.xlsx')
    sampleDB.sort_values('OrderDate', ascending=True, inplace=True)
    sampleDB.reset_index(inplace=True)
    sampleDB.drop(columns=["index"], axis = 1, inplace = True)
    sampleDB.info()
    
    sampleDB['TTLPrice']=(sampleDB['Qty']*sampleDB['UnitPrice'])-sampleDB['Discount']
    
    delLine=sampleDB.loc[(sampleDB['TTLPrice']<0)]
    inxDupl=-1
    inxMatch=-1
    listDelLine=[]
    for i in range(len(delLine['Description'][:])):
        inxDupl=-1
        inxMatch=-1
        for j in reversed(range(len(sampleDB['Description'][:]))):
            if( delLine.iloc[i]['Qty']==sampleDB.iloc[j]['Qty'] )and( delLine.iloc[i]['TTLPrice']==sampleDB.iloc[j]['TTLPrice'] )and( delLine.iloc[i]['VAT']==sampleDB.iloc[j]['VAT'] )and( delLine.iloc[i]['UnitPrice']==sampleDB.iloc[j]['UnitPrice'] )and( str(delLine.iloc[i]['Description'])==str(sampleDB.iloc[j]['Description']) )and( str(delLine.iloc[i]['CustCode'])==str(sampleDB.iloc[j]['CustCode']) )and( str(delLine.iloc[i]['CustName'])==str(sampleDB.iloc[j]['CustName']) )and( delLine.iloc[i]['OrderDate']==sampleDB.iloc[j]['OrderDate'] )and( str(delLine.iloc[i]['SO'])==str(sampleDB.iloc[j]['SO']) ):  
                inxDupl=j
            if( inxDupl>=0 )and( (delLine.iloc[i]['Qty']*-1)==sampleDB.iloc[j]['Qty'] )and( (delLine.iloc[i]['TTLPrice']*-1)==sampleDB.iloc[j]['TTLPrice'] )and( (delLine.iloc[i]['VAT']*-1)==sampleDB.iloc[j]['VAT'] )and( delLine.iloc[i]['UnitPrice']==sampleDB.iloc[j]['UnitPrice'] )and( str(delLine.iloc[i]['Description'])==str(sampleDB.iloc[j]['Description']) )and( str(delLine.iloc[i]['CustCode'])==str(sampleDB.iloc[j]['CustCode']) )and( str(delLine.iloc[i]['CustName'])==str(sampleDB.iloc[j]['CustName']) )and( delLine.iloc[i]['OrderDate']>=sampleDB.iloc[j]['OrderDate'] )and( str(delLine.iloc[i]['SO'])!=str(sampleDB.iloc[j]['SO']) ):  
                inxMatch=j
            if( inxDupl>=0 )and( inxMatch>=0 ):
                print(sampleDB.iloc[inxDupl])
                print([inxDupl])
                print(sampleDB.iloc[inxMatch])
                print([inxMatch])
                listDelLine.extend([inxMatch,inxDupl])
                break
            
    sampleDB1=sampleDB.drop(listDelLine) # inplace=True don't recomment
    sampleDB1.sort_values('OrderDate', ascending=True, inplace=True)
    sampleDB1.reset_index(inplace=True)
    sampleDB1.drop(columns=["index"], axis = 1, inplace = True)
    sampleDB1.info()
    
    delLine=sampleDB1.loc[(sampleDB1['TTLPrice']<0)]
    inxDupl=-1
    inxMatch=-1
    listDelLine=[]
    for i in range(len(delLine['Description'][:])):
        inxDupl=-1
        inxMatch=-1
        for j in range(len(sampleDB1['Description'][:])):
            if( delLine.iloc[i]['Qty']==sampleDB1.iloc[j]['Qty'] )and( delLine.iloc[i]['TTLPrice']==sampleDB1.iloc[j]['TTLPrice'] )and( delLine.iloc[i]['VAT']==sampleDB1.iloc[j]['VAT'] )and( delLine.iloc[i]['UnitPrice']==sampleDB1.iloc[j]['UnitPrice'] )and( str(delLine.iloc[i]['Description'])==str(sampleDB1.iloc[j]['Description']) )and( str(delLine.iloc[i]['CustCode'])==str(sampleDB1.iloc[j]['CustCode']) )and( str(delLine.iloc[i]['CustName'])==str(sampleDB1.iloc[j]['CustName']) )and( delLine.iloc[i]['OrderDate']==sampleDB1.iloc[j]['OrderDate'] )and( str(delLine.iloc[i]['SO'])==str(sampleDB1.iloc[j]['SO']) ):  
                inxDupl=j
            if( inxDupl>=0 )and( (delLine.iloc[i]['Qty']*-1)==sampleDB1.iloc[j]['Qty'] )and( (delLine.iloc[i]['TTLPrice']*-1)==sampleDB1.iloc[j]['TTLPrice'] )and( (delLine.iloc[i]['VAT']*-1)==sampleDB1.iloc[j]['VAT'] )and( delLine.iloc[i]['UnitPrice']==sampleDB1.iloc[j]['UnitPrice'] )and( str(delLine.iloc[i]['Description'])==str(sampleDB1.iloc[j]['Description']) )and( str(delLine.iloc[i]['CustCode'])==str(sampleDB1.iloc[j]['CustCode']) )and( str(delLine.iloc[i]['CustName'])==str(sampleDB1.iloc[j]['CustName']) )and( delLine.iloc[i]['OrderDate']<=sampleDB1.iloc[j]['OrderDate'] )and( str(delLine.iloc[i]['SO'])!=str(sampleDB1.iloc[j]['SO']) ):  
                inxMatch=j
            if( inxDupl>=0 )and( inxMatch>=0 ):
                print(sampleDB1.iloc[inxDupl])
                print([inxDupl])
                print(sampleDB1.iloc[inxMatch])
                print([inxMatch])
                listDelLine.extend([inxDupl,inxMatch])
                break
    
    sampleDB2=sampleDB1.drop(listDelLine) # inplace=True don't recomment
    sampleDB2.sort_values('OrderDate', ascending=True, inplace=True)
    sampleDB2.reset_index(inplace=True)
    sampleDB2.drop(columns=["index"], axis = 1, inplace = True)
    sampleDB2.info()
    
    sampleDB3=sampleDB2.drop(sampleDB2[sampleDB2['TTLPrice']<0].index)
    sampleDB4=sampleDB3.drop(sampleDB3[sampleDB3['Qty']<0].index)
    sampleDB4.sort_values('OrderDate', ascending=True, inplace=True)
    sampleDB4.reset_index(inplace=True)
    sampleDB4.drop(columns=["index"], axis = 1, inplace = True)
    sampleDB4.info()
    
    sampleDB4.to_excel('sampleDB4.xlsx',index=False)
    
    
    
'''
*****************************************
load xlsx file
*****************************************
'''
sampleDB=pd.read_excel('sampleDB4.xlsx')
sampleDB.sort_values('OrderDate', ascending=True, inplace=True)
sampleDB.reset_index(inplace=True)
sampleDB.drop(columns=["index"], axis = 1, inplace = True)
sampleDB.loc[sampleDB['Group']>14,'Group']=55
sampleDB.info()


monetary_md = sampleDB.groupby(['CustCode','Year','Month','OrderDate'])['TTLPrice'].sum().reset_index(name='TTLPrice')
monetary_md = monetary_md.reset_index()
monetary_md.tail()
monetary_md.info()
plt.plot(monetary_md[:]['TTLPrice'])
plt.show()

monetary_md = sampleDB.groupby(['Group','Year','Month','OrderDate'])['TTLPrice'].sum().reset_index(name='TTLPrice')
monetary_md = monetary_md.reset_index()
monetary_md.tail()
monetary_md.info()
plt.plot(monetary_md[:]['TTLPrice'])
plt.show()

monetary_md = sampleDB.groupby(['CustCode','Year','Month'])['TTLPrice'].sum().reset_index(name='TTLPrice')
monetary_md = monetary_md.reset_index()
monetary_md.tail()
monetary_md.info()
plt.plot(monetary_md[:]['TTLPrice'])
plt.show()

monetary_md = sampleDB.groupby(['Group','Year','Month'])['TTLPrice'].sum().reset_index(name='TTLPrice')
monetary_md = monetary_md.reset_index()
monetary_md.tail()
monetary_md.info()
plt.plot(monetary_md[:]['TTLPrice'])
plt.show()
