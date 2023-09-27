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
    



def KMeansClustering(rfm_normalized=None):
    #KMeans Clustering
    kmeans = KMeans(n_clusters=4, max_iter=50)
    kmeans.fit(rfm_normalized)
    
    ssd = []
    silh_avg = []
    range_n_clusters = [2,3,4,5,6,7,8,9,10]
    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
        kmeans.fit(rfm_normalized)
        
        ssd.append(kmeans.inertia_) #Elbow Curve
        #Silhouette score
        cluster_labels = kmeans.labels_
        silhouette_avg = silhouette_score(rfm_normalized, cluster_labels)
        silh_avg.append(silhouette_avg)
    
    ypoints = ssd
    plt.plot(ypoints)
    plt.show()
    
    ypoints = silh_avg
    plt.plot(ypoints)
    plt.show()
    
      
def plot_sillohette(range_n_clusters,samples, assignments, name=''):
    silhouette = [silhouette_score(samples, a) for a in assignments]
    n_clusts = range_n_clusters#range(2, len(silhouette) + 2)
    plt.bar(n_clusts, silhouette)
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette score')
    plt.title(name)
    for i in range(len(n_clusts)):
        plt.text(n_clusts[i], silhouette[i], "%.3f" %silhouette[i], ha = 'center')
    plt.savefig(".\\picture\\Score\\SC-"+name+".png",dpi=400)
    plt.show()
 
def plot_3D_clustering(X, Y, Z, T,labels,title=""):
 
    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    
    plt.cla()
    area = np.pi * ( T)**2  
    # plt.ylabel('Age', fontsize=18)
    # plt.xlabel('Income', fontsize=16)
    # plt.zlabel('Education', fontsize=16)
    ax.set_xlabel(X.name)
    ax.set_ylabel(Y.name)
    ax.set_zlabel(Z.name)
    # ax.set_title(title,fontsize=8)
    
    scatter = ax.scatter(X, Y, Z,s=area, c= labels.astype(np.float), alpha=0.5)
    #ax.legend()
    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                    loc="best", title="Cluster")
    ax.add_artist(legend1)
    
    fig.suptitle(title)
    # plt.title(title,fontsize=8)
    
    title=title.replace(',', '_').replace(' ', '').replace(':', '-')
    fig.savefig(".\\picture\\3D\\"+title+".png",dpi=400)
    # plt.savefig(".\\picture\\"+title+".png",dpi=400)
    
    plt.show()
    
def plot_2D_clustering(X, Y, T,labels,title=""):
    
    fig, ax = plt.subplots()
    
    area = np.pi * ( T)**2  
    scatter = ax.scatter(X, Y, s=area, c=labels.astype(np.float), alpha=0.5)
    
    legend1 = ax.legend(*scatter.legend_elements(),
                    loc="best", title="Cluster")
    ax.add_artist(legend1)

    plt.xlabel(X.name)
    plt.ylabel(Y.name)
    plt.title(title)
    
    title=title.replace(',', '_').replace(' ', '').replace(':', '-')
    plt.savefig(".\\picture\\2D\\"+title+".png",dpi=400)
    
    plt.show()
    
def plot_2D_clustering_rev0(X, Y, T,labels,title=""):
    area = np.pi * ( T)**2  
    plt.scatter(X, Y, s=area, c=labels.astype(np.float), alpha=0.5)
    plt.xlabel(X.name)
    plt.ylabel(Y.name)
    plt.title(title)
    
    title=title.replace(',', '_').replace(' ', '').replace(':', '-')
    plt.savefig(".\\picture\\2D\\"+title+".png",dpi=400)
    
    plt.show()
    
def boxplot_clustering(X='cluster', Y='Monetary', dataset=None, title=""):
    sns.boxplot(x=X, y=Y, data=dataset).set(title=title)
    title=title.replace(',', '_').replace(' ', '').replace(':', '-')
    plt.savefig(".\\picture\\BOX\\"+title+".png",dpi=400)
    plt.show()

def KMeansClustering_rev1(rfm_normalized=None,range_n_clusters = [3,4,5,6,7,8,9,10],name=''):
    # KMeans Clustering
    # SC, you want to find a model with SC values close to 1. If the SC is consistently less than zero the clustering model is probably not that useful.
    # The within cluster sum of squares (WCSS) and between cluster sum of squares (BCSS) are used for K-means clustering only. Ideally a good K-menas cluster model should have small WCSS and large BCSS.
    
    ssd = []
    silh_avg = []
    assignments = []
    km_models = []

    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters)
        assignment = kmeans.fit_predict(rfm_normalized)

        assignments.append(assignment)
        km_models.append(kmeans)
        ssd.append(kmeans.inertia_) #Elbow Curve
        #Silhouette score
        cluster_labels = kmeans.labels_
        silhouette_avg = silhouette_score(rfm_normalized, cluster_labels)
        silh_avg.append(silhouette_avg)
        
    bestClusIndex = silh_avg.index(max(silh_avg))
    bestClus = range_n_clusters[bestClusIndex]
    
    xpoints = range_n_clusters
    ypoints = ssd
    plt.plot(xpoints,ypoints)
    plt.title(name+', Best Cluster='+str(bestClus)+'-Elbow Curve')
    plt.savefig(".\\picture\\Score\\Elbow Curve-"+name+', Best Cluster='+str(bestClus)+".png",dpi=400)
    plt.show()

    ypoints = silh_avg
    plt.plot(xpoints,ypoints)
    plt.title(name+', Best Cluster='+str(bestClus)+'-Silhouette score')
    plt.savefig(".\\picture\\Score\\Silhouette score-"+name+', Best Cluster='+str(bestClus)+".png",dpi=400)
    plt.show()

    plot_sillohette(range_n_clusters, rfm_normalized, assignments,name+', Best Cluster='+str(bestClus))
    
    return bestClus, assignments, km_models

def DBSCANClustering(X=None, name=''):
    
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=1.5, vmax=4.5)
    
    # X=rfm_l_normalized
    epsMax=int(X.std().max())
    epsStep=epsMax/20;
    eps=0
    
    epsList=[]
    clusterList=[]
    noiseList=[]
    assignments=[]
    dbscan=[]
    
    for i in range(20):
        eps=round((eps+epsStep), 3)
        db = DBSCAN(eps=eps, min_samples=10)
        assignment=db.fit_predict(X)
        labels = db.labels_
    
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        
        print("eps: %f" % eps)
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        
        if n_clusters_ > 1 :
            epsList.append(eps)
            clusterList.append(n_clusters_)
            noiseList.append(n_noise_)
            assignments.append(assignment)
            dbscan.append(db)
        
    
    silhouette = [silhouette_score(X, a) for a in assignments]
    
    bestClusIndex = silhouette.index(max(silhouette))
    bestClus = clusterList[bestClusIndex]
    
    # n_clusts = range(1, len(silhouette) + 1)
    n_clusts = [str(clusterList[x])+','+str(epsList[x]) for x in range(len(clusterList))]
    c = [x for x in range(1, len(silhouette) + 1)]
    
    # plt.bar(n_clusts, silhouette, color=cmap(norm(c)),label=c)
    
    for i in range(len(c)): #Loop over color dictionary
        plt.bar(n_clusts[i], silhouette[i],color=cmap(norm(c[i])),label=noiseList[i])
        
    for i in range(len(n_clusts)):
        plt.text(n_clusts[i], silhouette[i], "%.3f" %silhouette[i], ha = 'center')
    
    plt.xlabel('Clusters, Eps')
    plt.ylabel('Silhouette score')
    plt.title(name+', Best Cluster='+str(clusterList[bestClusIndex])+', Eps='+str(epsList[bestClusIndex])+'-Silhouette score')
    plt.legend(title="Noise")
    plt.savefig(".\\picture\\Score\\"+'SC-'+name+', Best Cluster='+str(clusterList[bestClusIndex])+', Eps='+str(epsList[bestClusIndex])+".png",dpi=400)
    plt.show()
    
    return bestClusIndex,bestClus,epsList, clusterList, noiseList, assignments, dbscan



def AffinityPropagationClustering(X=None, name=''):

    # X=rfm_normalized
    
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
    
    preferenceList=[]
    clusterList=[]
    assignments=[]
    affinityPro=[]
    preferences=[ x*-1 for x in range(100)]
    for p in preferences :
        try :
            af = AffinityPropagation(preference=p,random_state=0)
            assignment = af.fit_predict(X)
            cluster_centers_indices = af.cluster_centers_indices_
            labels = af.labels_
            n_clusters_ = len(cluster_centers_indices)   
            print("preference: %f" % p)
            print("Estimated number of clusters: %d" % n_clusters_)
            
            if n_clusters_ > 1 :
                preferenceList.append(p)
                clusterList.append(n_clusters_)
                assignments.append(assignment)
                affinityPro.append(af)
        except :
            pass
        
    pfrMin=int(min(preferenceList))-0
    pfrStep=pfrMin/20;
    pfr=0
    
    preferenceList=[]
    clusterList=[]
    assignments=[]
    affinityPro=[]
    for i in range(20):
        try :
            pfr=round((pfr+pfrStep), 3)
            af = AffinityPropagation(preference=pfr,random_state=0)
            assignment = af.fit_predict(X)
            cluster_centers_indices = af.cluster_centers_indices_
            labels = af.labels_
            n_clusters_ = len(cluster_centers_indices)   
            print("preference: %f" % pfr)
            print("Estimated number of clusters: %d" % n_clusters_)
            
            if n_clusters_ > 1 :
                preferenceList.append(pfr)
                clusterList.append(n_clusters_)
                assignments.append(assignment)
                affinityPro.append(af)
        except :
            pass
        
        
        
    silhouette = [silhouette_score(X, a) for a in assignments]
    
    bestClusIndex = silhouette.index(max(silhouette))
    bestClus = clusterList[bestClusIndex]
    
    # n_clusts = range(1, len(silhouette) + 1)
    n_clusts = [str(clusterList[x])+","+str(x) for x in range(len(clusterList))]
    c = [x for x in range(1, len(silhouette) + 1)]
    
    # plt.bar(n_clusts, silhouette)
    
    for i in range(len(c)): #Loop over color dictionary
        plt.bar(n_clusts[i], silhouette[i],color=cmap(norm(c[i])),label=preferenceList[i])
    
    for i in range(len(n_clusts)):
        plt.text(n_clusts[i], silhouette[i], "%.3f" %silhouette[i], ha = 'center')
    
    plt.xlabel('Clusters, index')
    plt.ylabel('Silhouette score')
    plt.title(name+', Best Cluster='+str(clusterList[bestClusIndex])+', Perference='+str(preferenceList[bestClusIndex])+'-Silhouette score')
    plt.legend(title="Preference")
    plt.savefig(".\\picture\\Score\\"+'SC-'+name+', Best Cluster='+str(clusterList[bestClusIndex])+', Perference='+str(preferenceList[bestClusIndex])+".png",dpi=400)
    plt.show()
    
    return bestClusIndex,bestClus,preferenceList,clusterList,assignments,affinityPro



def MeanShiftClustering(X=None, name=''):

    # X=rfm_normalized
    
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
    
    quantileList=[]
    clusterList=[]
    assignments=[]
    meanShift=[]
    quantiles=[ x*0.1 for x in range(1,11)]
    for q in quantiles :
        q=round(q, 3)
        # The following bandwidth can be automatically detected using
        # bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
        bandwidth = estimate_bandwidth(X, quantile=q)
        
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        assignment = ms.fit_predict(X)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        
        print("quantile: %f" % q)
        print("number of estimated clusters : %d" % n_clusters_)
        
        if n_clusters_ > 2 :
            quantileList.append(q)
            clusterList.append(n_clusters_)
            assignments.append(assignment)
            meanShift.append(ms)
    
    silhouette = [silhouette_score(X, a) for a in assignments]
    
    bestClusIndex = silhouette.index(max(silhouette))
    bestClus = clusterList[bestClusIndex]
    
    # n_clusts = range(1, len(silhouette) + 1)
    n_clusts = [str(clusterList[x])+","+str(x) for x in range(len(clusterList))]
    c = [x for x in range(1, len(silhouette) + 1)]
    
    # plt.bar(n_clusts, silhouette)
    
    for i in range(len(c)): #Loop over color dictionary
        plt.bar(n_clusts[i], silhouette[i],color=cmap(norm(c[i])),label=quantileList[i])
    
    for i in range(len(n_clusts)):
        plt.text(n_clusts[i], silhouette[i], "%.3f" %silhouette[i], ha = 'center')
    
    plt.xlabel('Clusters, index')
    plt.ylabel('Silhouette score')
    plt.title(name+', Best Cluster='+str(clusterList[bestClusIndex])+', Quantile='+str(quantileList[bestClusIndex])+'-Silhouette score')
    plt.legend(title="Quantile")
    plt.savefig(".\\picture\\Score\\"+'SC-'+name+', Best Cluster='+str(clusterList[bestClusIndex])+', Quantile='+str(quantileList[bestClusIndex])+".png",dpi=400)
    plt.show()


    return bestClusIndex,bestClus,quantileList,clusterList,assignments,meanShift


def SpectralClustering_(X=None, name=''):

    # X=rfm_normalized
    
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
    
    clusters=[]
    clusterList=[]
    assignments=[]
    Spectral=[]
    n_clusters=[ x for x in range(2,11)]
    for n in n_clusters:
        sc = SpectralClustering(n_clusters=n)
        assignment = sc.fit_predict(X)
        labels = sc.labels_
        
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        
        print("clusters: %d" % n)
        print("number of estimated clusters : %d" % n_clusters_)
        
        if n_clusters_ > 1 :
            clusters.append(n)
            clusterList.append(n_clusters_)
            assignments.append(assignment)
            Spectral.append(sc)
    
    silhouette = [silhouette_score(X, a) for a in assignments]
    
    bestClusIndex = silhouette.index(max(silhouette))
    bestClus = clusterList[bestClusIndex]
    
    # n_clusts = range(1, len(silhouette) + 1)
    n_clusts = [str(clusterList[x])+","+str(x) for x in range(len(clusterList))]
    c = [x for x in range(1, len(silhouette) + 1)]
    
    # plt.bar(n_clusts, silhouette)
    
    for i in range(len(c)): #Loop over color dictionary
        plt.bar(n_clusts[i], silhouette[i],color=cmap(norm(c[i])),label=clusters[i])
    
    for i in range(len(n_clusts)):
        plt.text(n_clusts[i], silhouette[i], "%.3f" %silhouette[i], ha = 'center')
    
    plt.xlabel('Clusters, index')
    plt.ylabel('Silhouette score')
    plt.title(name+', Best Cluster='+str(clusterList[bestClusIndex])+', n_clusters='+str(n_clusters[bestClusIndex])+'-Silhouette score')
    plt.legend(title="n_clusters")
    plt.savefig(".\\picture\\Score\\"+'SC-'+name+', Best Cluster='+str(clusterList[bestClusIndex])+', n_clusters='+str(n_clusters[bestClusIndex])+".png",dpi=400)
    plt.show()
    
    return bestClusIndex,bestClus,clusters,clusterList,assignments,Spectral


def AgglomerativeClustering_(X=None, name=''):

    # X=rfm_normalized
    
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
    
    clusters=[]
    linkages=[]
    clusterList=[]
    assignments=[]
    Agglomerative=[]
    n_clusters=[ x for x in range(3,7)]
    
    for n in n_clusters:
        for linkage in ("ward", "average", "complete", "single"):
            ag = AgglomerativeClustering(linkage=linkage, n_clusters=n)
            assignment = ag.fit_predict(X)
            labels = ag.labels_
            
            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            
            print("n_clusters: %s\d" % n)
            print("linkage: %s" % linkage)
            print("number of estimated clusters : %d" % n_clusters_)
            
            if n_clusters_ > 1 :
                clusters.append(n)
                linkages.append(linkage)
                clusterList.append(n_clusters_)
                assignments.append(assignment)
                Agglomerative.append(ag)
    
    silhouette = [silhouette_score(X, a) for a in assignments]
    
    bestClusIndex = silhouette.index(max(silhouette))
    bestClus = clusterList[bestClusIndex]
    
    # n_clusts = range(1, len(silhouette) + 1)
    n_clusts = [str(clusterList[x])+','+str(linkages[x]) for x in range(len(clusterList))]
    c = [x for x in range(1, len(silhouette) + 1)]
    
    # plt.bar(n_clusts, silhouette, color=cmap(norm(c)),label=c)
    
    for i in range(len(c)): #Loop over color dictionary
        plt.bar(str(i), silhouette[i], color=cmap(norm(c[i])),label=n_clusts[i])
    
    for i in range(len(n_clusts)):
        plt.text(n_clusts[i], silhouette[i], "%.3f" %silhouette[i], ha = 'center')
    
    plt.xlabel('index')
    plt.ylabel('Silhouette score')
    plt.title(name+', Best Cluster='+str(clusterList[bestClusIndex])+', linkage='+str(linkages[bestClusIndex])+'-Silhouette score')
    plt.legend(title="Clusters, linkage")
    plt.savefig(".\\picture\\Score\\"+'SC-'+name+', Best Cluster='+str(clusterList[bestClusIndex])+', linkage='+str(linkages[bestClusIndex])+".png",dpi=400)
    plt.show()
    
    return bestClusIndex,bestClus,linkage,clusterList,assignments,Agglomerative



def GaussianMixtureClustering(X=None, name=''):
    
    # X=rfm_normalized
    
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=1.5, vmax=4.5)
    
    n_component=[]
    clusterList=[]
    assignments=[]
    gaussianMixture=[]
    n_comp=[ x for x in range(1,11)]
    
    for n in n_comp:
        gm = mixture.GaussianMixture(n_components=2)
        assignment=gm.fit_predict(X)
        labels = assignment #gm.labels_
        
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        
        print("n_components: %d" % n)
        print("number of estimated clusters : %d" % n_clusters_)
        
        if n_clusters_ > 1 :
            n_component.append(n)
            clusterList.append(n_clusters_)
            assignments.append(assignment)
            gaussianMixture.append(gm)
            
    silhouette = [silhouette_score(X, a) for a in assignments]
    
    bestClusIndex = silhouette.index(max(silhouette))
    bestClus = clusterList[bestClusIndex]
    
    # n_clusts = range(1, len(silhouette) + 1)
    n_clusts = [str(clusterList[x])+","+str(n_component[x]) for x in range(len(clusterList))]
    c = [x for x in range(1, len(silhouette) + 1)]
    
    # plt.bar(n_clusts, silhouette)
    
    for i in range(len(c)): #Loop over color dictionary
        plt.bar(n_clusts[i], silhouette[i],color=cmap(norm(c[i])),label=n_component[i])
        
    for i in range(len(n_clusts)):
        plt.text(n_clusts[i], silhouette[i], "%.3f" %silhouette[i], ha = 'center')
    
    plt.xlabel('Clusters, n_component')
    plt.ylabel('Silhouette score')
    plt.title(name+', Best Cluster='+str(clusterList[bestClusIndex])+', n_component='+str(n_component[bestClusIndex])+'-Silhouette score')
    # plt.legend(title="n_component")
    plt.savefig(".\\picture\\Score\\"+'SC-'+name+', Best Cluster='+str(clusterList[bestClusIndex])+', n_component='+str(n_component[bestClusIndex])+".png",dpi=400)
    plt.show()
 
    return bestClusIndex,bestClus,n_component,clusterList,assignments,gaussianMixture

'''
*****************************************
cleaning the xlsx file
*****************************************
'''
# cleaning()

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


'''
*****************************************
RFM and RFML data set
*****************************************
'''
#Monetary
monetary = sampleDB.groupby('CustCode')['TTLPrice'].sum()
monetary = monetary.reset_index()
monetary.tail()
monetary.info()

#RFM + K-Means extended
#Frequency
frequency = sampleDB.groupby('CustCode')['SO-line'].count()
frequency = frequency.reset_index()
frequency.tail()
frequency.info()

#Recency
# sampleDB['InvoiceDate'] = pd.to_datetime(sampleDB['InvoiceDate'],format='%m/%d/%Y %H:%M')
sampleDB['Diff'] = max(sampleDB['OrderDate'])-sampleDB['OrderDate']
recency = sampleDB.groupby('CustCode')['Diff'].min()
recency = recency.reset_index()
recency.tail()
recency['Diff'] = recency['Diff'].dt.days
recency.tail()
recency.info()

#Length
firstOrder = sampleDB.groupby('CustCode')['OrderDate'].min().reset_index(name='OrderDate')  
lastOrder = sampleDB.groupby('CustCode')['OrderDate'].max().reset_index(name='OrderDate')  
length = lastOrder.copy()
length['OrderDate'] = lastOrder['OrderDate']-firstOrder['OrderDate']
# length = length.reset_index()
length['OrderDate'] = length['OrderDate'].dt.days
length.tail()
length.info()


#RFM
rfm = pd.merge(recency, frequency, on='CustCode', how='inner')
rfm = pd.merge(rfm, monetary, on='CustCode', how='inner')
rfm.columns = ['CustCode', 'Recency', 'Frequency', 'Monetary']
rfm.info()

#RFM - L
rfm_l = pd.merge(recency, frequency, on='CustCode', how='inner')
rfm_l = pd.merge(rfm_l, monetary, on='CustCode', how='inner')
rfm_l = pd.merge(rfm_l, length, on='CustCode', how='inner')
rfm_l.columns = ['CustCode', 'Recency', 'Frequency', 'Monetary', 'Length']
rfm_l.info()


'''
*****************************************
RFM and RFML data set group by Variable Model  
*****************************************
'''
# #Variable Model
# #Monetary
# monetary_md = sampleDB.groupby(['CustCode','Model'])['TTLPrice'].sum()
# monetary_md = monetary_md.reset_index()
# monetary_md.tail()
# monetary_md.info()

# #Frequency
# frequency_md = sampleDB.groupby(['CustCode','Model'])['SO-line'].count()
# frequency_md = frequency_md.reset_index()
# frequency_md.tail()
# frequency_md.info()

# #Recency
# # sampleDB['InvoiceDate'] = pd.to_datetime(sampleDB['InvoiceDate'],format='%m/%d/%Y %H:%M')
# sampleDB['DiffModel'] = max(sampleDB['OrderDate'])-sampleDB['OrderDate']
# recency_md = sampleDB.groupby(['CustCode','Model'])['DiffModel'].min()
# recency_md = recency_md.reset_index()
# recency_md.tail()
# recency_md['DiffModel'] = recency_md['DiffModel'].dt.days
# recency_md.tail()
# recency_md.info()

# #Length
# firstOrder_md = sampleDB.groupby(['CustCode','Model'])['OrderDate'].min().reset_index(name='OrderDate')  
# lastOrder_md = sampleDB.groupby(['CustCode','Model'])['OrderDate'].max().reset_index(name='OrderDate')  
# length_md = lastOrder_md.copy()
# length_md['OrderDate'] = lastOrder_md['OrderDate']-firstOrder_md['OrderDate']
# # length_md = length_md.reset_index()
# length_md['OrderDate'] = length_md['OrderDate'].dt.days
# length_md.tail()
# length_md.info()


# #RFM by Variable Model
# rfm_md = pd.merge(recency_md, frequency_md, on=['CustCode','Model'], how='inner')
# rfm_md = pd.merge(rfm_md, monetary_md, on=['CustCode','Model'], how='inner')
# rfm_md.columns = ['CustCode', 'Model', 'Recency', 'Frequency', 'Monetary']
# rfm_md.info()

# #RFM - L by Variable Model
# rfm_l_md = pd.merge(recency_md, frequency_md, on=['CustCode','Model'], how='inner')
# rfm_l_md = pd.merge(rfm_l_md, monetary_md, on=['CustCode','Model'], how='inner')
# rfm_l_md = pd.merge(rfm_l_md, length_md, on=['CustCode','Model'], how='inner')
# rfm_l_md.columns = ['CustCode', 'Model', 'Recency', 'Frequency', 'Monetary', 'Length']
# rfm_l_md.info()


'''
*****************************************
RFM and RFML data set group by Variable Group 
***************************************** 
'''
# #Variable Group
# #Monetary
# monetary_gp = sampleDB.groupby(['CustCode','Group'])['TTLPrice'].sum()
# monetary_gp = monetary_gp.reset_index()
# monetary_gp.tail()
# monetary_gp.info()

# #Frequency
# frequency_gp = sampleDB.groupby(['CustCode','Group'])['SO-line'].count()
# frequency_gp = frequency_gp.reset_index()
# frequency_gp.tail()
# frequency_gp.info()

# #Recency
# # sampleDB['InvoiceDate'] = pd.to_datetime(sampleDB['InvoiceDate'],format='%m/%d/%Y %H:%M')
# sampleDB['DiffGroup'] = max(sampleDB['OrderDate'])-sampleDB['OrderDate']
# recency_gp = sampleDB.groupby(['CustCode','Group'])['DiffGroup'].min()
# recency_gp = recency_gp.reset_index()
# recency_gp.tail()
# recency_gp['DiffGroup'] = recency_gp['DiffGroup'].dt.days
# recency_gp.tail()
# recency_gp.info()

# #Length
# firstOrder_gp = sampleDB.groupby(['CustCode','Group'])['OrderDate'].min().reset_index(name='OrderDate')  
# lastOrder_gp = sampleDB.groupby(['CustCode','Group'])['OrderDate'].max().reset_index(name='OrderDate')  
# length_gp = lastOrder_gp.copy()
# length_gp['OrderDate'] = lastOrder_gp['OrderDate']-firstOrder_gp['OrderDate']
# # length_gp = length_gp.reset_index()
# length_gp['OrderDate'] = length_gp['OrderDate'].dt.days
# length_gp.tail()
# length_gp.info()


# #RFM by Variable Group
# rfm_gp = pd.merge(recency_gp, frequency_gp, on=['CustCode','Group'], how='inner')
# rfm_gp = pd.merge(rfm_gp, monetary_gp, on=['CustCode','Group'], how='inner')
# rfm_gp.columns = ['CustCode', 'Group', 'Recency', 'Frequency', 'Monetary']
# rfm_gp.info()

# #RFM - L by Variable Group
# rfm_l_gp = pd.merge(recency_gp, frequency_gp, on=['CustCode','Group'], how='inner')
# rfm_l_gp = pd.merge(rfm_l_gp, monetary_gp, on=['CustCode','Group'], how='inner')
# rfm_l_gp = pd.merge(rfm_l_gp, length_gp, on=['CustCode','Group'], how='inner')
# rfm_l_gp.columns = ['CustCode', 'Group', 'Recency', 'Frequency', 'Monetary', 'Length']
# rfm_l_gp.info()


# sys.exit()  

#Variable Group with OneHot,Sum
group = sampleDB.groupby('Group')['SO-line'].count().reset_index(name='Count')  
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit_transform(group[['Group']])
enc_df = pd.DataFrame(enc.transform(sampleDB[['Group']]).toarray())
group_colm=[str(x) for x in group['Group']]
enc_df.columns = group_colm

sampleDB=sampleDB.join(enc_df)

# # sum of Qty by Group
# for g in group_colm :
#     sampleDB.loc[sampleDB[g]>0, g]=sampleDB['Qty']


grps=[]
for g in group_colm :
    grps.append(sampleDB.groupby('CustCode')[g].sum().reset_index(name=g))

#RFM by Variable Group with OneHot,Sum
rfm_grp = pd.merge(recency, frequency, on='CustCode', how='inner')
rfm_grp = pd.merge(rfm_grp, monetary, on='CustCode', how='inner')
header=['CustCode', 'Recency', 'Frequency', 'Monetary']
for i in range(len(group_colm)) :
    rfm_grp = pd.merge(rfm_grp, grps[i], on='CustCode', how='inner')
    header.append(group_colm[i])
rfm_grp.columns = header
rfm_grp.info()

#RFM-L by Variable Group with OneHot,Sum
rfml_grp = pd.merge(recency, frequency, on='CustCode', how='inner')
rfml_grp = pd.merge(rfml_grp, monetary, on='CustCode', how='inner')
rfml_grp = pd.merge(rfml_grp, length, on='CustCode', how='inner')
header=['CustCode', 'Recency', 'Frequency', 'Monetary', 'Length']
for i in range(len(group_colm)) :
    rfml_grp = pd.merge(rfml_grp, grps[i], on='CustCode', how='inner')
    header.append(group_colm[i])
rfml_grp.columns = header
rfml_grp.info()


'''
*****************************************
Scaling and normalize
*****************************************
'''
# Scaling and normalize
scaler = StandardScaler()
#RFM
rfm_normalized = rfm[['Recency','Frequency','Monetary']]
rfm_normalized = scaler.fit_transform(rfm_normalized)
rfm_normalized = pd.DataFrame(rfm_normalized)
#RFM - L
rfm_l_normalized = rfm_l[['Recency','Frequency','Monetary', 'Length']]
rfm_l_normalized = scaler.fit_transform(rfm_l_normalized)
rfm_l_normalized = pd.DataFrame(rfm_l_normalized)


# #RFM by Variable Model
# rfm_md_normalized = rfm_md[['Recency','Frequency','Monetary']]
# rfm_md_normalized = scaler.fit_transform(rfm_md_normalized)
# rfm_md_normalized = pd.DataFrame(rfm_md_normalized)
# #RFM - L by Variable Model
# rfm_l_md_normalized = rfm_l_md[['Recency','Frequency','Monetary', 'Length']]
# rfm_l_md_normalized = scaler.fit_transform(rfm_l_md_normalized)
# rfm_l_md_normalized = pd.DataFrame(rfm_l_md_normalized)


# #RFM by Variable Group
# rfm_gp_normalized = rfm_gp[['Recency','Frequency','Monetary']]
# rfm_gp_normalized = scaler.fit_transform(rfm_gp_normalized)
# rfm_gp_normalized = pd.DataFrame(rfm_gp_normalized)
# #RFM - L by Variable Group
# rfm_l_gp_normalized = rfm_l_gp[['Recency','Frequency','Monetary', 'Length']]
# rfm_l_gp_normalized = scaler.fit_transform(rfm_l_gp_normalized)
# rfm_l_gp_normalized = pd.DataFrame(rfm_l_gp_normalized)

#RFM by Variable Group with OneHot,Sum
header=['Recency', 'Frequency', 'Monetary']
for i in range(len(group_colm)) :
    header.append(group_colm[i])
rfm_grp_normalized = rfm_grp[header]
rfm_grp_normalized = scaler.fit_transform(rfm_grp_normalized)
rfm_grp_normalized = pd.DataFrame(rfm_grp_normalized)

#RFM-L by Variable Group with OneHot,Sum
header=['Recency', 'Frequency', 'Monetary', 'Length']
for i in range(len(group_colm)) :
    header.append(group_colm[i])
rfml_grp_normalized = rfml_grp[header]
rfml_grp_normalized = scaler.fit_transform(rfml_grp_normalized)
rfml_grp_normalized = pd.DataFrame(rfml_grp_normalized)


'''
*****************************************
KMeans Clustering
*****************************************
'''
# #KMeans Clustering
# #RFM
# KMeansClustering(rfm_normalized)
# #RFM - L
# KMeansClustering(rfm_l_normalized)
# #RFM by Variable Model
# KMeansClustering(rfm_md_normalized)
# #RFM - L by Variable Model
# KMeansClustering(rfm_l_md_normalized)
# #RFM by Variable Group
# KMeansClustering(rfm_gp_normalized)
# #RFM - L by Variable Group
# KMeansClustering(rfm_l_gp_normalized)


#KMeans Clustering
def plot_2D_rfm_heatmap(rfm,assignments,T=2,title="",title1=""):
    rfm_list = ['Recency', 'Frequency', 'Monetary']
    for y in rfm_list :
        for x in rfm_list :
            plot_2D_clustering(rfm[x], rfm[y], T,assignments,title+', '+y+x+title1)

def plot_2D_rfml_heatmap(rfm,assignments,T=2,title="",title1=""):
    rfml_list = ['Recency', 'Frequency', 'Monetary', 'Length']
    for y in rfml_list :
        for x in rfml_list :
            plot_2D_clustering(rfm[x], rfm[y], T,assignments,title+', '+y+x+title1)
    
            
range_n_clusters=[3,4,5,6,7,8,9,10]           
     
# sys.exit() 
       
#RFM
name='KMeans, RFM'
bestClus, assignments, km_models = KMeansClustering_rev1(rfm_normalized,range_n_clusters,name)
rfm['Cluster_RFM']=assignments[range_n_clusters.index(bestClus)]
rfm.to_excel('rfm_km_'+ str(bestClus) +'.xlsx',index=False)
for h in ['Recency', 'Frequency', 'Monetary']:
    boxplot_clustering(X='Cluster_RFM', Y=h, dataset=rfm, title=name+", Cluster="+str(bestClus)+', '+h)
plot_3D_clustering(rfm['Recency'],
                   rfm['Frequency'],
                   rfm['Monetary'],
                   2,
                   assignments[range_n_clusters.index(bestClus)],
                   name+", Cluster="+str(bestClus))
plot_2D_rfm_heatmap(rfm,
                    assignments[range_n_clusters.index(bestClus)],
                    2,
                    name+", Cluster="+str(bestClus))
#RFM - L
name='KMeans, RFM-L'
bestClus, assignments, km_models = KMeansClustering_rev1(rfm_l_normalized,range_n_clusters,name)
rfm_l['Cluster_RFML']=assignments[range_n_clusters.index(bestClus)]
rfm_l.to_excel('rfml_km_'+ str(bestClus) +'.xlsx',index=False)
for h in ['Recency', 'Frequency', 'Monetary', 'Length']:
    boxplot_clustering(X='Cluster_RFML', Y=h, dataset=rfm_l, title=name+", Cluster="+str(bestClus)+', '+h)
plot_3D_clustering(rfm_l['Recency'],
                   rfm_l['Frequency'],
                   rfm_l['Monetary'],
                   minmax_scale(rfm_l['Length'],feature_range=(0, 10)),
                   assignments[range_n_clusters.index(bestClus)],
                   name+", Cluster="+str(bestClus)+', Marker size : Length')
plot_2D_rfml_heatmap(rfm_l,
                     assignments[range_n_clusters.index(bestClus)],
                     2,
                     name+", Cluster="+str(bestClus))
# #RFM by Variable Model
# name='KMeans, RFM by Model'
# bestClus, assignments, km_models = KMeansClustering_rev1(rfm_md_normalized,range_n_clusters,name)
# plot_3D_clustering(rfm_md['Recency'],
#                     rfm_md['Frequency'],
#                     rfm_md['Monetary'],
#                     2,
#                     assignments[range_n_clusters.index(bestClus)],
#                     name+", Cluster="+str(bestClus))
# plot_2D_rfm_heatmap(rfm_md,
#                     assignments[range_n_clusters.index(bestClus)],
#                     2,
#                     name+", Cluster="+str(bestClus))
# #RFM - L by Variable Model
# name='KMeans, RFM-L by Model'
# bestClus, assignments, km_models = KMeansClustering_rev1(rfm_l_md_normalized,range_n_clusters,name)
# plot_3D_clustering(rfm_l_md['Recency'],
#                     rfm_l_md['Frequency'],
#                     rfm_l_md['Monetary'],
#                     minmax_scale(rfm_l_md['Length'],feature_range=(0, 10)),
#                     assignments[range_n_clusters.index(bestClus)],
#                     name+", Cluster="+str(bestClus)+', Marker size : Length')
# plot_2D_rfml_heatmap(rfm_l_md,
#                       assignments[range_n_clusters.index(bestClus)],
#                       2,
#                       name+", Cluster="+str(bestClus))
# # RFM by Variable Group
# name='KMeans, RFM by Group'
# bestClus, assignments, km_models = KMeansClustering_rev1(rfm_gp_normalized,range_n_clusters,name)
# plot_3D_clustering(rfm_gp['Recency'],
#                     rfm_gp['Frequency'],
#                     rfm_gp['Monetary'],
#                     2,
#                     assignments[range_n_clusters.index(bestClus)],
#                     name+", Cluster="+str(bestClus))
# plot_2D_rfm_heatmap(rfm_gp,
#                     assignments[range_n_clusters.index(bestClus)],
#                     2,
#                     name+", Cluster="+str(bestClus))
# #RFM - L by Variable Group
# name='KMeans, RFM-L by Group'
# bestClus, assignments, km_models = KMeansClustering_rev1(rfm_l_gp_normalized,range_n_clusters,name)
# plot_3D_clustering(rfm_l_gp['Recency'],
#                     rfm_l_gp['Frequency'],
#                     rfm_l_gp['Monetary'],
#                     minmax_scale(rfm_l_gp['Length'],feature_range=(0, 10)),
#                     assignments[range_n_clusters.index(bestClus)],
#                     name+", Cluster="+str(bestClus)+', Marker size : Length')
# plot_2D_rfml_heatmap(rfm_l_gp,
#                       assignments[range_n_clusters.index(bestClus)],
#                       2,
#                       name+", Cluster="+str(bestClus))


#RFM by Variable Group with OneHot,Sum
name='KMeans, RFM by Group with OneHot'
bestClus, assignments, km_models = KMeansClustering_rev1(rfm_grp_normalized,range_n_clusters,name)
rfm_grp['Cluster_RFM_Group']=assignments[range_n_clusters.index(bestClus)]
rfm_grp.to_excel('rfm_grp_km_'+ str(bestClus) +'.xlsx',index=False)
for h in ['Recency', 'Frequency', 'Monetary']:
    boxplot_clustering(X='Cluster_RFM_Group', Y=h, dataset=rfm_grp, title=name+", Cluster="+str(bestClus)+', '+h)
plot_3D_clustering(rfm_grp['Recency'],
                   rfm_grp['Frequency'],
                   rfm_grp['Monetary'],
                   2,
                   assignments[range_n_clusters.index(bestClus)],
                   name+", Cluster="+str(bestClus))
for g in group_colm :
    plot_3D_clustering(rfm_grp['Recency'],
                       rfm_grp['Frequency'],
                       rfm_grp['Monetary'],
                       minmax_scale(rfm_grp[g],feature_range=(0, 10)),
                       assignments[range_n_clusters.index(bestClus)],
                       name+", Cluster="+str(bestClus)+', Marker size : '+g)
plot_2D_rfm_heatmap(rfm_grp,
                    assignments[range_n_clusters.index(bestClus)],
                    2,
                    name+", Cluster="+str(bestClus))
for g in group_colm :
    plot_2D_rfm_heatmap(rfm_grp,
                        assignments[range_n_clusters.index(bestClus)],
                        minmax_scale(rfm_grp[g],feature_range=(0, 10)),
                        name+", Cluster="+str(bestClus),
                        ', Marker size : '+g)


#RFM-L by Variable Group with OneHot,Sum
name='KMeans, RFM-L by Group with OneHot'
bestClus, assignments, km_models = KMeansClustering_rev1(rfml_grp_normalized,range_n_clusters,name)
rfml_grp['Cluster_RFML_Group']=assignments[range_n_clusters.index(bestClus)]
rfml_grp.to_excel('rfml_grp_km_'+ str(bestClus) +'.xlsx',index=False)
for h in ['Recency', 'Frequency', 'Monetary', 'Length']:
    boxplot_clustering(X='Cluster_RFML_Group', Y=h, dataset=rfml_grp, title=name+", Cluster="+str(bestClus)+', '+h)
plot_3D_clustering(rfml_grp['Recency'],
                   rfml_grp['Frequency'],
                   rfml_grp['Monetary'],
                   minmax_scale(rfml_grp['Length'],feature_range=(0, 10)),
                   assignments[range_n_clusters.index(bestClus)],
                   name+", Cluster="+str(bestClus)+', Marker size : Length')
for g in group_colm :
    plot_3D_clustering(rfml_grp['Recency'],
                       rfml_grp['Frequency'],
                       rfml_grp['Monetary'],
                       minmax_scale(rfml_grp[g],feature_range=(0, 10)),
                       assignments[range_n_clusters.index(bestClus)],
                       name+", Cluster="+str(bestClus)+', Marker size : '+g)
plot_2D_rfml_heatmap(rfml_grp,
                    assignments[range_n_clusters.index(bestClus)],
                    2,
                    name+", Cluster="+str(bestClus))
for g in group_colm :
    plot_2D_rfml_heatmap(rfml_grp,
                        assignments[range_n_clusters.index(bestClus)],
                        minmax_scale(rfml_grp[g],feature_range=(0, 10)),
                        name+", Cluster="+str(bestClus),
                        ', Marker size : '+g)





'''
*****************************************
DBSCAN Clustering
*****************************************
'''
#DBSCAN Clustering
#RFM
name='DBSCAN, RFM'
bestClusIndex,bestClus,epsList, clusterList, noiseList, assignments, dbscan = DBSCANClustering(X=rfm_normalized, name=name)

#RFM - L
name='DBSCAN, RFM-L'
bestClusIndex,bestClus,epsList, clusterList, noiseList, assignments, dbscan = DBSCANClustering(X=rfm_l_normalized, name=name)

#RFM by Variable Group with OneHot,Sum
name='DBSCAN, RFM by Group with OneHot'
bestClusIndex,bestClus,epsList, clusterList, noiseList, assignments, dbscan = DBSCANClustering(X=rfm_grp_normalized, name=name)

#RFM-L by Variable Group with OneHot,Sum
name='DBSCAN, RFM-L by Group with OneHot'
bestClusIndex,bestClus,epsList, clusterList, noiseList, assignments, dbscan = DBSCANClustering(X=rfml_grp_normalized, name=name)




'''
*****************************************
AffinityPropagation Clustering
*****************************************
'''
#AffinityPropagation Clustering
#RFM
name='AffinityPropagation, RFM'
bestClusIndex,bestClus,preferenceList,clusterList,assignments,affinityPro = AffinityPropagationClustering(X=rfm_normalized, name=name)
rfm['Cluster_RFM']=assignments[bestClusIndex]
rfm.to_excel('rfm_aff_'+ str(bestClus) +'.xlsx',index=False)
for h in ['Recency', 'Frequency', 'Monetary']:
    boxplot_clustering(X='Cluster_RFM', Y=h, dataset=rfm, title=name+", Cluster="+str(bestClus)+', '+h)
plot_3D_clustering(rfm['Recency'],
                   rfm['Frequency'],
                   rfm['Monetary'],
                   2,
                   assignments[bestClusIndex],
                   name+", Cluster="+str(bestClus))
plot_2D_rfm_heatmap(rfm,
                    assignments[bestClusIndex],
                    2,
                    name+", Cluster="+str(bestClus))

#RFM - L
name='AffinityPropagation, RFM-L'
bestClusIndex,bestClus,preferenceList,clusterList,assignments,affinityPro = AffinityPropagationClustering(X=rfm_l_normalized, name=name)
rfm_l['Cluster_RFML']=assignments[bestClusIndex]
rfm_l.to_excel('rfml_aff_'+ str(bestClus) +'.xlsx',index=False)
for h in ['Recency', 'Frequency', 'Monetary', 'Length']:
    boxplot_clustering(X='Cluster_RFML', Y=h, dataset=rfm_l, title=name+", Cluster="+str(bestClus)+', '+h)
plot_3D_clustering(rfm_l['Recency'],
                   rfm_l['Frequency'],
                   rfm_l['Monetary'],
                   minmax_scale(rfm_l['Length'],feature_range=(0, 10)),
                   assignments[bestClusIndex],
                   name+", Cluster="+str(bestClus)+', Marker size : Length')
plot_2D_rfml_heatmap(rfm_l,
                     assignments[bestClusIndex],
                     2,
                     name+", Cluster="+str(bestClus))

#RFM by Variable Group with OneHot,Sum
name='AffinityPropagation, RFM by Group with OneHot'
bestClusIndex,bestClus,preferenceList,clusterList,assignments,affinityPro = AffinityPropagationClustering(X=rfm_grp_normalized, name=name)
rfm_grp['Cluster_RFM_Group']=assignments[bestClusIndex]
rfm_grp.to_excel('rfm_grp_aff_'+ str(bestClus) +'.xlsx',index=False)
for h in ['Recency', 'Frequency', 'Monetary']:
    boxplot_clustering(X='Cluster_RFM_Group', Y=h, dataset=rfm_grp, title=name+", Cluster="+str(bestClus)+', '+h)
plot_3D_clustering(rfm_grp['Recency'],
                   rfm_grp['Frequency'],
                   rfm_grp['Monetary'],
                   2,
                   assignments[bestClusIndex],
                   name+", Cluster="+str(bestClus))
for g in group_colm :
    plot_3D_clustering(rfm_grp['Recency'],
                       rfm_grp['Frequency'],
                       rfm_grp['Monetary'],
                       minmax_scale(rfm_grp[g],feature_range=(0, 10)),
                       assignments[bestClusIndex],
                       name+", Cluster="+str(bestClus)+', Marker size : '+g)
plot_2D_rfm_heatmap(rfm_grp,
                    assignments[bestClusIndex],
                    2,
                    name+", Cluster="+str(bestClus))
for g in group_colm :
    plot_2D_rfm_heatmap(rfm_grp,
                        assignments[bestClusIndex],
                        minmax_scale(rfm_grp[g],feature_range=(0, 10)),
                        name+", Cluster="+str(bestClus),
                        ', Marker size : '+g)

#RFM-L by Variable Group with OneHot,Sum
name='AffinityPropagation, RFM-L by Group with OneHot'
bestClusIndex,bestClus,preferenceList,clusterList,assignments,affinityPro = AffinityPropagationClustering(X=rfml_grp_normalized, name=name)
rfml_grp['Cluster_RFML_Group']=assignments[bestClusIndex]
rfml_grp.to_excel('rfml_grp_aff_'+ str(bestClus) +'.xlsx',index=False)
for h in ['Recency', 'Frequency', 'Monetary', 'Length']:
    boxplot_clustering(X='Cluster_RFML_Group', Y=h, dataset=rfml_grp, title=name+", Cluster="+str(bestClus)+', '+h)
plot_3D_clustering(rfml_grp['Recency'],
                   rfml_grp['Frequency'],
                   rfml_grp['Monetary'],
                   minmax_scale(rfml_grp['Length'],feature_range=(0, 10)),
                   assignments[bestClusIndex],
                   name+", Cluster="+str(bestClus)+', Marker size : Length')
for g in group_colm :
    plot_3D_clustering(rfml_grp['Recency'],
                       rfml_grp['Frequency'],
                       rfml_grp['Monetary'],
                       minmax_scale(rfml_grp[g],feature_range=(0, 10)),
                       assignments[bestClusIndex],
                       name+", Cluster="+str(bestClus)+', Marker size : '+g)
plot_2D_rfml_heatmap(rfml_grp,
                    assignments[range_n_clusters.index(bestClus)],
                    2,
                    name+", Cluster="+str(bestClus))
for g in group_colm :
    plot_2D_rfml_heatmap(rfml_grp,
                        assignments[bestClusIndex],
                        minmax_scale(rfml_grp[g],feature_range=(0, 10)),
                        name+", Cluster="+str(bestClus),
                        ', Marker size : '+g)



'''
*****************************************
MeanShift Clustering
*****************************************
'''
#AffinityPropagation Clustering
#RFM
name='MeanShift, RFM'
bestClusIndex,bestClus,quantileList,clusterList,assignments,meanShift = MeanShiftClustering(X=rfm_normalized, name=name)
rfm['Cluster_RFM']=assignments[bestClusIndex]
rfm.to_excel('rfm_ms_'+ str(bestClus) +'.xlsx',index=False)
for h in ['Recency', 'Frequency', 'Monetary']:
    boxplot_clustering(X='Cluster_RFM', Y=h, dataset=rfm, title=name+", Cluster="+str(bestClus)+', '+h)
plot_3D_clustering(rfm['Recency'],
                   rfm['Frequency'],
                   rfm['Monetary'],
                   2,
                   assignments[bestClusIndex],
                   name+", Cluster="+str(bestClus))
plot_2D_rfm_heatmap(rfm,
                    assignments[bestClusIndex],
                    2,
                    name+", Cluster="+str(bestClus))

#RFM - L
name='MeanShift, RFM-L'
bestClusIndex,bestClus,quantileList,clusterList,assignments,meanShift = MeanShiftClustering(X=rfm_l_normalized, name=name)
rfm_l['Cluster_RFML']=assignments[bestClusIndex]
rfm_l.to_excel('rfml_ms_'+ str(bestClus) +'.xlsx',index=False)
for h in ['Recency', 'Frequency', 'Monetary', 'Length']:
    boxplot_clustering(X='Cluster_RFML', Y=h, dataset=rfm_l, title=name+", Cluster="+str(bestClus)+', '+h)
plot_3D_clustering(rfm_l['Recency'],
                   rfm_l['Frequency'],
                   rfm_l['Monetary'],
                   minmax_scale(rfm_l['Length'],feature_range=(0, 10)),
                   assignments[bestClusIndex],
                   name+", Cluster="+str(bestClus)+', Marker size : Length')
plot_2D_rfml_heatmap(rfm_l,
                     assignments[bestClusIndex],
                     2,
                     name+", Cluster="+str(bestClus))

#RFM by Variable Group with OneHot,Sum
name='MeanShift, RFM by Group with OneHot'
bestClusIndex,bestClus,quantileList,clusterList,assignments,meanShift = MeanShiftClustering(X=rfm_grp_normalized, name=name)
rfm_grp['Cluster_RFM_Group']=assignments[bestClusIndex]
rfm_grp.to_excel('rfm_grp_ms_'+ str(bestClus) +'.xlsx',index=False)
for h in ['Recency', 'Frequency', 'Monetary']:
    boxplot_clustering(X='Cluster_RFM_Group', Y=h, dataset=rfm_grp, title=name+", Cluster="+str(bestClus)+', '+h)
plot_3D_clustering(rfm_grp['Recency'],
                   rfm_grp['Frequency'],
                   rfm_grp['Monetary'],
                   2,
                   assignments[bestClusIndex],
                   name+", Cluster="+str(bestClus))
for g in group_colm :
    plot_3D_clustering(rfm_grp['Recency'],
                       rfm_grp['Frequency'],
                       rfm_grp['Monetary'],
                       minmax_scale(rfm_grp[g],feature_range=(0, 10)),
                       assignments[bestClusIndex],
                       name+", Cluster="+str(bestClus)+', Marker size : '+g)
plot_2D_rfm_heatmap(rfm_grp,
                    assignments[bestClusIndex],
                    2,
                    name+", Cluster="+str(bestClus))
for g in group_colm :
    plot_2D_rfm_heatmap(rfm_grp,
                        assignments[bestClusIndex],
                        minmax_scale(rfm_grp[g],feature_range=(0, 10)),
                        name+", Cluster="+str(bestClus),
                        ', Marker size : '+g)

#RFM-L by Variable Group with OneHot,Sum
name='MeanShift, RFM-L by Group with OneHot'
bestClusIndex,bestClus,quantileList,clusterList,assignments,meanShift = MeanShiftClustering(X=rfml_grp_normalized, name=name)
rfml_grp['Cluster_RFML_Group']=assignments[bestClusIndex]
rfml_grp.to_excel('rfml_grp_ms_'+ str(bestClus) +'.xlsx',index=False)
for h in ['Recency', 'Frequency', 'Monetary', 'Length']:
    boxplot_clustering(X='Cluster_RFML_Group', Y=h, dataset=rfml_grp, title=name+", Cluster="+str(bestClus)+', '+h)
plot_3D_clustering(rfml_grp['Recency'],
                   rfml_grp['Frequency'],
                   rfml_grp['Monetary'],
                   minmax_scale(rfml_grp['Length'],feature_range=(0, 10)),
                   assignments[bestClusIndex],
                   name+", Cluster="+str(bestClus)+', Marker size : Length')
for g in group_colm :
    plot_3D_clustering(rfml_grp['Recency'],
                       rfml_grp['Frequency'],
                       rfml_grp['Monetary'],
                       minmax_scale(rfml_grp[g],feature_range=(0, 10)),
                       assignments[bestClusIndex],
                       name+", Cluster="+str(bestClus)+', Marker size : '+g)
plot_2D_rfml_heatmap(rfml_grp,
                    assignments[range_n_clusters.index(bestClus)],
                    2,
                    name+", Cluster="+str(bestClus))
for g in group_colm :
    plot_2D_rfml_heatmap(rfml_grp,
                        assignments[bestClusIndex],
                        minmax_scale(rfml_grp[g],feature_range=(0, 10)),
                        name+", Cluster="+str(bestClus),
                        ', Marker size : '+g)


'''
*****************************************
Spectral Clustering
*****************************************
'''
#Spectral Clustering
#RFM
name='Spectral, RFM'
bestClusIndex,bestClus,clusters,clusterList,assignments,Spectral = SpectralClustering_(X=rfm_normalized, name=name)

#RFM - L
name='Spectral, RFM-L'
bestClusIndex,bestClus,clusters,clusterList,assignments,Spectral = SpectralClustering_(X=rfm_l_normalized, name=name)

#RFM by Variable Group with OneHot,Sum
name='Spectral, RFM by Group with OneHot'
bestClusIndex,bestClus,clusters,clusterList,assignments,Spectral = SpectralClustering_(X=rfm_grp_normalized, name=name)

#RFM-L by Variable Group with OneHot,Sum
name='Spectral, RFM-L by Group with OneHot'
bestClusIndex,bestClus,clusters,clusterList,assignments,Spectral = SpectralClustering_(X=rfml_grp_normalized, name=name)



'''
*****************************************
Agglomerative Clustering
*****************************************
'''
#Agglomerative Clustering
#RFM
name='Agglomerative, RFM'
bestClusIndex,bestClus,linkage,clusterList,assignments,Agglomerative = AgglomerativeClustering_(X=rfm_normalized, name=name)
rfm['Cluster_RFM']=assignments[bestClusIndex]
rfm.to_excel('rfm_agl_'+ str(bestClus) +'.xlsx',index=False)
for h in ['Recency', 'Frequency', 'Monetary']:
    boxplot_clustering(X='Cluster_RFM', Y=h, dataset=rfm, title=name+", Cluster="+str(bestClus)+', '+h)
plot_3D_clustering(rfm['Recency'],
                   rfm['Frequency'],
                   rfm['Monetary'],
                   2,
                   assignments[bestClusIndex],
                   name+", Cluster="+str(bestClus))
plot_2D_rfm_heatmap(rfm,
                    assignments[bestClusIndex],
                    2,
                    name+", Cluster="+str(bestClus))

#RFM - L
name='Agglomerative, RFM-L'
bestClusIndex,bestClus,linkage,clusterList,assignments,Agglomerative = AgglomerativeClustering_(X=rfm_l_normalized, name=name)
rfm_l['Cluster_RFML']=assignments[bestClusIndex]
rfm_l.to_excel('rfml_agl_'+ str(bestClus) +'.xlsx',index=False)
for h in ['Recency', 'Frequency', 'Monetary', 'Length']:
    boxplot_clustering(X='Cluster_RFML', Y=h, dataset=rfm_l, title=name+", Cluster="+str(bestClus)+', '+h)
plot_3D_clustering(rfm_l['Recency'],
                   rfm_l['Frequency'],
                   rfm_l['Monetary'],
                   minmax_scale(rfm_l['Length'],feature_range=(0, 10)),
                   assignments[bestClusIndex],
                   name+", Cluster="+str(bestClus)+', Marker size : Length')
plot_2D_rfml_heatmap(rfm_l,
                     assignments[bestClusIndex],
                     2,
                     name+", Cluster="+str(bestClus))

#RFM by Variable Group with OneHot,Sum
name='Agglomerative, RFM by Group with OneHot'
bestClusIndex,bestClus,linkage,clusterList,assignments,Agglomerative = AgglomerativeClustering_(X=rfm_grp_normalized, name=name)
rfm_grp['Cluster_RFM_Group']=assignments[bestClusIndex]
rfm_grp.to_excel('rfm_grp_agl_'+ str(bestClus) +'.xlsx',index=False)
for h in ['Recency', 'Frequency', 'Monetary']:
    boxplot_clustering(X='Cluster_RFM_Group', Y=h, dataset=rfm_grp, title=name+", Cluster="+str(bestClus)+', '+h)
plot_3D_clustering(rfm_grp['Recency'],
                   rfm_grp['Frequency'],
                   rfm_grp['Monetary'],
                   2,
                   assignments[bestClusIndex],
                   name+", Cluster="+str(bestClus))
for g in group_colm :
    plot_3D_clustering(rfm_grp['Recency'],
                       rfm_grp['Frequency'],
                       rfm_grp['Monetary'],
                       minmax_scale(rfm_grp[g],feature_range=(0, 10)),
                       assignments[bestClusIndex],
                       name+", Cluster="+str(bestClus)+', Marker size : '+g)
plot_2D_rfm_heatmap(rfm_grp,
                    assignments[bestClusIndex],
                    2,
                    name+", Cluster="+str(bestClus))
for g in group_colm :
    plot_2D_rfm_heatmap(rfm_grp,
                        assignments[bestClusIndex],
                        minmax_scale(rfm_grp[g],feature_range=(0, 10)),
                        name+", Cluster="+str(bestClus),
                        ', Marker size : '+g)

#RFM-L by Variable Group with OneHot,Sum
name='Agglomerative, RFM-L by Group with OneHot'
bestClusIndex,bestClus,linkage,clusterList,assignments,Agglomerative = AgglomerativeClustering_(X=rfml_grp_normalized, name=name)
rfml_grp['Cluster_RFML_Group']=assignments[bestClusIndex]
rfml_grp.to_excel('rfml_grp_agl_'+ str(bestClus) +'.xlsx',index=False)
for h in ['Recency', 'Frequency', 'Monetary', 'Length']:
    boxplot_clustering(X='Cluster_RFML_Group', Y=h, dataset=rfml_grp, title=name+", Cluster="+str(bestClus)+', '+h)
plot_3D_clustering(rfml_grp['Recency'],
                   rfml_grp['Frequency'],
                   rfml_grp['Monetary'],
                   minmax_scale(rfml_grp['Length'],feature_range=(0, 10)),
                   assignments[bestClusIndex],
                   name+", Cluster="+str(bestClus)+', Marker size : Length')
for g in group_colm :
    plot_3D_clustering(rfml_grp['Recency'],
                       rfml_grp['Frequency'],
                       rfml_grp['Monetary'],
                       minmax_scale(rfml_grp[g],feature_range=(0, 10)),
                       assignments[bestClusIndex],
                       name+", Cluster="+str(bestClus)+', Marker size : '+g)
plot_2D_rfml_heatmap(rfml_grp,
                    assignments[range_n_clusters.index(bestClus)],
                    2,
                    name+", Cluster="+str(bestClus))
for g in group_colm :
    plot_2D_rfml_heatmap(rfml_grp,
                        assignments[bestClusIndex],
                        minmax_scale(rfml_grp[g],feature_range=(0, 10)),
                        name+", Cluster="+str(bestClus),
                        ', Marker size : '+g)


'''
*****************************************
GaussianMixture Clustering
*****************************************
'''
#GaussianMixture Clustering
#RFM
name='GaussianMixture, RFM'
bestClusIndex,bestClus,n_component,clusterList,assignments,gaussianMixture = GaussianMixtureClustering(X=rfm_normalized, name=name)

#RFM - L
name='GaussianMixture, RFM-L'
bestClusIndex,bestClus,n_component,clusterList,assignments,gaussianMixture = GaussianMixtureClustering(X=rfm_l_normalized, name=name)

#RFM by Variable Group with OneHot,Sum
name='GaussianMixture, RFM by Group with OneHot'
bestClusIndex,bestClus,n_component,clusterList,assignments,gaussianMixture = GaussianMixtureClustering(X=rfm_grp_normalized, name=name)

#RFM-L by Variable Group with OneHot,Sum
name='GaussianMixture, RFM-L by Group with OneHot'
bestClusIndex,bestClus,n_component,clusterList,assignments,gaussianMixture = GaussianMixtureClustering(X=rfml_grp_normalized, name=name)




