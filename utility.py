# ##############################################################################
# Usage: from utility import *
# ##############################################################################
# 20210304, In Kyu Lee
# Desc: Data science utilities
# ############################################################################## 
# Categories: (needs to be separated later...)
#  - Preprocessing
#  - Statistics
#  - Plot
#  - Computer Vision
#  - Machine Learning
# ##############################################################################


import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from itertools import chain
from sklearn.cluster import KMeans
import scipy
import seaborn as sns
sns.set_theme(style="white")

## Preprocessing ##

def RemoveBlankCol(data):
    # Input: Dataframe
    # Ouput: Blank Columns removed dataframe
    # If the column is empty, the column will be removed.
    
    unnamed_list = [x for x in data.columns if 'Unnamed' in x]
    return data.drop(columns=unnamed_list)

def RemoveBlankRow(data):
    # Input: Dataframe
    # Ouput: Blank Columns removed dataframe
    # If the row is empty, the row will be removed.
    
    return data.dropna(how='all')

def RemoveBlank(data):
    # Input: Dataframe
    # Ouput: Blank Columns removed dataframe
    # RemoveBlankCol + RemoveBlankRow
    
    data = RemoveBlankCol(data)
    return RemoveBlankRow(data)

def Dropdupcol(data):
    # Input: pandas dataframe
    # Output: duplicated columns dropped dataframe
    print('Original Shape: ', data.shape, '\n')
    
    dup_cols = [x  for x in data.columns if x[-2:]=='.1']
    print("Duplicated columns: ", dup_cols, '\n')
    data = data.drop(columns=dup_cols)
    print('After dropping: ', data.shape)
    return data

def swap_col(data):
    # Input: pandas dataframe
    # Output: column changed df
    # move last column to the first column
    
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data = data[cols]
    return data
    
## Stats ##

def p_corr(x):
    # Input: X data
    # Ouput: Pearson Correlations
    return x.corr(method='pearson')

def t_value(r,n):
    # Input:
    #       - r: Pearson Correlation
    #       - n: Number of data points
    # Output:
    #       - t: t-score
    # Calculate t-score from Pearson correlation
    t = r*( (n-2)**0.5 ) * (1-r**2)**(-0.5)
    return t

def p_value(t,n):
    # Input:
    #       - t: t_score
    #       - n: Number of data points
    # Output:
    #       - p: p-value
    # Calculate p-value from t-score
    p = scipy.stats.t.sf(abs(t), df=(n-2))*2
    return p
    
## Plot ##

def subplot_square(data,size):
    # plot size x size images
    fig, ax = plt.subplots(size,size)
    
    for i in range(size):
        for j in range(size):
            ax[i,j].imshow(data[i*size+j],cmap='gray')
            ax[i,j].set_axis_off()
            
def ShowHeatMap(data,threshold=0, annotation=True):
    # Input: data matrix
    # annotation: True if you want to show value on the figure
    data = data[np.absolute(data)>threshold]
    data = data.dropna(axis=1, how='all')
    data = data.dropna(axis=0, how='all')
    plt.figure(figsize=(40,40))
    sns.heatmap(data,annot=annotation)
    return data 

def show_cor_heatmap(df_cor, n, x_col, y_col):
    df_cor = df_cor.drop(columns = x_col)
    df_cor = df_cor.drop(index = y_col)
    dim = df_cor.shape
    # Calculate P-value
    Ps = np.zeros(shape=[dim[0],dim[1]])
    for i in range(dim[0]):
        for j in range(dim[1]):
            r = df_cor.iloc[i,j]
            t = t_value(r,n)
            p = p_value(t,n)
            Ps[i,j] = p
    Ps = pd.DataFrame(Ps)
    Ps.columns = df_cor.columns
    Ps.index = df_cor.index
    
    data_p = df_cor[Ps<0.05]
    data_p = data_p.dropna(axis=1, how='all')
    data_p = data_p.dropna(axis=0, how='all')
    plt.figure(figsize=(12,12))
    sns.set(font_scale=1.5)
    sns.heatmap(data_p,annot=True,linewidths=.5,annot_kws={'size':15},cmap='seismic')
    plt.title('Correlation with p-value<0.05',fontsize=25)
    return data_p


## CV ##

def get_differential_filter():
    # Return differential filters in x & y directions
    filter_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    filter_y=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    return filter_x, filter_y

def filter_image(im, filter):
    # Apply filter to the image
    im_filtered = np.zeros((np.size(im,0),np.size(im,1)))
    im_pad = np.pad(im,((1,1),(1,1)), 'constant')

    tracker_i=-1
    for i in range(np.size(im_filtered,0)):
        tracker_i+=1
        tracker_j=-1
        for j in range(np.size(im_filtered,1)):
            tracker_j+=1
            v=0
            for k in range(np.size(filter,0)):
                for l in range(np.size(filter,1)):
                    v += filter[k][l] * im_pad[k+tracker_i][l+tracker_j]
            im_filtered[i,j] = v
    return im_filtered

## ML ##

def PCA_2(data):
# If there is no label, set labels = None
    data = StandardScaler().fit_transform(data)
    pca = PCA(n_components = 2)
    principalComponents = pca.fit_transform(data)
    weights = pca.components_
    
    PCs = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

    return PCs,weights
    
def K_means(x,y,n,plot,xlabel=None,ylabel=None):
    # n: number of clusters
    kmeans = KMeans(n_clusters = n)
    df = pd.DataFrame({
        'x':x,
        'y':y
    })
    kmeans.fit(df)
    labels = kmeans.predict(df)
    centroids = kmeans.cluster_centers_
    
    # Sorting from X_min centroid to X_max centroid
    new_i = centroids[:,0].argsort()
    if n==4:
        if centroids[new_i[1],1] < centroids[new_i[2],1]:
            temp = new_i[1]
            new_i[1] = new_i[2]
            new_i[2] = temp

    new_centroids = centroids[new_i]
    new_labels = []
    for label in labels:
        for i in range(len(new_centroids)):
            if label == i:
                new_labels.append(np.argwhere(new_i==i)[0,0])
                
    new_labels = np.array(new_labels)
    new_labels = np.int32(new_labels)
    
    if (plot):
        colmap = {1: 'r', 2: 'g', 3: 'b', 4:'y', 5:'k'}
        fig = plt.figure(figsize=(5, 5))
        colors = list(map(lambda x: colmap[x+1], new_labels))
        plt.scatter(df['x'], df['y'], color=colors, alpha=0.6, edgecolor=colors,s=15)

        # Plotting the centroids
        for idx, centroid in enumerate(new_centroids):
            plt.scatter(*centroid, color=colmap[idx+1],marker='^',s=120)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
    new_labels += 1
    return new_labels, new_centroids

def K_means_seg(img,n):
    # img: 3 channel image
    # n: # of clusters
    
    # Conver to 2D arrays
    img_resized = np.resize(img,(img.shape[0]*img.shape[1],3))
    
    # K_Means
    labels, centroids = K_means(img_resized[:,0],
                                img_resized[:,1],
                                img_resized[:,2],
                                n=n, plot=False)
    
    # Color segmentation
    centroids = np.uint8(centroids)
    img_seg = centroids[labels]
    img_seg = np.resize(img_seg, (img.shape[0],img.shape[1],3))
    
    mean = np.mean(centroids,axis=1)
    bg_label = mean.argmin()
    area_t = (labels!=bg_label).sum()
    for i in range(n):
        if i==bg_label:
            continue
        area = (i==labels).sum()
        print("Color:",centroids[i])
        print("Area occupied: ",area/area_t)
    
    return img_seg