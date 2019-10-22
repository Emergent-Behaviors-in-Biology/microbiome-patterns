import pandas as pd
import numpy as np
from community_simulator.usertools import MakeConsumerDynamics,MakeResourceDynamics
from community_simulator import Community
import pickle

#Choose data to analyze
folder = '../data/'
S = 2500
exp = 'Two resource HMP'
env = 'Site 1'
thresh = 1e-4

#Define dissimilarity and overlap
def D(x,y,thresh=0):
    S = np.where(np.logical_and(x>thresh,y>thresh))[0]
    
    xhat = x[S]/x[S].sum()
    yhat = y[S]/y[S].sum()
    m = (xhat+yhat)/2
    
    return np.sqrt(0.5*(xhat*np.log(xhat/m)+yhat*np.log(yhat/m)).sum())
def O(x,y,thresh=0):
    S = np.where(np.logical_and(x>thresh,y>thresh))[0]
    
    xtilde = x/x.sum()
    ytilde = y/y.sum()
    
    return 0.5*(xtilde[S]+ytilde[S]).sum()

#Load data
with open(folder+'_'.join(['comm']+exp.split(' '))+'S'+str(S)+'.dat','rb') as f:
    N,R,params,R0,metadata = pickle.load(f)
#Remove failed runs
metadata = metadata[np.isnan(N).sum()==0]
R = R.T[np.isnan(N).sum()==0].T
R0 = R0.T[np.isnan(N).sum()==0].T
N = N.T[np.isnan(N).sum()==0].T

#Select data and set threshold
wells = metadata.index[metadata['Environment']==env]
N = N[wells]
thresh = 1e-4

#Set up variables for computing Lotka-Volterra parameters
M = len(R)
Dmat = pd.DataFrame(params['D'],index=R.index,columns=R.index)
c = pd.DataFrame(params['c'],index=N.index,columns=R.index)
Q = np.eye(M) - Dmat*params['l']

#Compute Lotka-Volterra parameters for all wells
alpha_list = []
K_list = []
for well in wells:
    A = np.eye(M) + Q*(c.T.dot(N[well]))
    Ainv = pd.DataFrame(np.linalg.inv(A),index=A.index,columns=A.keys())
    alpha = ((c*(1-params['l'])).dot(Ainv).dot(Q).dot((c*R[well]).T))
    alpha_list.append((alpha.T/np.diag(alpha)).T.values)
    K_list.append((alpha.dot(N[well])/np.diag(alpha)).values)
    
#Subsample community pairs
pairs = []
for run1 in range(len(N.T)):
    for run2 in range(run1+1,len(N.T)):
        pairs.append((run1,run2))
subsample = np.random.choice(np.arange(len(pairs),dtype=int),size=10000)
pairs = np.asarray(pairs)[subsample]
#Compute dissimilarity and overlap
Dlist = []
Olist = []
alpha_diff = []
K_diff = []
pair_save = []
for item in range(len(pairs)):
    if np.remainder(item,1000) == 0:
        print(item)
    run1,run2 = pairs[item]
    well1 = wells[run1]
    well2 = wells[run2]
    x = N[well1].values
    y = N[well2].values
    if np.max(x) > thresh and np.max(y) > thresh and D(x,y,thresh=thresh) > thresh:
        pair_save.append(pairs[item])
        Dlist.append(D(x,y,thresh=thresh))
        Olist.append(O(x,y,thresh=thresh))
            
        shared = ((N[well1]>thresh)&(N[well2]>thresh)).values
            
        alpha1 = alpha_list[run1][shared,:]
        alpha1 = alpha1[:,shared]
        alpha2 = alpha_list[run2][shared,:]
        alpha2 = alpha2[:,shared]
            
        K1 = K_list[run1][shared]
        K2 = K_list[run2][shared]
            
        alpha_diff.append(np.sqrt(((alpha1-alpha2)**2).mean()/(0.5*(alpha1+alpha2).mean())**2))
        K_diff.append(np.sqrt(((K1-K2)**2).mean()/(0.5*(K1+K2).mean())**2))
#Compute dissimilarity and overlap for randomized data
Dlist_null = []
Olist_null = []
null = N.values.copy()
for k in range(len(null)):
    Inonzeros = np.where(null[k,:]>0)[0]
    null[k,Inonzeros] = null[k,np.random.permutation(Inonzeros)]
for item in range(len(pairs)):
    if np.remainder(item,1000) == 0:
        print(item)
    run1,run2 = pairs[item]
    x = null[:,run1]
    y = null[:,run2]
    if np.max(x) > thresh and np.max(y) > thresh and D(x,y,thresh=thresh) > thresh:
        Dlist_null.append(D(x,y,thresh=thresh))
        Olist_null.append(O(x,y,thresh=thresh))
#Make into numpy arrays
Olist = np.asarray(Olist)
Dlist = np.asarray(Dlist)
Olist_null = np.asarray(Olist_null)
Dlist_null = np.asarray(Dlist_null)

#Save results
with open(folder+'DOC_same.dat','wb') as f:
    pickle.dump([pair_save,Olist,Dlist,Olist_null,Dlist_null],f)
with open(folder+'LV_params.dat','wb') as f:
    pickle.dump([alpha_diff,K_diff],f)

#################################
###Run for distinct body sites###
#Load data
with open(folder+'_'.join(['comm']+exp.split(' '))+'S'+str(S)+'.dat','rb') as f:
    N,R,params,R0,metadata = pickle.load(f)
#Remove failed runs
metadata = metadata[np.isnan(N).sum()==0]
R = R.T[np.isnan(N).sum()==0].T
R0 = R0.T[np.isnan(N).sum()==0].T
N = N.T[np.isnan(N).sum()==0].T
wells = N.T.index

#Subsample pairs and compute dissimilarity and overlap
pairs = []
for run1 in range(len(N.T)):
    for run2 in range(run1+1,len(N.T)):
        if metadata['Environment'].loc[wells[run1]] != metadata['Environment'].loc[wells[run2]]:
            pairs.append((run1,run2))
subsample = np.random.choice(np.arange(len(pairs),dtype=int),size=10000)
pairs = np.asarray(pairs)[subsample]
pair_save = []
Dlist = []
Olist = []
for item in range(len(pairs)):
    if np.remainder(item,1000) == 0:
        print(item)
    run1,run2 = pairs[item]
    well1 = wells[run1]
    well2 = wells[run2]
    x = N[well1].values
    y = N[well2].values
    if np.max(x) > thresh and np.max(y) > thresh and D(x,y,thresh=thresh) > thresh:
        Dlist.append(D(x,y,thresh=thresh))
        Olist.append(O(x,y,thresh=thresh))
        pair_save.append(pairs[item])
#Compute dissimilarity and overlap for randomized data
Dlist_null = []
Olist_null = []
null = N.values.copy()
for k in range(len(null)):
    Inonzeros = np.where(null[k,:]>0)[0]
    null[k,Inonzeros] = null[k,np.random.permutation(Inonzeros)]
for item in range(len(pairs)):
    if np.remainder(item,1000) == 0:
        print(item)
    run1,run2 = pairs[item]
    well1 = wells[run1]
    well2 = wells[run2]
    x = null[:,run1]
    y = null[:,run2]
    if np.max(x) > thresh and np.max(y) > thresh and D(x,y,thresh=thresh) > thresh:
        Dlist_null.append(D(x,y,thresh=thresh))
        Olist_null.append(O(x,y,thresh=thresh))
#Make into numpy arrays
Olist = np.asarray(Olist)
Dlist = np.asarray(Dlist)
Olist_null = np.asarray(Olist_null)
Dlist_null = np.asarray(Dlist_null)

#Save results
with open(folder+'DOC_diff.dat','wb') as f:
    pickle.dump([pair_save,Olist,Dlist,Olist_null,Dlist_null],f)