#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 17:23:10 2017

@author: robertmarsland
"""
import pandas as pd
import numpy as np
from community_simulator.usertools import MakeConsumerDynamics,MakeResourceDynamics,MakeMatrices,MakeInitialState
from community_simulator import Community
import pickle

#folder = '/project/biophys/microbial_crm/data/'
folder= '../data/'

n_samples = 300
R0_food = 1000

mp = {'sampling':'Binary', #Sampling method
    'SA': np.ones(6)*800, #Number of species in each family
    'MA': np.ones(6)*50, #Number of resources of each type
    'Sgen': 20, #Number of generalist species
    'muc': 10, #Mean sum of consumption rates in Gaussian model
    'q': 0.9, #Preference strength (0 for generalist and 1 for specialist)
    'c0':0, #Background consumption rate in binary model
    'c1':1., #Specific consumption rate in binary model
    'fs':0.45, #Fraction of secretion flux with same resource type
    'fw':0.45, #Fraction of secretion flux to 'waste' resource
    'sparsity':0.3, #Variability in secretion fluxes among resources (must be less than 1)
    'regulation':'independent',
    'supply':'external',
    'response':'type I',
    'waste_type':5
    }
#Construct dynamics
def dNdt(N,R,params):
    return MakeConsumerDynamics(mp)(N,R,params)
def dRdt(N,R,params):
    return MakeResourceDynamics(mp)(N,R,params)
dynamics = [dNdt,dRdt]
#Construct matrices
c,D = MakeMatrices(mp)
#Make dictionaries to hold data
N = {}
metadata = {}

## Set up shared parameters
HMP_protocol = {'R0_food':R0_food, #unperturbed fixed point for supplied food
                'n_wells':3*n_samples, #Number of independent wells
                'S':2500, #Number of species per well
                'food':0 #index of food source
                }
HMP_protocol.update(mp)
#Make initial state
N0,R0 = MakeInitialState(HMP_protocol)

################Two external resources####################
R0 = np.zeros(np.shape(R0))
alpha = np.random.rand(n_samples)
for k in range(3):
    R0[2*k*50,k*n_samples:(k+1)*n_samples] = alpha*R0_food
    R0[(2*k+1)*50,k*n_samples:(k+1)*n_samples] = (1-alpha)*R0_food
R0 = pd.DataFrame(R0,index=D.index,columns=N0.keys())
init_state=[N0,R0]
#Make parameter list
m = 1+0.01*np.random.randn(len(c))
params=[{'w':1,
        'g':1,
        'l':0.8,
        'R0':R0.values[:,k],
        'r':1.,
        'tau':1
        } for k in range(len(N0.T))]
for k in range(len(params)):
    params[k]['c'] = c
    params[k]['D'] = D
    params[k]['m'] = m
HMP = Community(init_state,dynamics,params)
HMP.metadata = pd.DataFrame(['Site 1']*n_samples+['Site 2']*n_samples+['Site 3']*n_samples,
                            index=N0.T.index,columns=['Environment'])
HMP.SteadyState(plot=False,tol=1e-3,verbose=False)
with open(folder+'HMP_env_two.dat','wb') as f:
    pickle.dump([HMP.N,HMP.R,params[0],R0,HMP.metadata],f)
N['Two resource HMP'] = HMP.N
metadata['Two resource HMP'] = HMP.metadata

#############All external resources####################
R0 = np.zeros(np.shape(R0))
for k in range(3):
    R0_temp = np.random.rand(mp['MA'][2*k]+mp['MA'][2*k+1],n_samples)
    R0_temp = (R0_temp/R0_temp.sum(axis=0))*R0_food
    R0[mp['MA'][:2*k].sum():mp['MA'][:2*(k+1)].sum(),k*n_samples:(k+1)*n_samples] = R0_temp
R0 = pd.DataFrame(R0,index=D.index,columns=N0.keys())
init_state=[N0,R0]
#Make parameter list
m = 1+0.01*np.random.randn(len(c))
params=[{'w':1,
        'g':1,
        'l':0.8,
        'R0':R0.values[:,k],
        'r':1.,
        'tau':1
        } for k in range(len(N0.T))]
for k in range(len(params)):
    params[k]['c'] = c
    params[k]['D'] = D
    params[k]['m'] = m
HMP = Community(init_state,dynamics,params)
HMP.metadata = pd.DataFrame(['Site 1']*n_samples+['Site 2']*n_samples+['Site 3']*n_samples,
                            index=N0.T.index,columns=['Environment'])
HMP.SteadyState(plot=False,tol=1e-3,verbose=False)
with open(folder+'HMP_env_all.dat','wb') as f:
    pickle.dump([HMP.N,HMP.R,params[0],R0,HMP.metadata],f)
N['All resource HMP'] = HMP.N
metadata['All resource HMP'] = HMP.metadata

for exp in N.keys():
    N[exp].to_csv(folder+'_'.join(['N']+exp.split(' '))+'.csv')
    metadata[exp].to_csv(folder+'_'.join(['m']+exp.split(' '))+'.csv')
