import pandas as pd
import numpy as np
from community_simulator.usertools import BinaryRandomMatrix,MakeConsumerDynamics,MakeResourceDynamics,MakeMatrices,MakeInitialState
from community_simulator import Community
import pickle

#folder = '/project/biophys/microbial_crm/data/'
folder= '../data/'
R0_food = 1000
alpha = 0.5 #mixing ratio

mp = {'sampling':'Binary', #Sampling method
    'SA': np.ones(6)*800, #Number of species in each family
    'MA': np.ones(6)*50, #Number of resources of each type
    'Sgen': 200, #Number of generalist species
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
                'n_wells':6, #Number of independent wells
                'S':5000, #Number of species per well
                'food':0 #index of food source
                }
HMP_protocol.update(mp)
#Make initial state
N0,R0 = MakeInitialState(HMP_protocol)

################With modularity####################
exp = 'Modular assembly'
R0 = np.zeros(np.shape(R0))
R0[0,0] = R0_food
R0[mp['MA'][:1].sum(),1] = R0_food
R0[:,2] = alpha*(R0[:,0]+R0[:,1])
R0[mp['MA'][:2].sum(),3] = R0_food
R0[mp['MA'][:3].sum(),4] = R0_food
R0[:,5] = alpha*(R0[:,3]+R0[:,4])
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
HMP.metadata = pd.DataFrame(np.asarray([0,1,alpha,0,1,alpha]),index=N0.T.index,columns=['alpha'])
HMP.metadata['Environment'] = ['Site 1']*3 + ['Site 2']*3
HMP.SteadyState(plot=False,tol=1e-3,verbose=False)
with open(folder+'_'.join(['comm']+exp.split(' '))+'S'+str(HMP_protocol['S'])+'.dat','wb') as f:
    pickle.dump([HMP.N,HMP.R,params[0],R0,HMP.metadata],f)
HMP.N.to_csv(folder+'_'.join(['N']+exp.split(' '))+'S'+str(HMP_protocol['S'])+'.csv')
HMP.metadata.to_csv(folder+'_'.join(['m']+exp.split(' '))+'S'+str(HMP_protocol['S'])+'.csv')

################No modularity####################
exp = 'No modularity'
HMP_protocol.update({'SA': 6*800+200, #Number of species in each family
                    'MA': 6*50, #Number of resources of each type
                    'Sgen': 0,
                    'waste_type':0,
                    'n_wells':3
                    })
c,D = MakeMatrices(HMP_protocol)
N0,R0 = MakeInitialState(HMP_protocol)
R0 = np.zeros(np.shape(R0))
R0[0,0] = R0_food
R0[1,1] = R0_food
R0[:,2] = alpha*(R0[:,0]+R0[:,1])
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
HMP.metadata = pd.DataFrame(np.asarray([0,1,alpha]),index=N0.T.index,columns=['alpha'])
HMP.metadata['Environment'] = ['Site 1']*3
HMP.SteadyState(plot=False,tol=1e-3,verbose=False)
with open(folder+'_'.join(['comm']+exp.split(' '))+'S'+str(HMP_protocol['S'])+'.dat','wb') as f:
    pickle.dump([HMP.N,HMP.R,params[0],R0,HMP.metadata],f)
HMP.N.to_csv(folder+'_'.join(['N']+exp.split(' '))+'S'+str(HMP_protocol['S'])+'.csv')
HMP.metadata.to_csv(folder+'_'.join(['m']+exp.split(' '))+'S'+str(HMP_protocol['S'])+'.csv')