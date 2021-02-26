# -*- coding: utf-8 -*-
"""
@author: Jeremy Yeaton

Script implementing SRN simulations for sequence learning in baboons

"""


'''
~5 values of nu
~5 values of mom
~5 values of NH

training data from each of the 20 baboons

simulations for exp 1 -- each position in the sequence
simulations for exp 2 -- variable position & length 5

bayesian stats after for learning rate?
'''

import os, pickle
import numpy as np#, matplotlib.pyplot as plt, pandas as pd
os.chdir('C:/Users/LPC/Documents/JDY/NADs')
import SRN

babNames = ["ANGELE","ARIELLE","ARTICHO","ATMOSPHERE","DORA","DREAM",\
            "EWINE","FANA","FELIPE","FEYA","FLUTE","HARLEM","HERMINE","KALI",\
                "LIPS","LOME","MAKO","MALI","MUSE","NEKKE","VIOLETTE",\
                    "PIPO","BOBO","PETOULETTE"] # "CAUET","BRIGITTE", "0","B06"
nBabs = len(babNames)


## Define the structure of the network ####
NI  = 11                            # number of input units
NH  = 15                            # number of hidden units
NO  = 11                            # number of output units

nu  = 0.05                          # Learning rate parameter; can vary between 0 and 1
mom = 1e-10                         # Momentum parameter



momVals = [1e0,5e-1,1e-1,5e-2,1e-2]
nuVals = [1e0,5e-1,1e-1,5e-2,1e-2]
NHvals = [17,15,13,11,9]

# babNames = [babNames[0]]
for bab in babNames:
    print('Starting ' + bab)
    allModels = []
    for nu in nuVals:
        for mom in momVals:
            for NH in NHvals:
                model = SRN.Model(NI,NO,NH)
                baboon_name = bab
                model.load_trials('exp1/input/master/' + baboon_name +'.txt')
                model.make_sequence(start = 0, nTrials = 2000)
                # Draw nu and mom values from a gaussian about 
                model.nu = np.random.normal(loc=nu,scale = nu/3)
                model.mom = np.random.normal(loc=mom,scale = 0.1)
                model.run_simulation()
                # model.plot_error(plotIdxs = [])
            
                # model.testing(4,3)
                allModels.append(model)
    f = open('SRN_models/allModels_' + bab + '.pkl', "wb")
    pickle.dump(allModels, f)
    f.close()
    print(bab + ' done!')

#%% toy space

NH = 15
mom = .2
nu = .2
startVal = 0
nTrials = 2000
bab = 'ANGELE'

model = SRN.Model(NI,NO,NH)
baboon_name = bab
model.load_trials('exp1/input/master/' + baboon_name +'.txt')
model.make_sequence(start = startVal, nTrials = nTrials)
# Draw nu and mom values from a gaussian about target values
model.nu = np.random.normal(loc=nu,scale = nu/3)
model.mom = np.random.normal(loc=mom,scale = 0.1)
model.run_simulation()
#%%
model.plot_error(plotIdxs = [],start = 1500,nTrials = 500)

# model.testing(4,3)
#%%
f = open('allModels_' + bab + '.pkl', "rb")
allModels = pickle.load(f)
f.close()














#%%
import numpy as np, matplotlib.pyplot as plt
import os, pandas as pd
os.chdir('C:/Users/LPC/Documents/JDY/NADs')

## Define the structure of the network ####
NI  = 9                             # number of input units
NH  = 15                            # number of hidden units
NC  = 15                            # number of contextual units
NO  = 9                             # number of output units

nu  = 0.1                           # Learning rate parameter; can vary between 0 and 1
mom = 0.2                           # Momentum parameter


# Creation of the sequence of stimuli
# nRepetitions = 1000                 # Number of repetitions of the sequence in the stimuli
# sequence = [0,1,2,3,4,5] * nRepetitions

# # Laure's training data
# ls1 = [9,5,6,8,2,4,7,1,3]
# sequence = [i - 1 for i in ls1] * 1000


## Processing and Learning loop ####
for i in range(Nsweep-1):
    Input  = sequence[i]            # define the input
    Output = sequence[i + 1]        # define the selected output
    
    # Initialization of Unit activation
    IU = np.zeros((1,NI))           # Input units
    IU[0,Input] = 1                 # set active Input unit
    OU = np.zeros((1,NO))           # Output units
    HU = np.zeros((1,NH))           # Hidden units
    CU = HU                         # Context units
    
    EO = np.zeros((1,NO))           # Expected Output
    EO[0,Output] = 1                # set expected Output unit
    
    # Compute the net activation received by Hidden units from both Input & Context units
    HUnet = np.inner(IU, WIH.T) + np.inner(CU, WCH)
    HU = sigmoid(HUnet)             # Compute the activation of Hidden units through the sigmoid transfer function
    
    # Compute the net activation received by Output units from Hidden units
    OUnet = np.inner(HU,WHO.T)
    OU = sigmoid(OUnet)             # Compute the activation of Output units through the sigmoid transfer function
    
    # Compute the error between the actual output and the expected output
    ERROR[i] = np.sum(abs(EO - OU))
    
    # Backpropagation from Output to Hidden
    for j in range(NH):
        for k in range(NO):
            dWHO[j,k] = (EO[0,k] - OU[0,k]) * sig_deriv(HUnet[0,j])
    
    # Backpropagation from Hidden to Input
    for j in range(NI):
        for k in range(NH):
            temp = 0
            for l in range(NO):
                temp += dWHO[k,l] * WHO[k,l]
            dWIH[j,k] = sig_deriv(HUnet[0,k]) * temp
    
    # Backpropagation from Hidden to Context
    for j in range(NC):
        for k in range(NH):
            temp = 0
            for l in range(NO):
                temp += dWHO[k,l] * WHO[k,l]
            dWCH[j,k] = sig_deriv(HUnet[0,k]) * temp
    
    # Compute the new weights from Hidden to Output units
    for j in range(NH):
        for k in range(NO):
            momentumWHO[j,k] = mom * deltaWHO[j,k]
            deltaWHO[j,k] = nu * dWHO[j,k] * HU[0,j]
            WHO[j,k] += deltaWHO[j,k] + momentumWHO[j,k]
    
    # Compute the new weights from Input to Hidden units            
    for j in range(NI):
        for k in range(NH):
            momentumWIH[j,k] = mom * deltaWIH[j,k]
            deltaWIH[j,k] = nu * dWIH[j,k] * IU[0,j]
            WIH[j,k] += deltaWIH[j,k] + momentumWIH[j,k]
    
    # Compute the new weights from Context to Hidden units
    for j in range(NC):
        for k in range(NH):
            momentumWCH[j,k] = mom * deltaWCH[j,k]
            deltaWCH[j,k] = nu * dWCH[j,k] * CU[0,j]
            WCH[j,k] = WCH[j,k] + deltaWCH[j,k] + momentumWCH[j,k]
    
    # Copy of Hidden unit activity to Context units        
    CU = HU.copy()

## Visualization of the error rate ####
plt.figure()
plt.plot(range(len(sequence)),ERROR)

#%% Testing after learning
# stim = input('Enter a number (0 to 5): ')
# stim = int(stim)

plt.figure()
fig, axs = plt.subplots(3,3,sharey=True)
x,y = 0,0
for stim in range(NI):
    IU = np.zeros((1,NI))
    IU[0,stim] = 1
    HU = np.zeros((1,NH))
    OU = np.zeros((1,NO))
    
    # Compute activation from Input to Hidden
    for i in range(NI):
        for j in range(NH):
            HU[0,j] += np.multiply(IU[0,i],WIH[i,j])
    HU = sigmoid(HU)
    
    # Compute activation from Hidden to Output
    for i in range(NH):
        for j in range(NO):
            OU[0,j] += np.multiply(HU[0,i],WHO[i,j])
    OU = sigmoid(OU)
    
    print(x,y)
    # Plot Output
    axs[x,y].bar(range(NO),sum(OU))
    if y < 2:
        y += 1
    else:
        x += 1
        y = 0
    
