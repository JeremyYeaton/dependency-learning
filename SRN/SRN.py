# -*- coding: utf-8 -*-
"""
@author: Jeremy Yeaton

Script implementing a basic Simple Recurrent Network (SRN) model
Based on MATLAB code from Arnaud Rey

"""

import numpy as np

## Define functions ####
# Basic sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sig_deriv(x):
    return np.exp(-x) / (1 + 2 * np.exp(-2 * x))

class Model:
    import numpy as np
    def __init__(self,NI,NO,NH):
        NC = NH # Number of context units equals number of hidden units
        self.NI  = NI
        self.NO  = NO
        self.NH  = NH
        self.NC  = NH 
        
        ## Initialize weight matrices
        self.WIH = np.random.rand(NI, NH) - 0.5  # Connection weights from Input to Hidden units
        self.WHO = np.random.rand(NH, NO) - 0.5  # Connection weights from Hidden to Output units
        self.WCH = np.random.rand(NC, NH) - 0.5  # Connection weights from Context to Hidden units
        
        ## Matrices coding for derivative, delta, and momentum values for each set of connection weights ####
        # Input & Hidden
        self.dWIH        = np.zeros((NI, NH))
        self.deltaWIH    = np.zeros((NI, NH))
        self.momentumWIH = np.zeros((NI, NH))
        
        # Hidden & Output
        self.dWHO        = np.zeros((NH, NO))
        self.deltaWHO    = np.zeros((NH, NO))
        self.momentumWHO = np.zeros((NH, NO))
        
        # Context & Hidden
        self.dWCH        = np.zeros((NC, NH))
        self.deltaWCH    = np.zeros((NC, NH))
        self.momentumWCH = np.zeros((NC, NH)) 

    def load_trials(self,fileName,trial_len = 4):
        import pandas as pd
        # Import training data
        trlsAllCol = pd.read_csv(fileName,
                                 sep=',',
                                 names=['test','rwrd','nTch','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','block'])
        # Select touch index columns
        if trial_len == 5:
            trls_df = trlsAllCol[['t1','t2','t3','t4','t5','block']]
        else:
            trls_df = trlsAllCol[['t1','t2','t3','t4','block']]
        # Convert to list
        self.trial_len = trial_len
        self.trls = trls_df.values.tolist()
        # Get block labels
        # blockLabels = trlsAllCol[['block']]
        # self.blockLabs = blockLabels#.values.to_list()
    
    def make_sequence(self, start = 500,nTrials = 500, fixCross = True):
        trial_len = self.trial_len
        trls = self.trls
        sequence = []
        if fixCross:
            for i in range(start,start + nTrials):
                sequence.append(9) # starting cross
                for j in range(trial_len):
                    sequence.append(trls[i][j]-1)
                # sequence.append(10) # reward
                sequence.append(9)
            trial_len = trial_len + 2
        else:
            for i in range(start,start + nTrials):
                for j in range(trial_len):
                    sequence.append(trls[i][j]-1)
        self.sequence  = sequence
        # Number of iterations depending on the length of the stimulus set
        nSweeps        = len(sequence)
        self.nSweeps   = nSweeps
        self.nTrials   = int(nSweeps/trial_len)
        self.trial_len = trial_len
        self.block     = self.trls[start][-1]
        # Initialization of the array recording the error
        self.ERROR     = np.zeros((self.nTrials,trial_len))
    
    def backprop_OH(self):
        # Backpropagation from Output to Hidden
        for j in range(self.NH):
            for k in range(self.NO):
                self.dWHO[j,k] = (self.EO[0,k] - self.OU[0,k]) * sig_deriv(self.HUnet[0,j])
                
    def backprop_HI(self):
        # Backpropagation from Hidden to Input
        for j in range(self.NI):
            for k in range(self.NH):
                temp = 0
                for l in range(self.NO):
                    temp += self.dWHO[k,l] * self.WHO[k,l]
                self.dWIH[j,k] = sig_deriv(self.HUnet[0,k]) * temp
    
    def backprop_HC(self):
        # Backpropagation from Hidden to Context
        for j in range(self.NC):
            for k in range(self.NH):
                temp = 0
                for l in range(self.NO):
                    temp += self.dWHO[k,l] * self.WHO[k,l]
                self.dWCH[j,k] = sig_deriv(self.HUnet[0,k]) * temp
    
    def update_weights_HO(self):
        # Compute the new weights from Hidden to Output units
        for j in range(self.NH):
            for k in range(self.NO):
                self.momentumWHO[j,k] = self.mom * self.deltaWHO[j,k]
                self.deltaWHO[j,k] = self.nu * self.dWHO[j,k] * self.HU[0,j]
                self.WHO[j,k] += self.deltaWHO[j,k] + self.momentumWHO[j,k]
    
    def update_weights_IH(self):
        # Compute the new weights from Input to Hidden units            
        for j in range(self.NI):
            for k in range(self.NH):
                self.momentumWIH[j,k] = self.mom * self.deltaWIH[j,k]
                self.deltaWIH[j,k] = self.nu * self.dWIH[j,k] * self.IU[0,j]
                self.WIH[j,k] += self.deltaWIH[j,k] + self.momentumWIH[j,k]
    
    def update_weights_CH(self):
        # Compute the new weights from Context to Hidden units
        for j in range(self.NC):
            for k in range(self.NH):
                self.momentumWCH[j,k] = self.mom * self.deltaWCH[j,k]
                self.deltaWCH[j,k] = self.nu * self.dWCH[j,k] * self.CU[0,j]
                self.WCH[j,k] = self.WCH[j,k] + self.deltaWCH[j,k] + self.momentumWCH[j,k]
    
    def run_simulation(self):
        # Set indexes for error assignment
        x,y = 0,0
        ## Processing and Learning loop ####
        for i in range(self.nSweeps - 1):
            percent_done = (i/self.nSweeps) * 100
            if percent_done % 10 == 0:
                # print('Training ' + str(int(percent_done)) + '% done.')
                print(str(int(percent_done)) + '%.',end=' ')
            Input  = self.sequence[i]            # define the input
            Output = self.sequence[i + 1]        # define the selected output
            
            # Initialization of Unit activation
            IU = np.zeros((1,self.NI))           # Input units
            IU[0,Input] = 1                 # set active Input unit
            self.IU = IU
            OU = np.zeros((1,self.NO))           # Output units
            HU = np.zeros((1,self.NH))           # Hidden units
            self.CU = HU                         # Context units
            
            EO = np.zeros((1,self.NO))           # Expected Output
            EO[0,Output] = 1                # set expected Output unit
            self.EO = EO
            
            # Compute the net activation received by Hidden units from both Input & Context units
            HUnet = np.inner(IU, self.WIH.T) + np.inner(self.CU, self.WCH)
            self.HUnet = HUnet
            HU = sigmoid(HUnet)             # Compute the activation of Hidden units through the sigmoid transfer function
            self.HU = HU
            
            # Compute the net activation received by Output units from Hidden units
            OUnet = np.inner(HU,self.WHO.T)
            OU = sigmoid(OUnet)             # Compute the activation of Output units through the sigmoid transfer function
            self.OU = OU
            
            # Compute the error between the actual output and the expected output
            self.ERROR[x,y] = np.sum(abs(EO - OU))
            if y < self.trial_len - 1:
                y += 1
            else:
                y = 0
                x += 1
                
            # Backpropagation from Output to Hidden
            self.backprop_OH()
            # Backpropagation from Hidden to Input
            self.backprop_HI()
            # Backpropagation from Hidden to Context
            self.backprop_HC()
            # Compute the new weights from Hidden to Output units
            self.update_weights_HO()
            # Compute the new weights from Input to Hidden units            
            self.update_weights_IH()
            # Compute the new weights from Context to Hidden units
            self.update_weights_CH()
            # Copy of Hidden unit activity to Context units        
            self.CU = self.HU.copy()
        print('Training completed.')
    
    def plot_error(self,plotIdxs = [],start = 0, nTrials = []):
        import matplotlib.pyplot as plt
        if nTrials == []:
            nTrials = self.nTrials
        ## Visualization of the error rate ####
        plt.figure()
        if plotIdxs == []:
            plt.plot(range(nTrials-1),self.ERROR[start:start+nTrials-1,:])
            # plt.plot(range(self.nTrials-21),self.ERROR[20:-1,:])
        else:
            for i in plotIdxs:
                plt.plot(range(nTrials-1),self.ERROR[start:start+nTrials-1,i])
                # plt.plot(range(self.nTrials-21),self.ERROR[20:-1,i])
        plt.ylim(0)
    
    def testing(self,nrow,ncol):
        import matplotlib.pyplot as plt
        ## Testing after learning ####
        plt.figure()
        fig, axs = plt.subplots(nrow,ncol,sharey=True,sharex=True)
        titleText = 'Nu: ' + str(self.nu) + ', mom: ' + str(self.mom) + ', NH: ' + str(self.NH)
        fig.suptitle(titleText)
        x,y = 0,0
        for stim in range(self.NI):
            IU = np.zeros((1,self.NI))
            IU[0,stim] = 1
            HU = np.zeros((1,self.NH))
            OU = np.zeros((1,self.NO))
            
            # Compute activation from Input to Hidden
            for i in range(self.NI):
                for j in range(self.NH):
                    HU[0,j] += np.multiply(IU[0,i],self.WIH[i,j])
            HU = sigmoid(HU)
            
            # Compute activation from Hidden to Output
            for i in range(self.NH):
                for j in range(self.NO):
                    OU[0,j] += np.multiply(HU[0,i],self.WHO[i,j])
            OU = sigmoid(OU)
            
            # Plot Output
            axs[x,y].bar(range(self.NO),sum(OU))
            axs[x,y].set_ylim(0,1)
            axs[x,y].set_xlim(-0.5,self.NO - 0.5)
            if y < ncol-1:
                y += 1
            else:
                x += 1
                y = 0
        plt.show()
            