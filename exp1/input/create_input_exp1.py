# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 12:18:55 2020

@author: Jeremy D Yeaton
"""

import os, numpy as np, pandas as pd, pickle
os.chdir('C:\\Users\\LPC\\Documents\\JDY\\NADs\\input')

transTimes = pd.read_csv('transitionTimes.csv')
meanRTmaster = np.mean(transTimes['time'])
sdRt = np.std(transTimes['time'])

babNames = ["ANGELE","ARIELLE","ARTICHO","ATMOSPHERE","CAUET","DORA","DREAM",\
            "EWINE","FANA","FELIPE","FEYA","FLUTE","HARLEM","HERMINE","KALI",\
                "LIPS","LOME","MAKO","MALI","MUSE","NEKKE","VIOLETTE","0","B06",\
                    "PIPO","BOBO","PETOULETTE","BRIGITTE"]
nBabs = len(babNames)
# Shuffle baboons
np.random.shuffle(babNames)

# Define group size to have 6 groups
grpCut = [5,10,15,20,24,nBabs]

grpDict = {
    1: [1,2,3],
    2: [1,3,2],
    3: [2,3,1],
    4: [2,1,3],
    5: [3,1,2],
    6: [3,2,1]    
    }

# Define structure positions
strucs = [[0,1],[1,2],[2,3]]

# All possible positions:
positions = [1,2,3,4,5,6,7,8,9]
#%%
# select different transitions for different monkeys
# for each triplet should have more or less the same response time
# for each triplet, take avg, std, and range
# less than 10 ms rt difference
# three pairs with no overlap
# make sure average is constant over AB, CD, and EF over all monkeys
#%% Permute pairings
'''
For each baboon, select 3 pairs of points for which: 
 - the range of RTs under random conditions is not greater than 10ms
 - the mean RT of the pairs is not more than 1.5 SD from the overall mean
It will continue to reshuffle until these conditions are met

It will then count how many times each pair appears across all of the baboons, 
and report if any pair appears more than 5 times

It returns 3 [n baboons x 3] lists:
    pos1: first position in regularities for each baboon
    pos2: second position in regularities for each baboon
    rt: transition time in random sequencing to go from position 1 to position 2
'''
j = 0
allTrips = []
dcheck = 0
for bab in range(nBabs):
    dupCheck = 0
    rtCheck = 0
    while dupCheck + rtCheck != 2:
        samp = transTimes.sample(n = 3, replace = False)
        dupCheck = 0
        rtCheck = 0
        p1 = [i for i in samp['Pos1']]
        p2 = [i for i in samp['Pos2']]
        sameP1 = p1[0] == p1[1] or p1[1] == p1[2] or p1[0] == p1[2]
        sameP2 = p2[0] == p2[1] or p2[1] == p2[2] or p2[0] == p2[2]
        anyDup = False
        for i in p1:
            if i in p2:
                anyDup = True
                break
        if sameP1 or sameP2 or anyDup:
            j+=1
        else:
            dupCheck = 1
        rts = [i for i in samp['time']]
        leftover = abs(np.mean(rts) - meanRTmaster)
        if max(rts) - min(rts) > 10 or leftover > (1.5*sdRt):#
            j+=1
        else:
            rtCheck = 1
        if dupCheck + rtCheck == 2:
            allTrips.append(samp)
print('Number of tries: ' + str(j))
## %% Extract pairs and rts
pos1 = []
pos2 = []
rt = []
for bab in range(nBabs):
    samp = allTrips[bab]
    pos1.append([i for i in samp['Pos1']])
    pos2.append([i for i in samp['Pos2']])
    rt.append([i for i in samp['time']])

d = {}
for i in range(len(pos1)):
    p1 = pos1[i]
    p2 = pos2[i]
    for j in range(len(p1)):
        reg = str([p1[j],p2[j]])
        if reg in d:
            d[reg] += 1
        else:
            d[reg] = 1
for key in d:
    if d[key] > 5:
        print('Failed: '+ key + ': ' + str(d[key]))
        dcheck = 1
print(len(d))
if dcheck == 0:
    print('Succcess!')
#%% Save successful run
pairDict = {'pos1': pos1,
            'pos2': pos2,
            'rt': rt,
            'd': d}
f = open('pairDict.pkl', "wb")
pickle.dump(pairDict, f)
f.close()
#%% Assign to monkeys and write files
'''
For each baboon, output a file comprised of 2000 trials:
 - 500 random trials
 - 500 trials with regularity (reg1) in position A
 - 500 trials with regularity (reg2) in position B
 - 500 trials with regularity (reg3) in position C
Ordering of position is determined by group membership (cf grpDict)
The master file is then subdivided into files of 100 trials each
'''
# Load in pairs that satisfy constraints
f = open('pairDict_max5.pkl', "rb")
pairDict = pickle.load(f)
f.close()
pos1 = pairDict['pos1']
pos2 = pairDict['pos2']
rt = pairDict['rt']

# Create metadata file
g = open('metaInfo.txt','w')
g.write('babName\t group\t strucOrder\t reg1\t reg1RT\t reg2\t reg2RT\t reg3\t reg3RT\t meanRT\n')

# Loop over baboons
grpIdx = 0
for bab in range(nBabs):
    babName = babNames[bab]
    
    # Produce random block
    rndBlock = np.zeros((500,16),dtype = int)
    for i in range(len(rndBlock)):
        rndBlock[i,:4] = np.random.choice(positions,size=(1,4),replace=False)
        
    # Assign regularity pairs to baboon
    AB = [pos1[bab][0],pos2[bab][0]]
    CD = [pos1[bab][1],pos2[bab][1]]
    EF = [pos1[bab][2],pos2[bab][2]]
    regs = [AB, CD, EF]
    
    # Remove regularity items from 'positions' to avoid repetition
    posAB = positions.copy()
    posCD = positions.copy()
    posEF = positions.copy()
    for i in range(2):
        posAB.remove(AB[i])
        posCD.remove(CD[i])
        posEF.remove(EF[i])
        
    # Produce starting state random trials for test blocks    
    rndAB = np.zeros((500,16),dtype = int)
    rndCD = np.zeros((500,16),dtype = int)
    rndEF = np.zeros((500,16),dtype = int)
    for i in range(500):
        rndAB[i,:4] = np.random.choice(posAB,size=(1,4),replace=False)
        rndCD[i,:4] = np.random.choice(posCD,size=(1,4),replace=False)
        rndEF[i,:4] = np.random.choice(posEF,size=(1,4),replace=False)
        
    # Open trials master file    
    f = open('master\\'+babName+'.txt','w')
    
    # Write random blocks to master
    for i in range(len(rndBlock)):
        # Line must begin with test,1,4, and end with ,[BLOCKNAME]
        f.write('test,1,4,')
        for j in range(np.shape(rndBlock)[1]):
            f.write(str(rndBlock[i,j])+',')
        f.write('random\n')
        
    # Define group membership and set structure orders for the blocks
    if bab > grpCut[grpIdx] - 1: # minus 1 to account for python 0 indexing
        grpIdx += 1
    grp = grpIdx + 1
    strucOrder = grpDict[grp]
    
    # Write meta data
    meanRT = round(np.mean(rt[bab]),2)
    g.write('\t'.join([babName,str(grp),str(strucOrder),str(regs[0]),\
                      str(rt[bab][0]),str(regs[1]),str(rt[bab][1]),\
                          str(regs[2]),str(rt[bab][2]),str(meanRT)])+'\n')
        
    # Loop over position x regularity blocks and write to master file
    for blockNum in range(3):
        reg = regs[blockNum]
        strucNum = strucOrder[blockNum]
        struc = strucs[strucNum-1]
        if blockNum == 0:
            block = np.copy(rndAB)
        elif blockNum == 1:
            block = np.copy(rndCD)
        elif blockNum == 2:
            block = np.copy(rndEF)
        for i in range(len(block)):
            block[i,struc[0]] = reg[0]
            block[i,struc[1]] = reg[1]
        for i in range(len(block)):
            f.write('test,1,4,')
            for j in range(np.shape(block)[1]):
                f.write(str(block[i,j])+',')
            f.write('Pos'+str(strucNum)+'Test\n')
    f.close()
g.close()

# Divide long files into 100 trial chunks
for bab in range(nBabs):
    babName = babNames[bab]
    f = open('master\\'+babName+'.txt','r')
    allLines = []
    for line in f:
        allLines.append(line)
    for Idx, line in enumerate(allLines):
        if Idx % 100 == 0:
            g.close()
            g = open('blocks100\\'+babName+'_'+str(int((Idx/100)+1))+'.txt','w')
        g.write(line)
    g.close()
print('Done!')