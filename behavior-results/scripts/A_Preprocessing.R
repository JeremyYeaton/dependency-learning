# Data processing for experiment on regularity extraction in noise in baboons (2-item in 4-item sequence)
# (C) Jeremy D Yeaton
# September 2020

# Import and Preprocessing ------------------------------------------------
library(tidyverse)
'%notin%' <- function(x,y)!('%in%'(x,y))

rtMin = 100
rtMax = 800

# Read in raw data from site
raw_exp1 <- read.csv('data/results_raw_exp1_updated.txt',sep='\t') %>%
  select(-c(Programme,Manip,Famille,reward,Test,target6,target7,target8,
            target9,target10,target11,target12,rt5,rt6,rt7,rt8,rt9,rt10,rt11,
            rt12,rt13,rt14,rt15,rt16,touch1,touch2,touch3,touch4,RT,
            NbBloc,list1Criteria)) %>%
  rename(name = Nom,rawBlock = nObjAtteint,trialInBlock = nessai)

# Read in meta and group information (mostly about blocks and block orders)
brut_exp1 <- read.table('data/metaInfo.txt',sep = '\t', header = TRUE) %>%
  merge(read.table('groupInfo.csv',sep=',',header=TRUE), by = 'group') %>%
  select(c('Nom','group','pos1','pos2','pos3','blockType','blockOrder','nObjAtteint','typeOrder')) %>%
  rename(name = Nom,rawBlock = nObjAtteint) %>%
  merge(raw_exp1,by=c('name','rawBlock')) %>%
  mutate(trialNum = (100*typeOrder + trialInBlock - 1)) %>%
  filter(Score == 1)

# Get number of completed trials by baboon
trialCounts <- raw_exp1 %>%
  group_by(name) %>%
  summarise(nTrials = n())

# Filter, add columns, reshape long
brut_exp1 <- brut_exp1 %>%
  merge(trialCounts,by='name') %>%
  filter(nTrials == 2000) %>%
  select(-nTrials) %>%
  mutate(ptPair1 = paste('#',target1,sep = ''),
         ptPair2 = paste(target1,target2,sep=''),
         ptPair3 = paste(target2,target3,sep=''),
         ptPair4 = paste(target3,target4,sep=''),
         targetPair = case_when(blockType == 'pos1' ~ paste(target1,target2,sep=''),
                                blockType == 'pos2' ~ paste(target2,target3,sep=''),
                                blockType == 'pos3' ~ paste(target3,target4,sep=''),
                                blockType == 'rnd' ~ paste(0,0,sep=''))) %>%
  pivot_longer(cols = starts_with('rt'),names_to = 'rtType', values_to = 'rtVal') %>%
  mutate(ptPair = case_when(rtType == 'rt1' ~ ptPair1,
                            rtType == 'rt2' ~ ptPair2,
                            rtType == 'rt3' ~ ptPair3,
                            rtType == 'rt4' ~ ptPair4),
         pairType = case_when(blockType == 'rnd' ~ 'XX',
                              ptPair == targetPair ~ 'B',
                              substr(ptPair,2,2) == substr(targetPair,1,1) ~ 'A',
                              TRUE ~ 'XY')) %>%
  filter(rtVal <= rtMax & rtVal >= rtMin)

# Filter out RTs greater than 2.5 SD from baboon's mean
brut_exp1 <- brut_exp1 %>%
  group_by(name,rawBlock,rtType) %>%
  summarize(meanRT = mean(rtVal),sdRT = sd(rtVal)) %>%
  merge(brut_exp1) %>%
  mutate(rtZ = (rtVal-meanRT)/sdRT) %>%
  filter(abs(rtZ) <= 2.5)

exp1 <- brut_exp1 %>%
  select(-c(Score,bonnesrep,target1,target2,target3,target4,
            target5,pos1,pos2,pos3,meanRT,sdRT,rtZ,ptPair1,ptPair2,ptPair3,
            ptPair4,group,blockOrder,typeOrder)) %>%
  rename(cond = blockType) %>%
  mutate(condition = ifelse(cond == 'rnd','rnd4','fix4'),
         expNum = 1) %>%
  separate(Date,into = c('day','month','year'),sep='/') %>%
  separate(Heure,into=c('hour','minute','second'),sep=':')
  
## Random sequencing times ####
old_rnd_times <- read_csv('data/transitionTimes.csv') %>%
  rename(start = Pos1,
         stop = Pos2,
         oldTime = time) %>%
  mutate(ptPair = paste(start,stop,sep=''))

rndTimes <- exp1 %>%
  filter(cond == 'rnd') %>%
  group_by(ptPair) %>%
  summarize(baseRt = mean(rtVal),nTrials = n()) %>%
  merge(old_rnd_times,by= 'ptPair',all = TRUE) %>%
  mutate(start = substr(ptPair,1,1),
         stop = substr(ptPair,2,2))

## Experiment 2 ####
raw_exp2 <- read_delim('data/results_raw_exp2.txt',col_names=TRUE,delim='\t') %>%
  select(-c(Programme,Manip,Famille,reward,Test,target6,target7,target8,
            target9,target10,target11,target12,rt6,rt7,rt8,rt9,rt10,rt11,
            rt12,rt13,rt14,rt15,rt16,rt17,rt18,rt15_1,touch1,touch2,
            touch3,touch4,touch5,RT)) %>%
  rename(name = Nom,rawBlock = nObjAtteint,trialInBlock = nessai)

brut_exp2 <- raw_exp2 %>%
  group_by(name) %>%
  summarize(numFinished = n()) %>%
  merge(raw_exp2)%>%
  merge(read_csv('data/targetpairs_exp2.csv')) %>%
  mutate(cond = ifelse(rawBlock %in% c(0,1,7,8),'rnd','var')) %>%
  mutate(blockType = paste(cond,longsequence,sep=''),
         blockNum = ifelse(cond == 'rnd',
                           ifelse(rawBlock > 2,rawBlock - 7, rawBlock),
                           ifelse(rawBlock < 7,rawBlock - 2, rawBlock - 9))) %>%
  mutate(blockNum = blockNum + 1) %>%
  filter(numFinished > 1390 & Score == 1) %>%
  select(-c(numFinished,Score)) %>%
  mutate(ptPair1 = paste('#',target1,sep = ''),
         ptPair2 = paste(target1,target2,sep=''),
         ptPair3 = paste(target2,target3,sep=''),
         ptPair4 = paste(target3,target4,sep=''),
         ptPair5 = paste(target4,target5,sep='')) %>%
  pivot_longer(cols = starts_with('rt'),names_to = 'rtType', values_to = 'rtVal') %>%
  mutate(ptPair = case_when(rtType == 'rt1' ~ ptPair1,
                            rtType == 'rt2' ~ ptPair2,
                            rtType == 'rt3' ~ ptPair3,
                            rtType == 'rt4' ~ ptPair4,
                            rtType == 'rt5' ~ ptPair5),
         pairType = case_when(cond == 'rnd' ~ 'XX',
                              ptPair == targetPair ~ 'B',
                              substr(ptPair,2,2) == substr(targetPair,1,1) ~ 'A',
                              TRUE ~ 'XY')) %>%
  select(-c(target1,target2,target3,target4,target5,ptPair1,ptPair2,ptPair3,ptPair4,ptPair5)) %>%
  filter(rtVal <= rtMax & rtVal >= rtMin)

exp2 <- brut_exp2 %>%
  group_by(name) %>%
  summarize(mm = mean(rtVal), stdev = sd(rtVal)) %>%
  merge(brut_exp2) %>%
  mutate(Z = (rtVal - mm)/stdev) %>%
  filter(abs(Z) <= 2.5) %>%
  mutate(trialNum = (blockNum-1)*100 + trialInBlock,
         expNum = 2) %>%
  rename(condition = blockType) %>%
  separate(Date,into = c('day','month','year'),sep='/') %>%
  separate(Heure,into=c('hour','minute','second'),sep=':') %>%
  select(-c(Z,stdev,mm,blockNum))

# Merge data from both experiments
bothExp <- rbind(exp1,exp2)

bothExp <- bothExp %>%
  select(name) %>%
  distinct() %>%
  mutate(babNum = 1:24) %>%
  mutate(babNum = as.factor(babNum)) %>%
  merge(bothExp)

write_csv(bothExp,'data/both_experiments.csv')
write_csv(rndTimes,'data/rndTimes.csv')

rm(old_rnd_times,brut_exp1,brut_exp2,exp1,exp2,raw_exp1,raw_exp2,trialCounts)