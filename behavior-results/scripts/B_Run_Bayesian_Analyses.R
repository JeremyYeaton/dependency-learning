# Bayesian statistics for experiments on regularity extraction in noise in baboons
# (C) Jeremy D Yeaton
# March 2021

# Load packages -----------------------------------------------------------
library(tidyverse)
library(BayesFactor)
library(pracma)
library(coda)

# Read in data ------------------------------------------------------------
# source('scripts/A_Preprocessing.R')
bothExp <- read_csv('data/both_experiments.csv')

# Experiment 1 ------------------------------------------------------------
# Get slopes
slopes_exp1 <- bothExp %>%
  filter(expNum == 1, pairType %in% c('XX','B')) %>%
  mutate(allOnes = 1,
         babNum = factor(babNum),
         cond = factor(cond)) %>%
  group_by(babNum,cond) %>%
  summarize(slope = mldivide(cbind(allOnes,trialNum),rtVal)) %>%
  filter(slope < 1)

# Calculate overall BF
bf.exp1 <- slopes_exp1 %>%
  anovaBF(slope ~ cond + babNum, data = ., whichRandom = 'babNum')
summary(bf.exp1)

post.exp1 <- posterior(bf.exp1,iterations = 10000,columnFilter="^babNum$")
summary(post.exp1)

exp1_post.df <- as.matrix(post.exp1) %>%
  as.data.frame(.) %>%
  rename('pos1' = 'cond-pos1',
         'pos2' = 'cond-pos2',
         'pos3' = 'cond-pos3',
         'rnd' = 'cond-rnd') %>%
  select(c(pos1,pos2,pos3,rnd)) %>%
  pivot_longer(cols = c(pos1,pos2,pos3,rnd), names_to = 'condition', values_to = 'value') %>%
  mutate(condition = factor(condition,levels = c('rnd','pos1','pos2','pos3')))

exp1_pairWiseBFs <- slopes_exp1 %>%
  pivot_wider(names_from = cond,values_from = slope) %>%
  mutate(rndPos1 = pos1-rnd,
         rndPos2 = pos2-rnd,
         rndPos3 = pos3-rnd,
         pos1Pos2 = pos2-pos1,
         pos1Pos3 = pos3-pos1,
         pos2Pos3 = pos3-pos2) %>%
  pivot_longer(cols = c(rndPos1,rndPos2,rndPos3,pos1Pos2,pos1Pos3,pos2Pos3),
               names_to = 'conds',values_to='diff') %>%
  split(.$conds) %>%
  map(~ttestBF(x=.$diff,mu=0))

# Experiment 2 ------------------------------------------------------------
# Get slopes
slopes_both <- bothExp %>%
  filter(pairType %in% c('XX','B'),condition != 'rnd5') %>%
  mutate(allOnes = 1,
         babNum = factor(babNum),
         condition = factor(condition)) %>%
  group_by(babNum,condition) %>%
  summarize(slope = mldivide(cbind(allOnes,trialNum),rtVal)) %>%
  filter(slope < 1)

# Calculate overall BF
bf.both <- slopes_both %>%
  anovaBF(slope ~ condition * babNum, data = ., whichRandom = 'babNum')
summary(bf.both)

exp2_pairWiseBFs <- slopes_both %>%
  pivot_wider(names_from = condition,values_from = slope) %>%
  mutate(rndFix4 = fix4-rnd4,
         rndVar4 = var4-rnd4,
         rndVar5 = var5-rnd4,
         fix4Var4 = var4-fix4,
         fix4Var5 = var5-fix4,
         Var4Var5 = var5-var4) %>%
  pivot_longer(cols = c(rndFix4,rndVar4,rndVar5,fix4Var4,fix4Var5,Var4Var5),
               names_to = 'conds',values_to='diff') %>%
  filter(!is.na(diff)) %>%
  split(.$conds) %>%
  map(~ttestBF(x=.$diff,mu=0))

post.exp2 <- posterior(bf.both,iterations = 10000,columnFilter="^babNum$")
summary(post.exp2)

exp2_post.df <- as.matrix(post.exp2) %>%
  as.data.frame(.) %>%
  rename('fix4' = 'condition-fix4',
         'var4' = 'condition-var4',
         'var5' = 'condition-var5',
         'rnd' = 'condition-rnd4') %>%
  select(c(fix4,rnd,var4,var5)) %>%
  pivot_longer(cols = c(fix4,rnd,var4,var5), names_to = 'condition', values_to = 'value') %>%
  mutate(condition = factor(condition,levels = c('rnd','fix4','var4','var5')))