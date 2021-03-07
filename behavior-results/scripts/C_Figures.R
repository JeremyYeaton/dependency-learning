# Bayesian statistics for experiments on regularity extraction in noise in baboons
# (C) Jeremy D Yeaton
# March 2021

# Load packages -----------------------------------------------------------
library(tidyverse)
library(bayesplot)
library(ggpubr)
library(viridis) # TODO

# Read in data & model objects --------------------------------------------
source('scripts/B_Run_Bayesian_Analyses.R')

mcmc_intervals(post.exp1,pars=c('cond-pos1','cond-pos2','cond-pos3','cond-rnd'))

mcmc_areas(post.exp1,
           pars=c('cond-pos1','cond-pos2','cond-pos3','cond-rnd'),
           prob = .9,
           prob_outer = .99,
           point_est = 'mean')

mcmc_dens(post.exp1,pars=c('cond-pos1','cond-pos2','cond-pos3','cond-rnd'))

exp1.dense.plot <- exp1_post.df %>%
  ggplot(aes(x=value,color = condition)) +
  geom_density(size = 1.25) +
  lims(x=c(-.15,.15),y=c(0,42)) +
  labs(x='Slope',y='Density',color='Condition') +
  theme(legend.position = 'top')

mcmc_intervals(post.exp2,pars=c('condition-fix4','condition-var4','condition-var5','condition-rnd4'))

mcmc_areas(post.exp2,
           pars=c('condition-fix4','condition-var4','condition-var5','condition-rnd4'),
           prob = .9,
           prob_outer = .99,
           point_est = 'mean')

mcmc_dens(post.exp2,pars=c('condition-fix4','condition-var4','condition-var5','condition-rnd4'))

exp2.dense.plot <- exp2_post.df %>%
  ggplot(aes(x=value,color = condition)) +
  geom_density(size = 1.25) +
  lims(x=c(-.15,.15),y=c(0,42)) +
  labs(x='Slope',y='',color='Condition') +
  theme(legend.position = 'top')

ggarrange(exp1.dense.plot,exp2.dense.plot,ncol=2,labels = c('A','B'))

slopes_exp1 %>%
  ggplot(aes(x=slope,fill = cond)) +
  geom_density(alpha=.25)

slopes_both %>%
  ggplot(aes(x=slope,fill = condition)) +
  geom_density(alpha = .25)