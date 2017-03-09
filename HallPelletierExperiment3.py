# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:12:17 2016

@author: Hajime
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import time
import sys
import os

dir_file = os.path.dirname(os.path.abspath( __file__ ))
dir_par = os.path.abspath( os.path.join( dir_file, '..') )
sys.path.append(dir_par)

from ModelSelection import ModelSelection
from IVreg_GMM_class3 import IVreg_GMM_Est, IVreg_GMM_Eval


class HallPelletierExperiment3:
    def __init__(self,hypara_est, setting_est, hypara_cv, setting_cv ,\
    setting_sim,\
    rep_input,savedata=1,savegraph=0,showfit=0\
    ):
        self.rep = rep_input

        self.n = setting_sim['n']
        self.mis=setting_sim['mis']        
        self.estpoly = setting_sim['estpoly']        
        self.alpha=setting_sim['alpha']
        self.gamma=setting_sim['gamma']        
        self.N_inst = setting_sim['N_inst']                
        self.N_char1 = setting_sim['N_char1']
        self.N_char2 = setting_sim['N_char2']
        self.var_u = setting_sim['var_u']
        self.var_z = setting_sim['var_z']
        self.flag_endg=setting_sim['flag_endg']
       
        self.savedata=savedata
        self.savegraph=savegraph
        self.showfit=showfit
        
        self.hypara_est=hypara_est
        self.setting_est=setting_est
        self.hypara_cv=hypara_cv
        self.setting_cv=setting_cv
        
        self.setting_gmm = {'samedata':0}
        self.r_fold = hypara_cv['r_fold']
        
    def Sim(self):
        if self.mis[0]==2:
            gamma = self.gamma
        elif self.mis[0]==1:
            gamma = self.gamma/float(self.n)
        elif self.mis[0]==0:
            gamma = 0.
        if self.mis[1]==2:
            alpha = self.alpha
        elif self.mis[1]==1:
            alpha = self.alpha/float(self.n)
        elif self.mis[1]==0:
            alpha = 0.
                    
        z = np.random.normal(size=[self.n,self.N_inst*2])*np.sqrt(self.var_z)
        z1 = z[:,0:self.N_inst]
        z2 = z[:,self.N_inst:self.N_inst*2]
        u = np.random.normal(size=[self.n, self.N_char1+self.N_char2+1]) * np.sqrt(self.var_u)
        eps1 = np.random.normal(size=[self.n, self.N_char1])* np.sqrt(self.var_u)
        eps2 = np.random.normal(size=[self.n, self.N_char2])* np.sqrt(self.var_u)

        x1_data =  np.repeat( np.sum(z1,axis=1),  self.N_char1).reshape([self.n, self.N_char1])+u[:,0:self.N_char1]
        x2_data =  np.repeat( np.sum(z2,axis=1),  self.N_char2).reshape([self.n, self.N_char2])+u[:,self.N_char1:self.N_char1+self.N_char2]

        if self.flag_endg==1:
            x1_data=x1_data+eps1
            x2_data=x2_data+eps2

        #y = np.sum(x1_data,axis=1).flatten() + np.sum(x2_data,axis=1).flatten()+u[:,self.N_char1+self.N_char2]+alpha*z[:,-1]+gamma*z[:,0]
        y = np.sum(x1_data,axis=1).flatten() + np.sum(x2_data,axis=1).flatten()+alpha*z[:,-1]+gamma*z[:,0]
        if self.flag_endg==1:
            y=y+np.sum(eps1,axis=1)+np.sum(eps2,axis=1)
                    
        Data1 ={'x':x1_data, 'y':y, 'iv':z1}
        Data2 ={'x':x2_data, 'y':y, 'iv':z2}
        self.Data = [Data1, Data2]
        
        
    def Experiment(self):
        start = time.time()  
        #record results
        self.gmm_stat = np.zeros(self.rep) #difference
        self.cv_stat = np.zeros(self.rep) #difference        
        self.gmm_score = np.zeros([self.rep, 2]) #each model
        self.cv_score = np.zeros([self.rep, 2]) #each model
        #self.cv_lis_r = np.zeros([self.rep,2,self.r_fold])
        
        EstClasses = [IVreg_GMM_Est, IVreg_GMM_Est]
        EvalClasses = [IVreg_GMM_Eval, IVreg_GMM_Eval]        

        for i in range(self.rep):
            self.Sim()
            gmm = CV_fit_class.GMM_fit( EstClass_input = EstClasses, EvalClass_input = EvalClasses, Data_all = self.Data,\
            hypara_est=self.hypara_est, setting_est = self.setting_est, setting_gmm = self.setting_gmm)
            gmm.Evaluate()
            self.gmm_score[i,:] = gmm.GMMScore
            
            cv = CV_fit_class.CV_fit( EstClass_input = EstClasses, EvalClass_input = EvalClasses, Data_all = self.Data,\
            hypara_est=self.hypara_est, setting_est = self.setting_est,\
            hypara_CV = self.hypara_cv, setting_CV = self.setting_cv)
            cv.Evaluate()
            self.cv_lis_r[i,:,:] = cv.cv_lis_r
            self.cv_score[i,:] = cv.cv_score
            
        end = time.time()
        t = end-start
        print('elapsed time: '+str(t))    
