# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 23:16:52 2016

@author: Hajime
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import time
import sys

class IVreg_GMM_Est:
    
    def __init__(self, endg_col,x_use,z_use, reg_type, weight_type, data_col, data_col_test):
        self.x_use = x_use
        self.z_use = z_use
        self.endg_col = endg_col
        self.reg_type = reg_type
        self.weight_type = weight_type
        
    def Est(self):
        if self.reg_type=='2sls':
            bhat, f = self.ivreg_2sls(self.x, self.y, self.IV, self.weight)
        if self.reg_type=='2tepGMM':
            bhat, f = self.ivreg_2stepGMM(self.x, self.y, self.IV, self.weight)
            
        self.EstResult = {'bhat':bhat, 'W':self.weight}
        return bhat, f
    
    def fit(self,Data):
        self.x = Data[:,self.data_col['x'] ][:,self.x_use]
        self.y = Data[:,self.data_col['y'] ]
        self.iv = Data[:,self.data_col['iv'] ][:,self.z_use]
        self.n, self.n_char = self.x.shape
        self.n_iv = self.iv.shape[1]
        
        self.exog_col = np.delete(np.arange(self.n_char) , self.endg_col)
        
        self.IV = np.c_[ self.x[:,self.exog_col],self.iv ]
        self.n_IV = self.IV.shape[1]
        
        if self.weight_type =='identity':
            self.weight = np.eye(self.n_IV).astype(float)
        elif self.weight_type =='invA':
            self.weight = np.linalg.inv( np.dot( self.IV.T, self.IV ) )
        

        if self.reg_type=='2sls':
            self.bhat, self.f = self.ivreg_2sls(self.x, self.y, self.IV, self.weight)
        if self.reg_type=='2tepGMM':
            self.bhat, self.f = self.ivreg_2stepGMM(self.x, self.y, self.IV, self.weight)
            
        self.EstResult = {'bhat':self.bhat, 'W':self.weight}
        return self.bhat, self.f

    def score(self,Data):
        self.x_test = Data[:,self.data_col_test['x'] ][:,self.x_use]
        self.y_test = Data[:,self.data_col_test['y'] ]
        self.iv_test = Data[:,self.data_col_test['iv'] ][:,self.z_use]
        self.n_test, self.n_char_test = self.x.shape_test
        self.n_iv_test = self.iv_test.shape[1]
        
        self.IV_test = np.c_[ self.x_test[:,self.exog_col],self.iv_test ]
        self.n_IV_test = self.IV_test.shape[1]
 
        if self.weight_type =='identity':
            self.weight_test = np.eye(self.n_IV_test).astype(float)
        elif self.weight_type =='invA':
            self.weight_test = np.linalg.inv( np.dot( self.IV_test.T, self.IV_test ) )
       
        f = self.ivreg_resid(self.x_test, self.y_test, self.IV_test, self.bhat, self.weight_test)
        return f

    def ivreg_resid(self,x,y,z,bhat,invA=None):
        self.ivreg_resid_bhat=bhat
        n = y.size
        if invA is None:
            N_inst = z.shape[1]
            #invA = np.linalg.solve( np.dot(z.T,z), np.identity(N_inst) )
            invA = np.identity(N_inst) 
        gmmresid = y - np.dot(x, bhat)
        temp5=np.dot(gmmresid.T, z)
        f=np.dot(np.dot(temp5/n, invA),(temp5.T)/n) #divided by n  -> Q=G'WG G=1/n sum g
        del x,y,z,invA
        return f

        
    def ivreg_2sls(self,x,y,z,invA=None):
        n = y.size
        if invA is None:
            N_inst = z.shape[1]
            #invA = np.linalg.solve( np.dot(z.T,z), np.identity(N_inst) )
            invA = np.identity(N_inst) 
        temp1 = np.dot(x.T,z)
        temp2 = np.dot(y.T,z)
        temp3 = np.dot(np.dot(temp1,invA),temp1.T) #x'z(z'z)^{-1}z'x
        temp4 = np.dot(np.dot(temp1,invA),temp2.T) #x'z(z'z)^{-1}z'y
        if temp3.size==1 or temp4.size==1:
            bhat = temp4/temp3
        else:
            try:
                bhat = np.linalg.solve(temp3,temp4)
            except np.linalg.LinAlgError:
                self.flag_SingularMatrix=1
                print("singular matrix")
                return
        gmmresid = y - np.dot(x, bhat)
        temp5=np.dot(gmmresid.T, z)
        f=np.dot(np.dot(temp5/n, invA),(temp5.T)/n)
        del x,y,z,invA
        return bhat,f

    def ivreg_2stepGMM(self,x,y,z,invA=None):
        n = y.size
        if invA is None:
            N_inst = z.shape[1]
            #invA = np.linalg.solve( np.dot(z.T,z), np.identity(N_inst) )
            invA = np.identity(N_inst) 
        bhat_1st,f1 = self.ivreg_2sls(x,y,z,invA)
        eps = np.array([y - np.dot(x, bhat_1st)]).T
        W2 = (1./n) * np.dot( np.dot(z.T, eps), np.dot(eps.T, z) )
        bhat_2nd, f2 = self.ivreg_2sls(x,y,z,invA=W2)
        return bhat_2nd,f2,W2
'''
class IVreg_GMM_Eval:

    def __init__(self, Data, hypara, EstResult, setting):
        self.x = Data['x']
        self.y = Data['y']
        self.iv = Data['iv']
        
        self.endg_col = hypara['endg_col']
        self.exog_col = 1 - self.endg_col
        
        self.n, self.n_char = self.x.shape
        self.n_iv = self.iv.shape[1]
        
        self.IV = np.c_[ self.x[:,self.exog_col],self.iv ]
        self.n_IV = self.IV.shape[1]
        
        self.reg_type = setting['reg_type']
        self.weight_type = setting['weight_type']
        if self.weight_type =='identity':
            self.weight = np.eye(self.n_IV)
        elif self.weight_type =='invA':
            self.weight = np.dot( self.IV.T, self.IV )
            
        self.bhat_est = EstResult['bhat']
        if EstResult['W']!=None:
            self.W_est = EstResult['W']
        
    def Eval(self):
        f = self.ivreg_resid(x=self.x, y=self.y, z=self.IV, bhat=self.bhat_est, invA=self.W_est)
        self.EvalScore = f
        return f

    def ivreg_resid(self,x,y,z,bhat,invA=None):
        self.ivreg_resid_bhat=bhat
        n = y.size
        if invA is None:
            N_inst = z.shape[1]
            #invA = np.linalg.solve( np.dot(z.T,z), np.identity(N_inst) )
            invA = np.identity(N_inst) 
        gmmresid = y - np.dot(x, bhat)
        temp5=np.dot(gmmresid.T, z)
        f=np.dot(np.dot(temp5/n, invA),(temp5.T)/n) #divided by n  -> Q=G'WG G=1/n sum g
        del x,y,z,invA
        return f
'''
        
if __name__=='__main__':
    x=np.random.normal(size=[20,5])
    beta=np.array([1.,2.,3.,1.5,2.5])
    y=np.dot(x,beta)+np.random.normal(20)
    z=np.random.normal(size=[20,5])
    Data={'x':x,'y':y,'iv':z}
    hypara={'endg_col':np.array([0,1,3])}
    setting={'reg_type':'2sls','weight_type':'identity'}

