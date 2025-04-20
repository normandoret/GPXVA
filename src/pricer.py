import numpy as np
import numpy.random as npr 
import matplotlib.pyplot as plt
from scipy.stats import norm

#######################################################
#  European Option Black-Scholes
########################################################

def European_call_MC_BS(r,S0,sigma,T,K,Sample_size):
    
    G=npr.normal(0,1,size=(1,Sample_size))
    S=S0*np.exp((r-sigma**2/2)*T+sigma*np.sqrt(T)*G)  # WT=np.sqrt(T)*G
    
    payoff=np.exp(-r*T)*np.maximum(S-K,0) #call function

    MC_price=np.mean(payoff)

#     # 95% C.I

    STD=np.std(payoff) # standard deviation estimator

    error=1.96*STD/np.sqrt(Sample_size)

    CI_up=MC_price + error
    CI_down=MC_price -error
    
    
    # True price by Balck-Scholes formula 


    d1= 1./(sigma*np.sqrt(T))*(np.log(S0/K)+(r+sigma**2/2)*T)
    d2= 1./(sigma*np.sqrt(T))*(np.log(S0/K)+(r-sigma**2/2)*T)
    True_price= S0*norm.cdf(d1) -K*np.exp(-r*T)*norm.cdf(d2)
    
    #True Put price
    
    
    return MC_price, True_price, CI_up, CI_down, error


#######################################################
#  Asian Option Black-Scholes
########################################################

def European_Asian_call_MC_BS(r,S0,sigma,T,K,N,n):
    
    # N Sample size
    # n number of time steps
    
    G=npr.normal(0,1,(N,n))
    step=T/n
    
    Log_returns=sigma*np.sqrt(step)*G + (r-sigma**2/2)*step
    
    Log_returns_S0=np.concatenate((np.ones((N,1))*np.log(S0),Log_returns),axis=1)
    
    Log_Paths=np.cumsum(Log_returns_S0, axis=1)
    
    Spaths=np.exp(Log_Paths)
    
    #S_bar=np.mean(Spaths[:,:-1],axis=1) # Riemann sum
    S_trap_bar=np.mean((Spaths[:,:-1]+Spaths[:,1:])*0.5,axis=1)
    
    #payoff= np.exp(-r*T)*np.maximum(S_bar-K,0) # payaoff Riemann
    payoff= np.exp(-r*T)*np.maximum(S_trap_bar-K,0)
    # (S-K)_+ use np.maximum and not np.max
    
    
    price=np.mean(payoff)   # empirical mean
    std= np.std(payoff)    # empirical standard deviation


    # 95% Confidence interval bounds 
    
    CI_up= price +1.96*std/np.sqrt(N)     
    CI_down= price -1.96*std/np.sqrt(N)
    error= 1.96*std/np.sqrt(N)
    
    return price, error, CI_up, CI_down

def European_Asian_put_MC_BS(r,S0,sigma,T,K,N,n):
    
    # N Sample size
    # n number of time steps
    
    G=npr.normal(0,1,(N,n))
    step=T/n
    
    Log_returns=sigma*np.sqrt(step)*G + (r-sigma**2/2)*step
    
    Log_returns_S0=np.concatenate((np.ones((N,1))*np.log(S0),Log_returns),axis=1)
    
    Log_Paths=np.cumsum(Log_returns_S0, axis=1)
    
    Spaths=np.exp(Log_Paths)
    
    #S_bar=np.mean(Spaths[:,:-1],axis=1) # Riemann sum
    S_trap_bar=np.mean((Spaths[:,:-1]+Spaths[:,1:])*0.5,axis=1)
    
    #payoff= np.exp(-r*T)*np.maximum(S_bar-K,0) # payaoff Riemann
    payoff= np.exp(-r*T)*np.maximum(K-S_trap_bar,0)
    # (S-K)_+ use np.maximum and not np.max
    
    
    price=np.mean(payoff)   # empirical mean
    std= np.std(payoff)    # empirical standard deviation


    # 95% Confidence interval bounds 
    
    CI_up= price +1.96*std/np.sqrt(N)     
    CI_down= price -1.96*std/np.sqrt(N)
    error= 1.96*std/np.sqrt(N)
    
    return price, error, CI_up, CI_down


#######################################################
#  Barrier Option Black-Scholes
########################################################

def Barrier_call_MC_BS(r,S0,sigma,T,K,B,N,n):
    delta=float(T/n)

    G=npr.normal(0,1,size=(N,n))

    #Log returns
    LR=(r-0.5*sigma**2)*delta+np.sqrt(delta)*sigma*G
    # concatenate with log(S0)
    LR=np.concatenate((np.log(S0)*np.ones((N,1)),LR),axis=1)
    # cumsum horizontally (axis=1)
    LR=np.cumsum(LR,axis=1)
    # take the expo Spath matrix
    Spaths=np.exp(LR)
    
    #take the maximum over each path
    
    Spathsmax=np.max(Spaths,axis=1)
    
    payoff=np.exp(-r*T)*np.maximum(Spaths[:,-1]-K,0)*(Spathsmax>B)
    
    Barrier_MC_price=np.mean(payoff)

    # 95% C.I

    sigma=np.std(payoff) # standard deviation estimator

    error=1.96*sigma/np.sqrt(N)

    CI_up=Barrier_MC_price + error
    CI_down=Barrier_MC_price -error
    
    return Barrier_MC_price,CI_up,CI_down,error



def Barrier_put_MC_BS(r,S0,sigma,T,K,B,N,n):
    delta=float(T/n)

    G=npr.normal(0,1,size=(N,n))

    #Log returns
    LR=(r-0.5*sigma**2)*delta+np.sqrt(delta)*sigma*G
    # concatenate with log(S0)
    LR=np.concatenate((np.log(S0)*np.ones((N,1)),LR),axis=1)
    # cumsum horizontally (axis=1)
    LR=np.cumsum(LR,axis=1)
    # take the expo Spath matrix
    Spaths=np.exp(LR)
    
    #take the maximum over each path
    
    Spathsmin=np.max(Spaths,axis=1)
    
    payoff=np.exp(-r*T)*np.maximum(K-Spaths[:,-1],0)*(Spathsmin<B)
    
    Barrier_MC_price=np.mean(payoff)

    # 95% C.I

    sigma=np.std(payoff) # standard deviation estimator

    error=1.96*sigma/np.sqrt(N)

    CI_up=Barrier_MC_price + error
    CI_down=Barrier_MC_price -error
    
    return Barrier_MC_price,CI_up,CI_down,error


