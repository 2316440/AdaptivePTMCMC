import numpy as np
from scipy.stats import norm, multivariate_normal
import random


# the Metropolis-Hastings algorithm

def metropolis(logprob, n, Q, theta_min, theta_max, start_theta, T, info=False):
    """
    Parameters
    ----------
    logprob: function
        Function which caluclates the logposterior (log of the sampled distribution)
    n: int
        Number of samples
    Q: matrix
        Covariance matrix of the multivariate Gaussian jump proposal distribution
    theta_min, theta_max: arrays/lists of floats
        Prior bounds
    start_theta: array/list of floats
        Starting location of the Markov chain (logprob(start_theta) has to be different from 0)
    T: float
        Temperature of the posterior
    info: Boolean
        if True, the function outputs an dictionary containing information about acceptance rates
    
    """    
    
    acc = 0       # initialise acceptance counter
    i = 0         # initialise the counter
    pos = []      # initialise the list of accepted states
    ndim = np.array(theta_min).size        # extract the dimensionality of the problem
    current_theta = start_theta            # set the starting location 
    current_log_prob = logprob(current_theta, theta_min, theta_max, T)   # find log posterior prob at the current location

    while i<n:
        new_theta = current_theta + multivariate_normal.rvs(mean=ndim*[0],cov=Q)  # choose proposed location from a multivariate normal
        new_log_prob = logprob(new_theta, theta_min, theta_max, T)   # find log posterior prob at proposed location
        log_R = np.log(np.random.uniform(0,1))                       # draw a random number in range (0,1)
        if new_log_prob - current_log_prob > log_R:       # rejection sampling: compare posterior probability-ratios to r
            current_log_prob = new_log_prob               # if the jump is accepted, update current log posterior 
            current_theta = new_theta                     # if the jump is accepted, update current state
            acc += 1                                      # update acceptance count 
        pos.append(current_theta)                 # append the chain with current location
                
        i += 1               # increment counter
        
    if info==True:
        infodict = {'acceptance ratio': acc/n}     # calculate jump acceptance rate
        return np.array(pos), infodict
    
    return np.array(pos)     # return the chain of accepted locations


# the Adaptive PTMCM algorithm

def parallel_tempering(logprob, logL, n, nswap, temps, Q, theta_min, theta_max, theta_start, swap=True, adapt=True, infs=False):
    """
    Parameters
    ----------
    logprob: function
       Caluclates the tempered logposterior (log of the sampled distribution) given theta and T
    logL: function
       Caluclates the loglikelihood given theta
    n: int
        Number of samples
    nswap: int
        Number of Metropolis-Hastings (M-H) updates before swaps are proposed
    temps: array/list of floats
        Temperatures of the tempered posteriors
    Q: matrix
        Intitial covariance matrix of the multivariate Gaussian jump proposal distributions
    theta_min, theta_max: arrays/lists of floats
        Prior bounds
    theta_start: array/list of floats
        Starting location of the Markov chains (logprob(start_theta) has to be different from 0)
    swap: Boolean
        if True, PTMCMC swaps are proposed, if False this reduces to the standard M-H algorithm (Default: True)
    adapt: Boolean
        if True, the AP adaptation is turned on (Default=True)
    infs: Boolean
        if True, the function outputs an dictionary containing information about M-H acceptance rates and accepted swaps
    
    """   
    
    i=0 #initiate a counter
    pos = {T: [] for T in temps}        #initialize a dictionary to store sampler positions for chains at different temperatures
    state = {T: theta_start for T in temps}           #set the initial state to start_theta for all chains
    QQ = {T: Q for T in temps}                        #set the initial covariance of the proposal distribution for all chains
    infodict = {T: {'M-H acceptance ratio': 0}  for T in temps}   #initialize an info dictionary 
    swaps=[]                                        # list to keep an account of swaps
    N=float(len(theta_start))                       # extract dimensionality of the problem
    k=(2.38)**2 /N                                  # dimensionality dependent scaling constant 
    
    while i<int(n/nswap): 
        
        for T in temps:                                                 #get M-H updates for all tempered chains
            samples, acc = metropolis(logprob, nswap-1,QQ[T],theta_min,theta_max,state[T],T,info=True)
            pos[T].extend(samples)                                                    #append the chain with 99 samples
            state[T] = samples[-1]                                                    #record the end state of the chain for proposal swap
            infodict[T]['M-H acceptance ratio']+=list(acc.values())[0]                #get acceptance rate information
            
            if adapt == True and i*nswap > 1000:       #update covariance matrix for the jump proposal (begin after 1000 initial samples) 
                QQ[T]=k*np.cov(np.array(pos[T][-100:]).transpose()) + np.diag(np.ones(int(N))*1e-12)   
            
        if swap==True:                                                 #propose state swaps between tempered chains
            #create a decreasing temperature ladder for state swaps
            t_ladder=np.arange(0,len(temps)-1,1)   
            t_ladder=t_ladder[::-1]           

            #propose state swaps
            for idx in t_ladder:
                T_low=temps[idx]
                T_high=temps[idx+1]
                log_l_T_low = logL(state[T_low])                  #likelihood of the lower temp state  
                log_l_T_high = logL(state[T_high])                   #likelihood of the higher temp state  

                log_R = np.log(np.random.uniform(0,1))                            # draw a random number in range (0,1)
                if (1/T_low - 1/T_high)*(log_l_T_high-log_l_T_low) > log_R:        # compare prob-ratios with random number (in log space)
                    state[T_low], state[T_high] = state[T_high], state[T_low]           # do the swap
                    swaps.append(idx)                                                   # keep track of accepted swaps

        #record the new states, same or swapped.
        for T in temps:
            pos[T].append(state[T])
        
        i += 1               # increment counter
        
    for T in temps:                    #change position lists to arrays
            pos[T]=np.array(pos[T])
        
    if infs==True:                  #calculate some information about the chains (average M-H acceptance rates, accepted swaps)
        for T in temps:
            infodict[T]['M-H acceptance ratio']/= int(n/nswap)  
        infodict['swaps']=np.array(swaps)  
        return pos, infodict
        
    return pos
