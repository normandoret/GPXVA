import scipy.stats as st
import numpy as np
import math

def bsformula(cp, s, k, rf, t, v, div):
        """ Price an option using the Black-Scholes model.
        cp: +1/-1 for call/put
        s: initial stock price
        k: strike price
        t: expiration time
        v: volatility
        rf: risk-free rate
        div: dividend
        """

        d1 = (np.log(s/k)+(rf-div+0.5*v*v)*t)/(v*np.sqrt(t))
        d2 = d1 - v*np.sqrt(t)

        optprice = (cp*s*np.exp(-div*t)*st.norm.cdf(cp*d1)) - (cp*k*np.exp(-rf*t)*st.norm.cdf(cp*d2))
        delta = cp*st.norm.cdf(cp*d1)
        vega  = s*np.sqrt(t)*st.norm.pdf(d1)
        return optprice, delta, vega

#######################################################
#  Digital Option Black-Scholes
########################################################

def Digital_call_MC_BS(r,S,sigma,T,K,eps):
    price1, delta1, vega1 = bsformula(1, S, K-eps/2, r, T, sigma, 0)
    price2, delta2, vega2 = bsformula(1, S, K+eps/2, r, T, sigma, 0)
    return (price1-price2)/eps, (delta1-delta2)/eps, (vega1-vega2)/eps

def Digital_put_MC_BS(r,S,sigma,T,K,eps):
    price1, delta1, vega1 = bsformula(-1, S, K+eps/2, r, T, sigma, 0)
    price2, delta2, vega2 = bsformula(-1, S, K-eps/2, r, T, sigma, 0)
    return (price1-price2)/eps, (delta1-delta2)/eps, (vega1-vega2)/eps

###################################################

if __name__ == "__main__":
     ex = black_scholes(-1, 100.0, 110.0, 2.5, 0.4, 0.05, 0.0)
