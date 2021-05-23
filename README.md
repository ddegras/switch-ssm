# switch-ssm
Markov-Switching State-Space Models

This is a suite of Matlab functions for fitting Markov-switching state-space models (SSMs) to multivariate time series data by maximum likelihood. We consider three switching SSMs: switching dynamics, switching observations, and swiching vector autoregressive (VAR). The maximum likelihood estimator is calculated via an approximate EM algorithm. (Exact calculations are not tractable because of exponential number of possible regime histories, M^T with M the number of states/regimes for the Markov chain and T the length of the time series.) To keep calculations tractable, we use the filtering/smoothing algorithm of Kim (1994) in the E-step of the EM algorithm.

## Functions
The user-level functions of the package are of the form `xxx_yyy`, where the prefix `xxx` indicates what the function does and the suffix `yyy` indicates which model the function applies to.  
The possible prefixes are: 
- `init`: find starting values for EM algorithm
- `switch`: fit EM algorithm
- `fast`: fit EM algorithm with fixed regime sequence
- `reestimate`: estimate model parameters by least squares with fixed regime sequence
- `bootstrap`: perform parametric bootstrap   
- `simulate`: simulate a realization of the model
The possible suffixes are:   
- `dyn`: switching dynamics model
- `obs`: switching observations model 
- `var`: swiching vector autoregressive model

## Authors
**Author:** David Degras
**Contributors:** Chee Ming Ting @CheeMingTing, Siti Balqis Samdin

## References
- Degras, D., Ting, C.M., and Ombao, H.: Markov-Switching State-Space Models with Applications to Neuroimaging. _In preparation_ (2021)
- Kim, C.J.: Dynamic linear models with Markov-switching. _J. Econometrics_ 60(1-2), 1–22 (1994)
- Kim, C.J., Nelson, C.R.: State-Space Models with Regime Switching: Classical and Gibbs-Sampling Approaches with Applications. _The MIT Press_ (1999)
- Murphy, K.P.: Switching Kalman filters. Tech. rep., _University of California Berkeley_ (1998)
