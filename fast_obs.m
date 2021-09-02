function [xf,xs,outpars,LL] = fast_obs(y,M,p,r,S,pars,control,equal,fixed,scale)
%--------------------------------------------------------------------------
% Title:    Parameter estimation and inference in state-space models with 
%           regime switching (switching observations) assuming regimes known
%
% Function: Infer hidden state vectors and regimes by switching Kalman 
%           filtering/smoothing (aka Hamilton filtering or Kim filtering) 
%           and estimate model parameters by maximum likelihood (EM algorithm).
%
% Usage:    [Mf,Ms,Sf,Ss,xf,xs,outpars,LL] = ... 
%               fast_obs(y,M,p,S,pars,control,equal,fixed,scale)
% 
% Inputs:  
%       y - Time series (dimension NxT)
%       M - number of regimes
%       p - order of VAR model for state vector 
%       pars - optional structure with fields
%           A - Initial estimates of VAR matrices A(l,j) in system equation 
%               x(t,j) = sum(l=1:p) A(l,j) x(t-l,j) + v(t,j), j=1:M (dimension rxrxpxM)  
%           C - Initial estimates of observation matrices C(j) in equation 
%               y(t) = C(j) x(t,j) + w(t), j=1:M (dimension NxrxM)
%           Q - Initial estimates of state noise covariance Cov(v(t,j)) (dimension rxrxM)
%           R - Pilot estimate of observation noise covariance Cov(w(t)) (dimension NxN)                  
%           mu - Pilot estimate of mean state mu(j)=E(x(t,j)) for t=1:p (dimension rxM) 
%           Sigma - Pilot estimate of covariance Sigma(j)=Cov(x(t,j)) for t=1:p (dimension rxrxM)           
%       S - regime sequence (length T)
%       control - optional struct variable with fields: 
%           'eps': tolerance for EM termination; defaults to 1e-8
%           'ItrNo': number of EM iterations; defaults to 1000 
%           'beta0': initial inverse temperature parameter for deterministic annealing; default 1 
%           'betarate': decay rate for temperature; default 1 
%           'safe': if true, regularizes variance matrices to be well-conditioned 
%            before taking inverse. If false, no regularization (faster but less safe)
%           'abstol': absolute tolerance for eigenvalues in matrix inversion (only effective if safe = true)
%           'reltol': relative tolerance for eigenvalues in matrix inversion
%            = inverse condition number (only effective if safe = true)
%       equal - optional struct variable with fields:
%           'A': if true, VAR transition matrices A(l,j) are equal across regimes j=1,...,M
%           'C': if true, observation matrices C(j) are equal across regimes
%           'Q': if true, VAR innovation matrices Q(j) are equal across regimes
%           'mu': if true, initial mean state vectors mu(j) are equal across regimes
%           'Sigma': if true, initial variance matrices Sigma(j) are equal across regimes
%       fixed - optional struct variable with fields 'A','C','Q','R','mu','Sigma'.
%           If not empty, each field must contain a matrix with 2 columns, the first for 
%           the location of fixed coefficients and the second for their values. 
%       scale - optional struct variable with fields:
%           'A': upper bound for norm of eigenvalues of A matrices. Must be in (0,1).
%           'C': value of the (euclidean) column norms of the matrices C(j). Must be positive.
%
% Outputs: 
%       Mf - State probability estimated by switching Kalman Filter
%       Ms - State probability estimated by switching Kalman Smoother
%       Sf - Estimated states (Kalman Filter) 
%       Ss - Estimated states (Kalman Smoother) 
%       xf - Filtered state vector
%       xs - Smoothed state vector
%       outpars - structure with fields
%           A - Estimated system matrix
%           C - Estimated observation matrix
%           Q - Estimated state noise cov
%           R - Estimated observation noise cov
%           mu - Estimated initial mean of state vector
%           Sigma - Estimated initial variance of state vector 
%       LL  - Log-likelihood

% Variables:
%       T = length of signal
%       N = dimension of observation vector
%       r = dimension of state vector
%       M = number of regimes/states
%
% Author:       David Degras
%               University of Massachusetts Boston
%
% Contributors: Ting Chee Ming, cmting@utm.my
%               Siti Balqis Samdin
%               Centre for Biomedical Engineering, Universiti Teknologi Malaysia.
%               
% Version date: February 7, 2021
%--------------------------------------------------------------------------







%-------------------------------------------------------------------------%
%                           Initialization                                %
%-------------------------------------------------------------------------%


narginchk(5,10);

% Data dimensions
[N,T] = size(y);

% x(t,j): state vector for j-th process at time t (size r0)
% x(t) = x(t,1),...,x(t,M): state vector for all processes at time t (size M*r0)
% X(t,j) = x(t,j),...,x(t-p+1,j)): state vector for j-th process at times t,...,t-p+1 (size r=p*r0)
% X(t) = x(t,1),...,x(t,M): state vector for all processes at times t,...,t-p+1 (size M*p*r0)
% We assume t the initial vectors x(1),...,x(1-p+1) are iid ~ N(mu,Sigma)

% Check that time series has same length as regime sequence
assert(size(y,2) == numel(S));

% Check that all regime values S(t) are in 1:M
assert(all(ismember(S,1:M)));

% Data centering
y = y - mean(y,2);

%@@@@ Initialize optional arguments if not specified

if ~exist('fixed','var')
    fixed = struct();
end
if ~exist('equal','var')
    equal = struct();
end
if ~exist('control','var')
    control = struct();
end
if ~exist('scale','var')
    scale = struct();
end



%@@@@ Initialize estimators by OLS if not specified @@@@%
pars0 = struct('A',[], 'C',[], 'Q',[], 'R',[], 'mu',[], 'Sigma',[], ...
    'Pi',[], 'Z',[]);
if exist('pars','var') && isstruct(pars)
    fname = fieldnames(pars0);
    for i = 1:8
        name = fname{i};  
        if isfield(pars,name)
            pars0.(name) = pars.(name);
        end
    end
end
pars = pars0;
if any(structfun(@isempty,pars))
    pars = init_obs(y,M,p,r,[],control,equal,fixed,scale);
end

Pi = zeros(M,1);
Pi(S(1)) = 1;
pars.Pi = Pi;
fixed.Pi = [];
Z = crosstab(S(1:T-1),S(2:T));
Z = Z ./ sum(Z,2);
Z(isnan(Z)) = 1/M;
pars.Z = Z; 
fixed.Z = [];


% Preprocess input arguments
[pars,control,equal,fixed,scale,skip] = ... 
    preproc_obs(M,N,p,r,pars,control,equal,fixed,scale);

abstol = control.abstol;
reltol = control.reltol;
eps = control.eps;
ItrNo = control.ItrNo;
verbose = control.verbose;
safe = control.safe;

% Parameter sizes
% A: r x pr x M, C: N x r x M, Q: r x r x M, R: N x N, 
% mu: r x M, Sigma: r x r x M 
% (Only size of A is changed)


%@@@@ Initialize other quantities

LL = zeros(1,ItrNo); % Log-likelihood
LLbest = -Inf;
LLflag = 0; % counter for convergence of of log-likelihood 
sum_yy = y * y.'; % sum(t=1:T) y(t)*y(t)'
Ms = zeros(M,T); % P(S(t)=j), for use in Q-function
% Ms(sub2ind([M,T],S,T)) = 1;
sum_Ms2 = zeros(M);
for j = 1:M
    Ms(j,S == j) = 1;
    for k = 1:M
        sum_Ms2(j,k) = sum(S(1:end-1) == j & S(2:end) == k);
    end
end




for i = 1:ItrNo
    
   

%-------------------------------------------------------------------------%
%                     Filtering and smoothing + E-step                    %
%-------------------------------------------------------------------------%



    [xf,xs,x0,P0,L,sum_CP,sum_MP,sum_Mxy,sum_P,sum_Pb] = ...
        kfs_obs(y,M,p,r,S,pars,safe,abstol,reltol);    

    % Log-likelihood
    LL(i) = L; 
    if verbose
        fprintf('Iteration-%d   Log-likelihood = %g\n',i,LL(i));
        Qval = Q_obs(pars,Ms,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,...
            sum_P,sum_Pb,sum_yy,x0);
        fprintf('Iteration-%d   Q-function = %g (before M-step)\n',i,Qval);
    end
        
    % Check if current solution is best to date
    if i == 1 || LL(i) > LLbest
        LLbest = LL(i);
        xfbest = xf;
        xsbest = xs;
        outpars = pars;
    end
     
    % Monitor convergence of log-likelihood
    if i>1 && (LL(i)-LL(i-1)) < (eps * abs(LL(i-1)))
        LLflag = LLflag + 1;
    else
        LLflag = 0;
    end  
    
    % Terminate EM algorithm if no sufficient reduction in log-likelihood
    % for 5 successive iterations
    if LLflag == 5
        break;
    end
            
        
    
    
%-------------------------------------------------------------------------%
%                               M-step                                    %
%-------------------------------------------------------------------------%


    pars = M_obs(pars,Ms,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,...
        sum_P,sum_Pb,sum_yy,x0,control,equal,fixed,scale,skip);
    
    if verbose
        Qval = Q_obs(pars,Ms,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,...
            sum_P,sum_Pb,sum_yy,x0);
        fprintf('Iteration-%d   Q-function = %g (after M-step)\n',i,Qval);
    end
    
    
end % END MAIN LOOP



%-------------------------------------------------------------------------%
%                               Output                                    %
%-------------------------------------------------------------------------%

% Return best estimates (i.e. with highest log-likelihood) 
% after reshaping them in compact form
outpars.A = reshape(outpars.A,r,r,p,M);
outpars.Pi = Pi;
outpars.Z = Z;
xf = xfbest;
xs = xsbest;
LL = LL(1:i);

end

