function [Mf,Ms,Sf,Ss,xf,xs,outpars,LL] = ... 
    switch_obs(y,M,p,r,pars,control,equal,fixed,scale)

%--------------------------------------------------------------------------
% Title:    Parameter estimation and inference in state-space models with regime switching
%            (switching observations model)
%
% Function: Infer hidden state vectors and regmes by switching Kalman filtering/smoothing
%            (aka Hamilton filtering or Kim filtering) and estimate model parameters by 
%            maximum likelihood (EM algorithm).
%
% Usage:    [Mf,Ms,Sf,Ss,xf,xs,outpars,LL] = ... 
%               switch_obs(y,M,p,pars,control,equal,fixed,scale)
%
% Inputs:   y - Time series (size NxT)
%           M - number of regimes
%           p - order of VAR model for state vector 
%           pars - optional 'struct' of EM starting values
%               A - Initial estimates of VAR matrices A(l,j) in system equation 
%               x(t,j) = sum(l=1:p) A(l,j) x(t-l,j) + v(t,j), j=1:M (size rxrxpxM)  
%               C - Initial estimates of observation matrices C(j) in equation 
%               y(t) = C(j) x(t,j) + w(t), j=1:M (size NxrxM)
%               Q - Initial estimates of state noise covariance Cov(v(t,j)) (size rxrxM)
%               R - Pilot estimate of observation noise covariance Cov(w(t)) (size NxN)                  
%               mu - Pilot estimate of mean state mu(j)=E(x(t,j)) for t=1:p (size rxM) 
%               Sigma - Pilot estimate of covariance Sigma(j)=Cov(x(t,j)) for t=1:p (size rxrxM)           
%               Pi - Initial state probability (size Mx1)
%               Z - Pilot Markov transition probability matrix (size MxM) 
%           control - optional struct variable with fields: 
%                   'eps': tolerance for EM termination; defaults to 1e-8
%                   'ItrNo': number of EM iterations; dfaults to 100 
%                   'beta0': initial inverse temperature parameter for deterministic annealing; default 1 
%                   'betarate': decay rate for temperature; default 1 
%                   'safe': if true, regularizes variance matrices to be well-conditioned 
%                    before taking inverse. If false, no regularization (faster but less safe)
%                   'abstol': absolute tolerance for eigenvalues in matrix inversion (only effective if safe = true)
%                   'reltol': relative tolerance for eigenvalues in matrix inversion
%                    = inverse condition number (only effective if safe = true)
%            equal - optional struct variable with fields:
%                   'A': if true, VAR transition matrices A(l,j) are equal across regimes j=1,...,M
%                   'C': if true, observation matrices C(j) are equal across regimes
%                   'Q': if true, VAR innovation matrices Q(j) are equal across regimes
%                   'mu': if true, initial mean state vectors mu(j) are equal across regimes
%                   'Sigma': if true, initial variance matrices Sigma(j) are equal across regimes
%            fixed - optional struct variable with fields 'A','C','Q','R','mu','Sigma'.
%                   If not empty, each field must contain a matrix with 2 columns, the first for 
%                   the location of fixed coefficients and the second for their values. 
%            scale - optional struct variable with fields:
%                   'A': upper bound for norm of eigenvalues of A matrices. Must be in (0,1).
%                   'C': value of the (euclidean) column norms of the matrices C(j). Must be positive.
%                   
% Outputs:  Mf - State probability estimated by switching Kalman Filter (size MxT)
%           Ms - State probability estimated by switching Kalman Smoother (size MxT)
%           Sf - Estimated states (Kalman Filter) 
%           Ss - Estimated states (Kalman Smoother) 
%           xf - Filtered state vector (size rxT)
%           xs - Smoothed state vector (size MxT)
%           outpars - 'struct'
%               A - Estimated system matrix (size rxrxpxM)
%               C - Estimated observation matrix (size Nxr)
%               Q - Estimated state noise covariance (size rxrxM)
%               R - Estimated observation noise covariance (size NxN)
%               mu - Estimated initial mean of state vector (size rxM)
%               Sigma - Estimated initial variance of state vector (size rxrxM)
%           LL  - Sequence of log-likelihood values
%                    
% Author:   David Degras, david.degras@umb.edu
%           University of Massachusetts Boston
%
% Contributors: Ting Chee Ming, cmting@utm.my
%           Siti Balqis Samdin
%           Centre for Biomedical Engineering, Universiti Teknologi Malaysia.
%               
% Version:  January 8, 2021
%--------------------------------------------------------------------------





%-------------------------------------------------------------------------%
%                           Initialization                                %
%-------------------------------------------------------------------------%

narginchk(4,9);

% Data dimensions
[N,T] = size(y);

% Center data
y = y - mean(y,2);

% x(t,j): state vector for j-th process at time t (size r)
% x(t) = x(t,1),...,x(t,M): state vector for all processes at time t (size M*r)
% X(t,j) = x(t,j),...,x(t-p+1,j)): state vector for process j at times t,...,t-p+1 (size p*r)
% X(t) = X(t,1),...,X(t,M): state vector for all processes at times t,...,t-p+1 (size M*p*r)
% Assumption: initial vectors x(1),...,x(1-p+1) are iid ~ N(mu,Sigma)


%@@@@@ Initialize optional arguments if not specified

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



% Trivial case M = 1
if M == 1
    if ~exist('pars','var')
        pars = [];
    end
    S = ones(1,T); Mf = S; Ms = S; Sf = S; Ss = S;
    [xf,xs,outpars,LL] = fast_obs(y,M,p,r,S,pars,control,equal,fixed,scale);
    return
end



%@@@@ Initialize estimators @@@@%
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

if any(structfun(@isempty,pars0)) 
    pars = init_obs(y,M,p,r,pars0,control,equal,pars0,scale);
end

[pars,control,equal,fixed,scale,skip] = ... 
    preproc_obs(M,N,p,r,pars,control,equal,fixed,scale);

abstol = control.abstol;
reltol = control.reltol;
betarate = control.betarate;
eps = control.eps;
ItrNo = control.ItrNo;
verbose = control.verbose;
safe = control.safe;



% Initial parameters 'A','C',... are expanded from r-space (x(t)) to
% (p*r)-space (X(t)). The new parameters 'Ahat','Chat',... have dimensions:
% Ahat: (p*r)x(p*r)xM, Chat: Nx(p*r)xM, Qhat: (p*r)x(p*r)xM, Rhat: NxN,
% muhat: (p*r)xM, Sigmahat: (p*r)x(p*r)xM. These parameters respect the
% fixed coefficients and equality constraints (either default values or
% user-specified ones).

% The structure 'fixed' has fields 'fixed.A',... (one field per model
% parameter). Each field is a matrix with two columns containing the
% locations and values of fixed coefficients in the corresponding
% parameter.

% The structure 'equal' has fields 'equal.A',... representing equality
% constraints on parameters across regimes j=1,...,M. By default, equality
% constraints are: A: false, C: false, Q: false, (R: not applicable), mu:
% true, Sigma: true. In other words, only the initial parameters mu and
% Sigma are assumed to be common across regimes. If provided, user values
% will override default values.

% Various control parameters ('eps','ItrNo',...) are set either to their
% default values or to user-specified values through argument 'control'.




%@@@@@ Initialize other quantities @@@@@%

LL = zeros(1,ItrNo); % Log-likelihood
LLbest = -Inf; % best attained log-likelihood 
LLflag = 0; % counter for monitoring progress of log-likelihood 
sum_yy = y * y.'; % sum(t=1:T) y(t)*y(t)'
beta = control.beta0; % initial temperature for deterministic annealing 



% Function for switching Kalman filtering and smoothing
if p == 1 
        skfs_fun = @skfs_p1_obs;
else
    skfs_fun = @skfs_obs;
end




for i=1:ItrNo
    
   

%-------------------------------------------------------------------------%
%                                 E-step                                  %
%-------------------------------------------------------------------------%



    % Kim/Hamilton filtering and smoothing
    [Mf,Ms,xf,xs,x0,P0,L,sum_CP,sum_MP,sum_Ms2,sum_Mxy,sum_P,sum_Pb] = ... 
        skfs_fun(y,M,p,pars,beta,safe,abstol,reltol);
            
    % Log-likelihood
    LL(i) = L; 
    if verbose
        fprintf('Iteration-%d   Log-likelihood = %g\n',i,LL(i));
        Qval = Q_obs(pars,Ms,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,sum_P,...
            sum_Pb,sum_yy,x0);
        fprintf('Q-function before M-step = %g\n',Qval);
    end
        
    % Check if current solution is best to date
    if i == 1 || LL(i) > LLbest
        LLbest = LL(i);
        outpars = pars;
        Mfbest = Mf;
        Msbest = Ms;
        xfbest = xf;
        xsbest = xs;
    end
     
    % Monitor progress of log-likelihood
    if i>1 && LL(i)-LL(i-1) < eps * abs(LL(i-1))
        LLflag = LLflag + 1;
    else
        LLflag = 0;
    end  
    % Terminate EM algorithm if no sufficient reduction in log-likelihood
    % for 10 successive iterations
    if LLflag == 5
        break;
    end
        
    % Update inverse temperature parameter (DAEM)
    beta = min(beta * betarate, 1);
        
    
    
    
%-------------------------------------------------------------------------%
%                               M-step                                    %
%-------------------------------------------------------------------------%



    pars = M_obs(pars,Ms,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,sum_P,...
        sum_Pb,sum_yy,x0,control,equal,fixed,scale,skip);

 
    % Evaluate and display Q-function value if required 
    if verbose
        Qval = Q_obs(pars,Ms,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,sum_P,...
            sum_Pb,sum_yy,x0);
        fprintf('Q-function after M-step  = %g\n',Qval);
    end
        
    
end % END MAIN LOOP



%-------------------------------------------------------------------------%
%                               Output                                    %
%-------------------------------------------------------------------------%

% Return best estimates (i.e. with highest log-likelihood) 
% after reshaping them to original size
outpars.A = reshape(outpars.A,r,r,p,M);
Mf = Mfbest;
Ms = Msbest;
[~,Sf] = max(Mf);
[~,Ss] = max(Ms);
xf = xfbest;
xs = xsbest;

LL = LL(1:i);

end

