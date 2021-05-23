function [Mf,Ms,Sf,Ss,xf,xs,outpars,LL] = ... 
    switch_dyn(y,M,p,r,pars,control,equal,fixed,scale)

%--------------------------------------------------------------------------
% Title:    Parameter estimation and inference in state-space models with 
%           regime switching (switching dynamics)
%
% Purpose:  Infer hidden state vectors and regimes by switching Kalman 
%           filtering/smoothing (aka Hamilton filtering or Kim filtering) 
%           and estimate model parameters by maximum likelihood (EM algorithm).
%
% Usage:    [Mf,Ms,Sf,Ss,xf,xs,outpars,LL] = ... 
%               switch_dyn(y,M,p,r,pars,control,equal,fixed,scale)
%
% Inputs:   y - Time series (size NxT with N=#variables and T=#time points)
%           M - number of possible regimes for switching variable S(t) 
%           p - order of VAR model for state vector x(t)
%           r - size of state vector x(t)
%           pars - optional 'struct' of starting values for EM. Fields:
%               A - Initial estimate of VAR matrices A(l,j) in system equation 
%               x(t) = sum(l=1:p) A(l,j) x(t-l) + v(t) conditional on S(t)=j, 
%               j=1:M (size rxrxpxM)  
%               C - Initial estimate of observation matrix C in equation 
%               y(t) = C x(t) + w(t) (size Nxr)
%               Q - Initial estimate of system noise covariance Q(j)=Cov(v(t)|S(t)=j), 
%               j=1:M (size rxrxM)
%               R - Initial estimate of observation noise covariance R=Cov(w(t)) 
%               (size NxN)                  
%               mu - Initial estimate of mean state mu(j)=E(x(1)|S(1)=j), j=1:M  
%               (size rxM) 
%               Sigma - Initial estimate of state covariance Sigma(j)=Cov(x(1,j)), 
%                j=1:M (size rxrxM) 
%               Pi - Initial estimate of probability Pi(j)=P(S(1)=j), j=1:M (size Mx1)
%               Z - Initial estimate of transition probabilities Z(i,j) = 
%               P(S(t)=j|S(t-1)=i), i,j=1:M (size MxM) 
%           control - optional 'struct' of algorithm parameters. Fields: 
%                   'eps': tolerance for EM termination; default = 1e-8
%                   'ItrNo': number of EM iterations; default = 1000 
%                   'beta0': initial inverse temperature parameter for 
%                       deterministic annealing; default = 1 
%                   'betarate': decay rate for temperature; default = 1 
%                   'safe': if true, regularizes variance matrices in 
%                       switching Kalman filtering to be well-conditioned 
%                       before inversion. If false, no regularization (faster 
%                       but less safe)
%                   'abstol': absolute tolerance for eigenvalues before
%                       inversion of covariance matrices (eigenvalues less 
%                       than abstol are set to this value) 
%                   'reltol': relative tolerance for eigenvalues before 
%                       inversion of covariance matrices (eigenvalues less
%                       than (reltol * largest eigenvalue) are set to this
%                       value)
%            equal - optional 'struct' with fields:
%                   'A': if true, estimates of transition matrices A(l,j) 
%                       are equal across regimes j=1,...,M. Default = false
%                   'Q': if true, estimates of innovation matrices Q(j) are
%                       equal across regimes. Default = false
%                   'mu': if true, estimates of initial mean state vectors 
%                       mu(j) are equal across regimes. Default = true
%                   'Sigma': if true, estimates of initial variance matrices 
%                       Sigma(j) are equal across regimes. Default = true
%            fixed - optional 'struct' with fields 'A','C','Q','R',
%                   'mu','Sigma'. If not empty, each field must be an array
%                   of the same dimension as the parameter. Numeric values 
%                   in the array are interpreted as fixed coefficients whereas 
%                   NaN's represent free coefficients. For example, a
%                   diagonal structure for 'R' would be specified as
%                   fixed.R = diag(NaN(N,1)). 
%            scale - optional 'struct' with fields:
%                   'A': upper bound on norm of eigenvalues of A matrices. 
%                       Must be in (0,1) to guarantee stationarity of state process.
%                   'C': value of the (euclidean) column norms of the matrices C(j).
%                       Must be positive.
%                   
% Outputs:  Mf - State probability estimated by switching Kalman Filter (size MxT)
%           Ms - State probability estimated by switching Kalman Smoother (size MxT)
%           Sf - Estimated states (Kalman Filter) 
%           Ss - Estimated states (Kalman Smoother) 
%           xf - Filtered state vector (size rxT)
%           xs - Smoothed state vector (size MxT)
%           outpars - 'struct' of estimated model parameters 
%               A - Estimated system matrix (size rxrxpxM)
%               C - Estimated observation matrix (size Nxr)
%               Q - Estimated state noise covariance (size rxrxM)
%               R - Estimated observation noise covariance (size NxN)
%               mu - Estimated initial mean of state vector (size rxM)
%               Sigma - Estimated initial variance of state vector (size rxrxM)
%               Pi - Estimated initial state probabilities (Mx1)
%               Z - Estimated state transition probabilities (MxM)
%           LL  - Sequence of log-likelihood values
%                    
% Author:   David Degras, david.degras@umb.edu
%           University of Massachusetts Boston
%
% Contributors: Ting Chee Ming, cmting@utm.my
%           Siti Balqis Samdin
%           Centre for Biomedical Engineering, Universiti Teknologi Malaysia.
%               
% Version:  February 16, 2021
%--------------------------------------------------------------------------




% We assume t conditional on S(1)=j, the initial state vectors x(2-p),...,x(1)
% are iid ~ N(mu(j),Sigma(j))



%-------------------------------------------------------------------------%
%                           Initialization                                %
%-------------------------------------------------------------------------%


narginchk(4,9);

warning('off','MATLAB:singularMatrix');
warning('off','MATLAB:nearlySingularMatrix');
warning('off','MATLAB:illConditionedMatrix');

% Data dimensions
N = size(y,1);

% Centering
y = y - mean(y,2);

% 'small' state vector x(t), size = r
% 'big' state vector X(t) = (x(t),...,x(t-p+1)), size = p*r

% Initialize optional arguments if not specified
if ~exist('fixed','var') 
    fixed = [];
end
if ~exist('equal','var') 
    equal = [];
end
if ~exist('control','var') 
    control = [];
end
if ~exist('scale','var') 
    scale = [];
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
pars = pars0;

if any(structfun(@isempty,pars)) 
    pars = init_dyn(y,M,p,r,[],control,equal,fixed,scale);
end

[pars,control,equal,fixed,scale,skip] = ...
    preproc_dyn(M,N,p,r,pars,control,equal,fixed,scale);

abstol = control.abstol;
reltol = control.reltol;
betarate = control.betarate;
eps = control.eps;
ItrNo = control.ItrNo;
verbose = control.verbose;
safe = control.safe;


%@@@@ Initialize other quantities
LL = zeros(1,ItrNo); % Log-likelihood
LLbest = -Inf; % best attained value of log-likelihood 
LLflag = 0; % counter for convergence of of log-likelihood 
sum_yy = y * y.'; % sum(t=1:T) y(t)*y(t)'
beta = control.beta0; % initial temperature for deterministic annealing 



% Function for switching Kalman filtering and smoothing
if p == 1 && r == 1 
        skfs_fun = @skfs_p1r1_dyn;
elseif p == 1
    skfs_fun = @skfs_p1_dyn;    
elseif r == 1
        skfs_fun = @skfs_r1_dyn;
else
    skfs_fun = @skfs_dyn;
end


for i = 1:ItrNo
    
   

%-------------------------------------------------------------------------%
%                               E-step                                    %
%-------------------------------------------------------------------------%




    % Kim/Hamilton filtering and smoothing
    [Mf,Ms,xf,xs,L,MP0,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,sum_P] = ... 
        skfs_fun(y,M,p,r,pars,beta,safe,abstol,reltol);
    
    % Log-likelihood
    LL(i) = L; 
    
    % Needed for Q-function
    sum_xy = xs(1:r,:) * y.';

    % Evaluate Q-function before M-step and display LL & Q if required        
    if verbose
        fprintf('Iteration-%d   Log-likelihood = %g\n',i,LL(i));
        Qval = Q_dyn(pars,MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,...
            sum_P,sum_xy,sum_yy);
        fprintf('Q-function before M-step = %g\n',Qval);
    end
       
    % Check if current solution is best to date
    if (i == 1 || LL(i) > LLbest)
        LLbest = LL(i);
        outpars = pars;
        Mfbest = Mf; Msbest = Ms; 
        xfbest = xf; xsbest = xs;
    end
     
    % Monitor progress of log-likelihood
    if (i>1 &&  LL(i)-LL(i-1) < eps * abs(LL(i-1)))
        LLflag = LLflag + 1;
    else
        LLflag = 0;
    end  
    % Terminate EM algorithm if no sufficient increase in log-likelihood
    % for 10 iterations in a row
    if LLflag == 5
        break;
    end
        
    % Update inverse temperature parameter (DAEM)
    beta = min(beta * betarate, 1);
    
    
    
%-------------------------------------------------------------------------%
%                               M-step                                    %
%-------------------------------------------------------------------------%



    parsold = pars;
    pars = M_dyn(parsold,MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,...
        sum_P,sum_xy,sum_yy,control,equal,fixed,scale,skip);

    % Display Q-function if required 
    if verbose
        Qval = Q_dyn(pars,MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,...
            sum_P,sum_xy,sum_yy);
        fprintf('Q-function after M-step = %g\n',Qval);
    end

    
end % END MAIN LOOP



%-------------------------------------------------------------------------%
%                               Output                                    %
%-------------------------------------------------------------------------%

% Return best estimates (i.e. with highest log-likelihood) 
% after reshaping them in compact form

outpars.A = reshape(outpars.A,[r,r,p,M]); 
Mf = Mfbest; Ms = Msbest;
[~,Sf] = max(Mf); [~,Ss] = max(Ms);
xf = xfbest; xs = xsbest;
LL = LL(1:i);

end

