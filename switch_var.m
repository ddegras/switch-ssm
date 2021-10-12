function [Mf,Ms,Sf,Ss,outpars,LL] = ... 
    switch_var(y,M,p,pars,control,equal,fixed,scale)

%--------------------------------------------------------------------------
% Title:    Parameter estimation and inference in state-space models with 
%           regime switching (switching vector autoregressive)
%
% Purpose:  Infer hidden state vectors and regimes by switching Kalman 
%           filtering/smoothing (aka Hamilton filtering or Kim filtering) 
%           and estimate model parameters by maximum likelihood (EM algorithm).
%
% Usage:    [Mf,Ms,Sf,Ss,outpars,LL] = ... 
%               switch_var(y,M,p,pars,control,equal,fixed,scale)
% 
% Inputs:   y - Time series (size NxT)
%           M - number of regimes (values) for switching variable S(t) 
%           p - order of VAR model for y(t)
%           pars - structure with fields
%               A - Initial estimate of VAR matrices A(l,j) in model  
%                   y(t) = sum(l=1:p) A(l,j) y(t-l) + v(t) conditional on 
%                   S(t)=j, j=1:M (size rxrxpxM)  
%               Q - Initial estimate of system noise covariance Q(j) =
%                   V(v(t)|S(t)=j), j=1:M (size rxrxM)
%               mu - Initial estimate of mean state mu(j)=E(y(1)|S(1)=j),
%                   j=1:M (size rxM) 
%               Sigma - Initial estimate of state covariance Sigma(j) = 
%                   V(y(1)|S(1)=j), j=1:M (size rxrxM) 
%               Pi - Initial estimate of probability Pi(j)=P(S(1)=j), j=1:M (size Mx1)
%               Z - Initial estimate of transition probabilities Z(i,j) = 
%               P(S(t)=j|S(t-1)=i), i,j=1:M (size MxM) 
%           control - optional struct variable with fields: 
%               eps - tolerance for EM termination; default = 1e-8
%               ItrNo - number of EM iterations; default = 100 
%               beta0 - initial inverse temperature parameter for 
%                       deterministic annealing; must be in [0,1]
%                       default = 1 (regular EM)
%               betarate - decay rate for temperature; must be >= 1; default = 1 
%               safe - if true, regularizes variance matrices in 
%                       switching Kalman filtering to be well-conditioned 
%                       before inversion. If false, no regularization (faster 
%                       but less safe)
%                abstol - absolute tolerance for eigenvalues before
%                       inversion of covariance matrices (eigenvalues less 
%                       than abstol are set to this value) 
%                reltol - relative tolerance for eigenvalues before 
%                       inversion of covariance matrices (eigenvalues less
%                       than (reltol * largest eigenvalue) are set to this
%                       value)
%           equal - optional struct variable with fields:
%               A - if true, estimates of transition matrices A(l,j) 
%                       are equal across regimes j=1,...,M. Default = false
%               Q - if true, estimates of innovation matrices Q(j) are
%                       equal across regimes. Default = false
%               mu - if true, estimates of initial mean state vectors 
%                       mu(j) are equal across regimes. Default = true
%               Sigma - if true, estimates of initial variance matrices 
%                       Sigma(j) are equal across regimes. Default = true
%           fixed - optional struct variable with fields 'A','C','Q','R',
%                   'mu','Sigma'. If not empty, each field must be an array
%                   of the same dimension as the parameter. Numeric values 
%                   in the array are interpreted as fixed coefficients whereas 
%                   NaN's represent free coefficients. For example, a
%                   diagonal structure for 'R' would be specified as
%                   fixed.R = diag(NaN(N,1)). 
%           scale - optional struct variable with field:
%               A - upper bound on norm of eigenvalues of A matrices. 
%                   Must be in (0,1) to guarantee stationarity of state process.
%
% Outputs:  Mf - State probability estimated by switching Kalman Filter (size MxT)
%           Ms - State probability estimated by switching Kalman Smoother (size MxT)
%           Sf - Estimated states (Kalman Filter) 
%           Ss - Estimated states (Kalman Smoother) 
%           outpars - structure of parameter estimates with fields:
%               A - Estimated system matrix (size rxrxpxM)
%               Q - Estimated state noise covariance (size rxrxM)
%               mu - Estimated initial mean of state vector (size rxM)
%               Sigma - Estimated initial variance of state vector (size rxrxM)
%           LL  - Sequence of log-likelihood values
%                    
% Author:   David Degras (University of Massachusetts Boston)
%
% Contributors: Chee Ming Ting (Monash University Malaysia)
%           Siti Balqis Samdin (Xiamen University Malaysia)
%
%--------------------------------------------------------------------------





%-------------------------------------------------------------------------%
%                           Initialization                                %
%-------------------------------------------------------------------------%


narginchk(3,8);


warning('off','MATLAB:singularMatrix');
warning('off','MATLAB:nearlySingularMatrix');
warning('off','MATLAB:illConditionedMatrix');


% Centering
y = y - mean(y,2);

% Data dimensions
[N,T] = size(y);
r = N;

% Initialize optional arguments if not specified
fixed0 = struct('C',eye(r),'R',1e-20 * eye(r));
if exist('fixed','var') && isstruct(fixed)
    fixed.C = fixed0.C;
    fixed.R = fixed0.R;
else
    fixed = fixed0;
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

% Special case M = 1
if M == 1
    Mf = ones(1,T); Ms = Mf; Sf = Mf; Ss = Mf;
    [outpars,LL] = fast_var(y,M,p,Ss,control,equal,fixed,scale);
    outpars.Pi = 1; outpars.Z = 1;
    return
end



%@@@@ Initialize estimators @@@@%
pars0 = struct('A',[], 'C',fixed.C, 'Q',[], 'R',fixed.R, 'mu',[], ... 
    'Sigma',[], 'Pi',[], 'Z',[]);
if exist('pars','var') && isstruct(pars)
    name = fieldnames(pars0);
    for i = 1:numel(name) 
        if isfield(pars,name{i})
            pars0.(name{i}) = pars.(name{i});
        end
    end
end

if any(structfun(@isempty,pars0)) 
    pars0 = init_dyn(y,M,p,r,[],control,equal,pars0,scale);
end

[pars,control,equal,fixed,scale,skip] = ... 
    preproc_dyn(M,N,p,r,pars0,control,equal,fixed,scale);
pars = rmfield(pars,{'C','R'});

% The structure 'fixed' has fields 'fixed.A',... (one field per model
% parameter). Each field is a matrix with two columns containing the
% locations and values of fixed coefficients in the corresponding
% parameter.

% The structure 'skip' specified whether a parameter (A,C,Q...) is entirely
% fixed, in which case its update in the M-step can be skipped altogether

% We assume that conditional on S(1:p), the initial vectors y(1),...,y(p)
% are independent and that conditional on S(t)=j (t=1:p), y(t) ~ N(mu(j),Sigma(j))



%@@@@ Initialize other quantities

ItrNo = control.ItrNo;
LL = zeros(ItrNo,1); % Log-likelihood
LLbest = -Inf; % best attained value of log-likelihood 
LLflag = 0; % counter for convergence of of log-likelihood 
beta = control.beta0; % initial temperature for deterministic annealing 
betarate = control.betarate;
eps = control.eps;
verbose = control.verbose;


 
% Quantities required for Q-function and for M-step
X = zeros(p*r,T-p);
for k = 1:p
    X((k-1)*r+1:k*r,:) = y(:,p+1-k:T-k);
end
sum_MP = zeros(r,r,M);
sum_MPb = zeros(p*r,p*r,M);
sum_MCP = zeros(r,p*r,M);


    
for i=1:ItrNo
    
   

%-------------------------------------------------------------------------%
%                               E-step                                    %
%-------------------------------------------------------------------------%


 
    % Kim/Hamilton filtering and smoothing
    [Mf,Ms,L,sum_Ms2] = skfs_var(y,M,p,pars,beta);

    % Log-likelihood
    LL(i) = L; 

    for j = 1:M
        yj = y(:,p+1:T) .* sqrt(Ms(j,p+1:T));
        sum_MP(:,:,j) = (yj * yj.');
        Xj = X .* sqrt(Ms(j,p+1:T));
        sum_MPb(:,:,j) = (Xj * Xj.');
        sum_MCP(:,:,j) = yj * Xj.';
    end
    clear Xj yj
    
    % Evaluate Q-function before M-step and display LL & Q if required        
    if verbose
        fprintf('Iteration-%d   Log-likelihood = %g\n',i,LL(i));
        Qval = Q_var(pars,Ms,sum_MCP,sum_MP,sum_MPb,sum_Ms2,y);
        fprintf('Q-function before M-step = %g\n',Qval);
    end
       
    % Check if current solution is best to date
    if (LL(i) > LLbest)
        LLbest = LL(i);
       outpars = pars;
        Mfbest = Mf;
        Msbest = Ms;
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
    beta = beta * betarate;
    if beta > 1
        beta = 1;
    end
    
    
    
%-------------------------------------------------------------------------%
%                               M-step                                    %
%-------------------------------------------------------------------------%


    pars = M_var(pars,Ms,sum_MCP,sum_MP,sum_MPb,...
        sum_Ms2,y,control,equal,fixed,scale,skip);


    % Evaluate and display Q-function if required 
    if verbose
        Qval = Q_var(pars,Ms,sum_MCP,sum_MP,sum_MPb,sum_Ms2,y);
        fprintf('Q-function after M-step  = %g\n',Qval);
    end

    
end % END MAIN LOOP



%-------------------------------------------------------------------------%
%                               Output                                    %
%-------------------------------------------------------------------------%

% Return best estimates (i.e. with highest log-likelihood) 


outpars.A = reshape(outpars.A,[N,N,p,M]);
Mf = Mfbest;
Ms = Msbest;
[~,Sf] = max(Mf);
[~,Ss] = max(Ms);
LL = LL(1:i);




