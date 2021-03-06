function [Aboot,Qboot,muboot,Sigmaboot,Piboot,Zboot,LLboot] = ... 
    bootstrap_var(pars,T,B,control,equal,fixed,scale,parallel)

%-------------------------------------------------------------------------%
%
% Title:    Parametric bootstrap in regime-switching state-space models
%           (switching vector autoregressive model)
%
% Purpose:  This function performs parametric bootstrap of the switching
%           vector autoregresssive (VAR) model 
%           x(t) = A(1,S(t)) x(t-1) + ... + A(p,S(t)) x(t-p) + v(t,S(t))
%           where x(t) is the observed measurement vector, S(t) is a latent 
%           variable indicating the current (unobserved) regime, and
%           v(t,S(t)) is a noise term. The model parameters are estimated
%           by maximum likelihood (EM algorithm). 
%  
% Usage:    [Aboot,Qboot,muboot,Sigmaboot,Piboot,Zboot,LLboot] = ... 
%               bootstrap_var(pars,T,B,control,equal,fixed,scale,parallel)
%
%  Inputs:  pars - structure of estimated model parameters with fields 'A',
%               'Q','mu','Sigma','Pi', and 'Z'. Typically, this  
%               structure is obtaining by calling init_var, switch_var, 
%               fast_var, or reestimate_var
%           T - Time series length
%           B - Number of bootstrap replicates (default = 500)
%           control - optional struct variable with fields: 
%                   eps tolerance for EM termination; default = 1e-8
%                   'ItrNo': number of EM iterations; default = 100 
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
%            equal - optional structure with fields:
%               A - if true, estimates of transition matrices A(l,j) 
%                       are equal across regimes j=1,...,M. Default = false
%               Q - if true, estimates of innovation matrices Q(j) are
%                       equal across regimes. Default = false
%               mu - if true, estimates of initial mean state vectors 
%                       mu(j) are equal across regimes. Default = true
%               Sigma - if true, estimates of initial variance matrices 
%                       Sigma(j) are equal across regimes. Default = true
%            fixed - optional struct variable with fields 'A','Q','mu',
%                   'Sigma', 'Pi', 'Z'. If not empty, each field must be an 
%                   array of the same dimension as the parameter. Numeric 
%                   values in the array are interpreted as fixed coefficients 
%                   whereas NaN's represent free (unconstrained) coefficients. 
%            scale - optional structure with field:
%                   'A': upper bound on norm of eigenvalues of A matrices. 
%                       Must be in (0,1) to guarantee stationarity of state 
%                       process.
%                   
%                   
% Outputs:  Aboot - Bootstrap distribution of A (size rxrxpxMxB)
%           Qboot - Bootstrap distribution of Q (size rxrxMxB)
%           muboot - Bootstrap distribution of mu (size rxMxB) 
%           Sigmaboot - Bootstrap distribution of Sigma (size rxrxMxB)
%           Piboot - Bootstrap distribution of Pi (size MxB)
%           Zboot - Bootstrap distribution of Z (size MxMxB)
%           LLboot - Bootstrap distribution of attained maximum log-
%           likelihood (size 1xB)
%                    
% Author:   David Degras, david.degras@umb.edu
%           University of Massachusetts Boston
%
%-------------------------------------------------------------------------%


% Check number of arguments
narginchk(3,8);

% Initialize missing arguments if needed
if ~exist('B','var')
    B = 500;
end
if ~exist('control','var') ||  ~isstruct(control)
    control = struct('verbose',false);
end
if isfield(control,'verbose')
    verbose = control.verbose;
else
    verbose = false;
end
if ~exist('equal','var')
    equal = [];
end
if ~exist('fixed','var')
    fixed = [];
end
if ~exist('scale','var')
    scale = [];
end
if ~exist('parallel','var')
    parallel = false;
end

% Model dimensions
[r,~,p,M] = size(pars.A);

% Bootstrap estimates
Aboot = zeros(r,r,p,M,B);
Qboot = zeros(r,r,M,B);
muboot = zeros(r,M,B);
Sigmaboot = zeros(r,r,M,B);
Piboot = zeros(M,B);
Zboot = zeros(M,M,B);
LLboot = zeros(1,B);
warning('off');

% Set up parallel pool if needed
if parallel     
    pool = gcp('nocreate');
    if isempty(pool)
        pool = gcp;
    end
    poolsize = pool.NumWorkers;
    control.verbose = false;
else
    poolsize = 0;
end
 
% Auxiliary quantities
LQ = zeros(r,r,M);
for j = 1:M
    LQ(:,:,j) = chol(pars.Q(:,:,j),'lower');
end

parfor (b=1:B, poolsize) 
    
    % Initialize progress bar if required
    if verbose && parallel  
        parfor_progress(B);
    end
    
    % Temporary variables
    pars1 = pars;
    A = reshape(pars1.A,r,p*r,M);
    LQ_ = LQ;
    x = zeros(r,T);
    Xtm1 = [];
    cumZ = cumsum(pars1.Z,2);
        
    % Parametric bootstrap
    for t=1:T       
        if t == 1
            c = cumsum(Pi); c(M) = 1;
        else        
            Stm1 = St;
            c = cumZ(Stm1,:);
        end          
        St = sum(rand(1) > c) + 1;        
        if t <= p
            x(:,t) = mvnrnd(pars1.mu(:,St)',pars1.Sigma(:,:,St))';
            Xtm1 = [x(:,t) ; Xtm1];
        else
            vt = LQ_(:,:,St) * randn(N,1);
            x(:,t) = A(:,:,St) * Xtm1 + vt;
            Xtm1 = [x(:,t) ; Xtm1(1:(p-1)*r)];               
        end        
    end       
        
    % EM algorithm 
    [~,~,~,~,Ab,Qb,mub,Sigmab,Pib,Zb,LL] = ... 
            switch_var(x,M,p,pars,control,equal,fixed,scale);   
    Aboot(:,:,:,:,b) = Ab;
    Qboot(:,:,:,b) = Qb;
    muboot(:,:,b) = mub;
    Sigmaboot(:,:,:,b) = Sigmab;
    Piboot(:,b) = Pib;
    Zboot(:,:,b) = Zb;
    LLboot(b) = max(LL);

    % Display progress if required
    if verbose && parallel
        parfor_progress;
    end
end
    
if verbose && parallel
    parfor_progress(0);
end

end

    