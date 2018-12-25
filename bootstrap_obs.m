function [Aboot,Cboot,Qboot,Rboot,muboot,Sigmaboot,Piboot,Zboot,LLboot] = ... 
    bootstrap_obs(A,C,Q,R,mu,Sigma,Pi,Z,T,B,control,equal,fixed,scale,parallel)

%-------------------------------------------------------------------------%
%
% Title:    Parametric bootstrap in regime-switching state-space models
%           (switching observations)
%
% Purpose:  This function performs parametric bootstrap of the switching
%           observations model 
%           y(t) = C(S(t)) x(t,S(t)) + w(t)
%           x(t,j) = A(1,j) x(t-1,j) + ... + A(p,j) x(t-p,j) + v(t,j)
%           where y indicates the observed measurement vector, x the 
%           unbserved state vector, and S the current (unobserved) regime.
%           The terms v and w are noise vectors. All model parameters are 
%           (re)estimated by maximum likelihood. ML estimation is 
%           implemented via the EM algorithm. 
%  
% Usage:    [Aboot,Cboot,Qboot,Rboot,muboot,Sigmaboot,Piboot,Zboot,LLboot]...
%               = bootstrap_obs(A,C,Q,R,mu,Sigma,Pi,Z,T,B,control,equal,...
%                   fixed,scale,parallel)
%
% Inputs:   A - Estimate of transition matrix A(l,j) for l=1:p (lag) and  
%               j=1:M (regime) (size rxrxpxM)
%           C - Estimate of observation matrix C(j), j=1:M (size NxrxM)
%           Q - Estimate of noise covariance Q(j)=Cov(v(t)|S(t)=j), j=1:M
%               (size rxrxM)   
%           R - Estimate of observation noise covariance R = V(w(t))
%           mu - Estimate of initial mean mu(j)=E(x(1,j)), j=1:M (size rxM) 
%           Sigma - Estimate of initial variance Sigma(j)=V(x(1,j)), j=1:M
%               (size rxrxM) 
%           Pi - Estimate of initial probability Pi(j)=P(S(1)=j), j=1:M 
%               (size Mx1)
%           Z - Estimate of transition probability Z(i,j)=P(S(t)=j|S(t-1)=i)
%               i,j=1:M (size MxM) 
%           T - Time series length
%           B - Number of bootstrap replicates (default = 100)
%           control - optional struct variable with fields: 
%                   'eps': tolerance for EM termination; default = 1e-8
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
%                   'A': if true, estimates of transition matrices A(l,j) 
%                       are equal across regimes j=1,...,M. Default = false
%                   'C': if true, observation matrices C(j) are equal across 
%                       regimes (default = false)
%                   'Q': if true, estimates of innovation matrices Q(j) are
%                       equal across regimes. Default = false
%                   'mu': if true, estimates of initial mean state vectors 
%                       mu(j) are equal across regimes. Default = true
%                   'Sigma': if true, estimates of initial variance matrices 
%                       Sigma(j) are equal across regimes. Default = true
%            fixed - optional struct variable with fields 'A','C','Q','R',
%                   'mu','Sigma'. If not empty, each field must be an array
%                   of the same dimension as the parameter. Numeric values 
%                   in the array are interpreted as fixed coefficients 
%                   whereas NaN's represent free coefficients. For example,
%                   a diagonal structure for 'R' would be specified as
%                   fixed.R = diag(NaN(N,1)). 
%            scale - optional structure with fields:
%                   'A': upper bound on norm of eigenvalues of A matrices. 
%                       Must be in (0,1) to guarantee stationarity of state 
%                       process.
%                   'C': value of the (Euclidean) column norms of the 
%                       matrices C(j). Must be positive.
%                   
%                   
% Outputs:  Aboot - Bootstrap distribution of A (size rxrxpxMxB)
%           Cboot - Bootstrap distribution of C (size NxrxMxB)
%           Qboot - Bootstrap distribution of Q (size rxrxMxB)
%           Rboot - Bootstrap distribution of R (size NxNxB)
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
narginchk(9,15);

% Initialize missing arguments if needed
if ~exist('B','var')
    B = 100;
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
M = numel(Pi);
N = size(C,1);
p = size(A,3);
r = size(mu,1);

% Bootstrap estimates
Aboot = zeros(r,r,p,M,B);
Cboot = zeros(N,r,M,B);
Qboot = zeros(r,r,M,B);
Rboot = zeros(N,N,B);
muboot = zeros(r,M,B);
Sigmaboot = zeros(r,r,M,B);
Piboot = zeros(M,B);
Zboot = zeros(M,M,B);
LLboot = zeros(B,1);
warning('off');

Amat = reshape(A,[r,p*r,M]);


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
 
% Choleski decompositions
LQ = zeros(r,r,M);
for j = 1:M
    LQ(:,:,j) = chol(Q(:,:,j),'lower');
end
LR = chol(R,'lower');


parfor (b=1:B, poolsize) 
    
    % Initialize progress bar if required
    if verbose && parallel  
        parfor_progress(B);
    end
    
    % Temporary variables
    Amat_ = Amat;
    C_ = C;
    mu_ = mu;
    Sigma_ = Sigma;
    Z_ = Z;
    LQ_ = LQ;
    LR_ = LR;
    Xtm1 = zeros(p*r,M);
    y = zeros(N,T);
    
    % Parametric bootstrap
    for t = 1:T
        if t == 1
            c = cumsum(Pi);
            St = M + 1 - sum(rand(1) <= c);
            for j = 1:M
                Xtm1(:,j) = reshape(mvnrnd(mu_(:,j)',Sigma_(:,:,j),p),p*r,1);
            end
            xt = Xtm1(1:p,:);
        else
            Stm1 = St;
            c = cumsum(Z_(Stm1,:));              
            St = M + 1 - sum(rand(1) <= c);        
            for j = 1:M             
                vtj = LQ_(:,:,j) * randn(r,1);
                xt(:,j) = Amat_(:,:,j) * Xtm1(:,j) + vtj;
            end
            Xtm1 = vertcat(xt,Xtm1(1:(p-1)*r,:));
        end
        y(:,t) = C_(:,:,St) * xt(:,St) + LR_ * randn(N,1);
    end   

    % EM algorithm
    [~,~,~,~,~,~,Ab,Cb,Qb,Rb,mub,Sigmab,Pib,Zb,LL] = ... 
            switch_obs(y,M,p,r,A,C,Q,R,mu,Sigma,Pi,Z,control,equal,fixed,scale);   
    Aboot(:,:,:,:,b) = Ab;
    Cboot(:,:,b) = Cb;
    Qboot(:,:,:,b) = Qb;
    Rboot(:,:,b) = Rb;
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


    