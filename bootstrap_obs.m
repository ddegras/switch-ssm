function [Aboot,Cboot,Qboot,Rboot,muboot,Sigmaboot,Piboot,Zboot,LLboot] = ... 
    bootstrap_obs(pars,T,B,control,equal,fixed,scale,parallel)

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
%  Inputs:   pars - structure of estimated model parameters with fields 'A',
%               'C','Q','R','mu','Sigma','Pi', and 'Z'. Typically, this  
%               structure is obtaining by calling init_obs, switch_obs, 
%               fast_obs, or reestimate_obs
%           T - Time series length
%           B - Number of bootstrap replicates (default = 500)
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
M = numel(pars.Pi);
[N,r] = size(pars.C);
p = size(pars.A,3);

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
    LQ(:,:,j) = chol(pars.Q(:,:,j),'lower');
end
LR = chol(pars.R,'lower');


parfor (b=1:B, poolsize) 
    
    % Initialize progress bar if required
    if verbose && parallel  
        parfor_progress(B);
    end
    
    % Temporary variables
    pars1 = pars;
    A = reshape(pars1.A,r,p*r,M);
    C = pars1.C;
    LQ1 = LQ;
    LR1 = LR;
    Xtm1 = zeros(p*r,M);
    y = zeros(N,T);
    
    % Parametric bootstrap
    cumZ = cumsum(pars1.Z,2); cumZ(:,M) = 1;
    for t = 1:T
        if t == 1
            c = cumsum(pars1.Pi); c(M) = 1;
            St = sum(rand(1) > c) + 1;
            for j = 1:M
                Xtm1(:,j) = mvnrnd(pars1.mu(:,j)',pars1.Sigma(:,:,j),p);
                Xtm1(:,j) = reshape(Xtm1(:,j)',p*r,1);
            end
            xt = Xtm1(1:r,:);
        else
            Stm1 = St;
            c = cumZ(Stm1,:);              
            St = sum(rand(1) > c) + 1;        
            for j = 1:M             
                vtj = LQ1(:,:,j) * randn(r,1);
                xt(:,j) = A(:,:,j) * Xtm1(:,j) + vtj;
            end
            Xtm1 = vertcat(xt,Xtm1(1:(p-1)*r,:));
        end
        y(:,t) = C(:,:,St) * xt(:,St) + LR1 * randn(N,1);
    end   

    % EM algorithm
    [~,~,~,~,~,~,parsboot,LL] = ... 
            switch_obs(y,M,p,r,pars,control,equal,fixed,scale);   
    Aboot(:,:,:,:,b) = parsboot.A;
    Cboot(:,:,b) = parsboot.C;
    Qboot(:,:,:,b) = parsboot.Q;
    Rboot(:,:,b) = parsboot.R;
    muboot(:,:,b) = parsboot.mu;
    Sigmaboot(:,:,:,b) = parsboot.Sigma;
    Piboot(:,b) = parsboot.Pi;
    Zboot(:,:,b) = parsboot.Z;
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


    