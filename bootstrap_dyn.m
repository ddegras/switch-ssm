function [outpars,LL] = ... 
    bootstrap_dyn(pars,T,B,opts,control,equal,fixed,scale,parallel,match)

%-------------------------------------------------------------------------%
%
% Title:    Parametric bootstrap in regime-switching state-space models
%           (switching dynamics)
%
% Purpose:  This function performs parametric bootstrap of the switching
%           dynamics model 
%           y(t) = C x(t) + w(t)
%           x(t) = A(1,S(t)) x(t-1) + ... + A(p,S(t)) x(t-p) + v(t)
%           where y indicates the observed measurement vector, x the 
%           unbserved state vector, and S the current (unobserved) regime.
%           The terms v and w are noise vectors. All model parameters are 
%           (re-)estimated by maximum likelihood. ML estimation is 
%           implemented via the EM algorithm. 
%  
% Usage:    [outpars,LL] = bootstrap_dyn(pars,T,B,control,equal,...
%                   fixed,scale,parallel,match)
%
% Inputs:   pars - structure of estimated model parameters with fields 'A',
%               'C','Q','R','mu','Sigma','Pi', and 'Z'. Typically, this  
%               structure is obtaining by calling init_dyn, switch_dyn, 
%               fast_dyn, or reestimate_dyn
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
%           equal - optional structure with fields:
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
%           fixed - optional struct variable with fields 'A','C','Q','R',
%                   'mu','Sigma'. If not empty, each field must be an array
%                   of the same dimension as the parameter. Numeric values 
%                   in the array are interpreted as fixed coefficients 
%                   whereas NaN's represent free coefficients. For example,
%                   a diagonal structure for 'R' would be specified as
%                   fixed.R = diag(NaN(N,1)). 
%           scale - optional structure with fields:
%                   'A': upper bound on norm of eigenvalues of A matrices. 
%                       Must be in (0,1) to guarantee stationarity of state 
%                       process.
%                   'C': value of the (Euclidean) column norms of the 
%                       matrices C(j). Must be positive.
%           match - parameter to use to match the bootstrap replicates to
%                   the maximum likelihood estimate: 'A', 'AQ', 'COV', 
%                   'COR'. Default = 'COV'   
%                   
%                   
% Outputs:  outpars - struct with fields 
%               'A': Bootstrap distribution of A (size rxrxpxMxB)
%               'C': Bootstrap distribution of C (size NxrxB)
%           	'Q': Bootstrap distribution of Q (size rxrxMxB)
%               'R': Bootstrap distribution of R (size NxNxB)
%           	'mu': Bootstrap distribution of mu (size rxMxB) 
%           	'Sigma': Bootstrap distribution of Sigma (size rxrxMxB)
%           	'Pi': Bootstrap distribution of Pi (size MxB)
%           	'Z': Bootstrap distribution of Z (size MxMxB)
%           LL - Bootstrap distribution of attained log-likelihood (1xB)
%                    
% Author:   David Degras, david.degras@umb.edu
%           University of Massachusetts Boston
%
%-------------------------------------------------------------------------%


% Check number of arguments
narginchk(3,10);

% Initialize missing arguments if needed
if ~exist('B','var') || isempty(B)
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
if ~exist('parallel','var') || ~islogical(parallel)
    parallel = true;
end
if ~exist('opts','var')
    opts = [];
end
if ~exist('match','var') || isempty(match)
    match = 'COV';
end
assert(ismember(match,{'A','AQ','COV','COR','no'}))

% Model dimensions
M = numel(pars.Pi);
[N,r] = size(pars.C);
p = size(pars.A,3);

% Bootstrap estimates
Aboot = NaN(r,r,p,M,B);
Cboot = NaN(N,r,B);
Qboot = NaN(r,r,M,B);
Rboot = NaN(N,N,B);
muboot = NaN(r,M,B);
Sigmaboot = NaN(r,r,M,B);
Piboot = NaN(M,B);
Zboot = NaN(M,M,B);
LLboot = NaN(B,1);
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
 
% Initialize progress bar if required
if verbose  
    parfor_progress(B);
end


parfor (b=1:B, poolsize) 

    % Resample data by parametric bootstrap
    % Try to ensure that each regime occurs at least once
    count = 0; Meff = 0;
    while count < 20 && Meff < M
        count = count + 1;
        [y,S] = simulate_dyn(pars,T);
        Meff = numel(unique(S));
    end
    
    % Run EM algorithm  
    try
        pars0 = init_dyn(y,M,p,r,opts,control,equal,fixed,scale);
        [~,~,~,~,~,~,parsboot,LL] = ... 
           switch_dyn(y,M,p,r,pars0,control,equal,fixed,scale); 
    catch
        continue
    end
    
    % Store results
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

outpars = struct('A',Aboot, 'C',Cboot, 'Q',Qboot, 'R',Rboot, 'mu',muboot, ...
    'Sigma',Sigmaboot, 'Pi',Piboot, 'Z',Zboot);
LL = LLboot;

% Match bootstrap replicates to MLE
if ~strcmp(match,'no')
    outpars = bootstrap_match(outpars,pars,match);
end

if verbose && parallel
    parfor_progress(0);
end




    