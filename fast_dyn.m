function [xf,xs,outpars,LL] = ... 
    fast_dyn(y,M,p,r,S,pars,control,equal,fixed,scale)

%--------------------------------------------------------------------------
%
%       PARAMETER ESTIMATION AND INFERENCE IN STATE-SPACE MODEL 
%           WITH SWITCHING DYNAMICS ASSUMING REGIMES KNOWN 
%
% PURPOSE
% FAST_DYN estimates model parameters and infers hidden state vectors 
% by the EM algorithm in state-space models with switching dynamics under
% the assumption that the regime (i.e. switching) variables are known.
% The function can be used to fit the model under a trajectory of regimes 
% that is highly likely. In the case of only one regime (no
% switching), it can also be used to fit the standard linear
% state-space model.
%
% USAGE
%   [xf,xs,outpars,LL] = fast_dyn(y,M,p,r,S,pars,control,equal,fixed,scale)
% 
% INPUTS  
% y:    time series data (dimension NxT)
% M:    number of regimes
% p:    order of VAR model for state vector x(t)
% r:    size of state vector x(t)
% S:    fixed sequence of regimes S(t), t=1:T
% pars: optional structure with fields:
%       'A': Initial estimate of VAR matrices A(l,j) in system equation 
%        x(t,j) = sum(l=1:p) A(l,j) x(t-l,j) + v(t,j), j=1:M (dimension rxrxpxM)  
%       'C': Initial estimates of observation matrices C(j) in equation 
%        y(t) = C(j) x(t,j) + w(t), j=1:M (dimension NxrxM)
%       'Q': Initial estimate of state noise covariance Cov(v(t,j)) (dimension rxrxM)
%       'R': Initial estimate of observation noise covariance Cov(w(t)) (dimension NxN)                  
%       'mu': Initial estimate of mean state mu(j)=E(x(t,j)) for t=1:p (dimension rxM) 
%       'Sigma': Initial estimate of covariance Sigma(j)=Cov(x(t,j)) for t=1:p (dimension rxrxM)           
% control:  optional structure with fields: 
%       'eps': tolerance for EM termination; defaults to 1e-8
%       'ItrNo': number of EM iterations; defaults to 100 
%       'beta0': initial inverse temperature parameter for deterministic annealing; default 1 
%       'betarate': decay rate for temperature; default 1 
%       'safe': if true, regularizes variance matrices to be well-conditioned 
%           before taking inverse. If false, no regularization (faster but less safe)
%       'abstol': absolute tolerance for eigenvalues in matrix inversion (only effective if safe = true)
%       'reltol': relative tolerance for eigenvalues in matrix inversion
%           = inverse condition number 
% equal:  optional structure with fields:
%       'A': if true, VAR transition matrices A(l,j) are equal across regimes j=1,...,M
%       'Q': if true, VAR innovation matrices Q(j) are equal across regimes
%       'mu': if true, initial mean state vectors mu(j) are equal across regimes
%       'Sigma': if true, initial variance matrices Sigma(j) are equal across regimes
% fixed:  optional struct variable with fields 'A','C','Q','R','mu','Sigma'.
%       Each specified field must contain an array of the same dimensions 
%       as the input it represents, e.g. fixed.A must have the same dimensions
%       as A. Fixed coefficients are given by numerical values i the array 
%       and free coefficients are indicated by NaN's.
% scale:  optional struct variable with fields:
%       'A': upper bound for modulus of eigenvalues of A matrices. Must be positive.
%       'C': value of the (euclidean) column norms of the matrices C(j). Must be positive.
%
% OUTPUTS
% xf:  Filtered state vector
% xs:  Smoothed state vector
% outpars: struct variable with fields
%       A:  Estimated system (VAR) matrix
%       C:  Estimated observation matrix
%       Q:  Estimated state noise cov
%       R:  Estimated observation noise cov
%       mu:  Estimated initial mean of state vector
%       Sigma:  Estimated initial variance of state vector 
% LL :  Log-likelihood
%                    
% AUTHOR       
% David Degras, david.degras@umb.edu
% University of Massachusetts Boston
%
% CONTRIBUTORS
% Ting Chee Ming, cmting@utm.my
% Siti Balqis Samdin
% Center for Biomedical Engineering, Universiti Teknologi Malaysia.
%              
%--------------------------------------------------------------------------





%-------------------------------------------------------------------------%
%                           Initialization                                %
%-------------------------------------------------------------------------%


narginchk(5,10)

% Data dimensions
[N,T] = size(y);

% Data centering 
y = y - mean(y,2);

% 'small' state vector: x(t), size r
% 'big' state vector: X(t)=(x(t),...,x(t-p+1)), size p*r
% We assume t the initial vectors x(1),...,x(p) are iid ~ N(mu,Sigma)
% and  mutually independent with S(1),...,S(p).

% Check that time series has same length as regime sequence
assert(size(y,2) == numel(S));

%@@@@@ Initialize optional arguments if not specified @@@@@%
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



%@@@@@ Initialize estimators by OLS if not specified @@@@@%
pars0 = struct('A',[], 'C',[], 'Q',[], 'R',[], 'mu',[], 'Sigma',[]);
if exist('pars','var') && isstruct(pars)
    fname = fieldnames(pars0);
    for i = 1:6 
        name = fname{i};
        if isfield(pars,name)
            pars0.(name) = pars.(name);
        end
    end
end
pars = pars0;

Pi = zeros(M,1);
Pi(S(1)) = 1;
pars.Pi = Pi;
pars.Z = eye(M); 

if any(structfun(@isempty,pars)) 
    pars = reestimate_dyn(y,M,p,r,S,control,equal,fixed,scale);
end

[pars,control,equal,fixed,scale,skip] = ...
    preproc_dyn(M,N,p,r,pars,control,equal,fixed,scale);

abstol = control.abstol;
reltol = control.reltol;
eps = control.eps;
ItrNo = control.ItrNo;
verbose = control.verbose;
safe = control.safe;




%@@@@ Initialize other quantities @@@@@%

LL = zeros(1,ItrNo); % Log-likelihood
LLflag = 0; % counter for convergence of of log-likelihood 
sum_yy = y * y.'; % sum(t=1:T) y(t)*y(t)'
Ms = zeros(M,T);
for j = 1:M
    Ms(j,S == j) = 1;
end
sum_Ms2 = zeros(M);
for i = 1:M
    for j = 1:M
        sum_Ms2(i,j) = sum(S(1:T-1) == i & S(2:T) == j);
    end
end

for i = 1:ItrNo
    
   
    
%-------------------------------------------------------------------------%
%                               E-step                                    %
%-------------------------------------------------------------------------%




    % Kalman filtering and smoothing
    [xf,xs,L,MP0,Mx0,sum_MCP,sum_MP,sum_MPb,sum_P] = ... 
        kfs_dyn(y,M,p,r,pars,S,safe,abstol,reltol);
 
    sum_xy = xs * y.'; % sum(t=1:T) E(x(t)|y(1:T)) y(t)'

    % Log-likelihood
    LL(i) = L; 
    if verbose
        fprintf('Iteration-%d   Log-likelihood = %g\n',i,LL(i));
    end
    
    % Check if current solution is best to date
    if (i == 1 || LL(i) > LLbest)
        LLbest = LL(i);
        outpars = pars;        
        xfbest = xf; 
        xsbest = xs;
    end
     
    % Monitor convergence of log-likelihood
    if (i>1 &&  LL(i)-LL(i-1) < eps * abs(LL(i-1)))
        LLflag = LLflag + 1;
    else
        LLflag = 0;
    end  
    % Terminate EM algorithm if no sufficient reduction in log-likelihood
    % for 5 iterations in a row
    if LLflag == 10
        break;
    end
            

       
%-------------------------------------------------------------------------%
%                               M-step                                    %
%-------------------------------------------------------------------------%




    pars = M_dyn(pars,MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,...
        sum_P,sum_xy,sum_yy,control,equal,fixed,scale,skip);
    
    % Display Q-function if required 
    if verbose
        Qval = Q_dyn(pars,MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,...
            sum_P,sum_xy,sum_yy);
        fprintf('Q-function after M-step = %g\n',Qval);
    end

    
end % END MAIN LOOP


% Wrap-up
outpars.A = reshape(outpars.A,[r,r,p,M]);
LL = LL(1:i);
xf = xfbest;
xs = xsbest;

end
