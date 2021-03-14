
function [pars,Shat] = init_var(y,M,p,opts,control,equal,fixed,scale)

%--------------------------------------------------------------------------
%
%                   INITIALIZATION OF EM ALGORITHM 
%               FOR SWITCHING VECTOR AUTOREGRESSIVE MODEL
%
% PURPOSE 
% This function calculates initial parameter estimates for the main
% EM fitting function 'switch_var'
%
% USAGE 
% [pars,Shat] = init_var(y,M,p,opts,control,equal,fixed,scale)
%
% INPUTS
% y:    data (time = cols, variables = rows)
% M:    number of regimes for Markov chain
% p:    order of VAR state process
% opts:  optional structure with fields:
%       'segmentation':  with possible values 'fixed' (fixed segments) and
%           'binary' (binary segmentation). Default = 'fixed'
%       'len':  segment length. Only for fixed segmentation. 
%       'delta':  minimal distance between two consecutive change points.
%           Only for binary segmentation.
%       'tol':  minimum relative decrease in loss function for a point to be
%           acceptable as change point. Only for binary segmentation. See
%           function find_single_cp for more details.
% control:  optional structure with fields
%       'abstol':  absolute tolerance for eigenvalues when regularizing 
%           estimates of covariance matrices Q, R, and Sigma. Eigenvalues 
%           less than the lower bound abstol are replaced by it
%       'reltol':  relative tolerance for eigenvalues when regularizing 
%           estimates of covariance matrices Q, R, and Sigma. Eigenvalues 
%           less than the lower bound (max eigenvalue * reltol) are replaced 
%           by it
% 
% OUTPUTS
% Ahat:     estimate of transition matrices (rxrxpxM)
% Qhat:     estimate of state noise covariance matrices (rxrxM)
% muhat:    estimate of initial mean of state vector (rx1)
% Sigmahat:  estimate of initial covariance of state vector (rxr)
% Pihat:    estimate of probabilities of initial state of Markov chain (Mx1)
% Zhat:     estimate of transition probabilities (MxM)
% Shat:     estimate of Markov chain states S(t) (Tx1)
%
%--------------------------------------------------------------------------





% Check number of inputs
narginchk(3,8);

% Data dimensions
N = size(y,1);

% Initialize optional arguments if needed
if ~exist('opts','var')
    opts = [];
end
if ~exist('control','var')
    control = [];
end
if ~exist('equal','var')
    equal = [];
end
if ~exist('scale','var')
    scale = [];
end
if ~exist('fixed','var')
    fixed = struct();
end

% Pass arguments to init_dyn
fixed.R = 1e-10 * eye(N);
fixed.C = eye(N); 
[pars,Shat] = init_dyn(y,M,p,N,opts,control,equal,fixed,scale);
pars = rmfield(pars,'C');
pars = rmfield(pars,'R');



