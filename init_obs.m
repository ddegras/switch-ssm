function [Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,Shat] ... 
    = init_obs(y,M,p,r,opts,control,equal,fixed,scale)

%--------------------------------------------------------------------------
%
%       INITIALIZATION OF EM ALGORITHM FOR STATE-SPACE MODEL 
%                 WITH MARKOV-SWITCHING OBSERVATIONS
%
% PURPOSE  
% This function calculates initial parameter estimates for the main
% EM fitting function 'switch_obs'. The underlying model is 
%           y(t) = C(S(t)) * x(t,S(t)) + w(t), t=1:T                      
%           x(t,j) = sum(l=1:p) A(l,j) * x(t-l,j) + v(t,j), j=1:M         
%           S(t) = switching variable (Markov chain) taking values in {1:M} 
% 
% USAGE    
% [Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,Shat] ... 
%               = init_obs(y,M,p,r,opts,control,equal,fixed,scale)
%
% INPUTS:
% y -   data (size NxT with time in cols, variables in rows) 
% M -   number of regimes for Markov chain S(t), t=1:T
% p -   order of vector autoregressive (VAR) model for state vectors x(t,j)
% r -   dimension of state vectors x(t,j), j=1:M
% opts - optional structure with fields:
%       'segmentation': possible values 'fixed' for fixed segments and
%           'binary' for binary segmentation 
%       'reestimation': if 'true', parameters A & Q are re-estimated 
%           after clustering; if 'false', A & Q are estimated by the
%           cluster centroids.
%       'delta': minimal distance between two consecutive change points
%           in binary segmentation
%       'tol': minimum relative decrease in loss function for a point to be
%           a valid change point (for binary segmentation). See function 
%           find_single_cp for more details.
%
% OUTPUTS:
% Ahat -    estimate of transition matrices A(l,j) (size rxrxpxM)
% Chat -    estimate of observation matrices C(j), j=1:M (size NxrxM) 
% Qhat -    estimate of state noise variance matrices Q(j) = V(v(t,j)) 
%           (size rxrxM)
% Rhat -    estimate of observation noise variance matrix R = V(w(t)) 
%           (size NxN)
% muhat -   estimate of initial mean of state vector mu(j) = E(x(1,j))
%           (size rxM)
% Sigmahat - estimate of initial variance of state vector Sigma(j) = V(x(1,j)) 
%           (size rxrxM)
% Pihat -   estimate of initial probabilities Pi(j) = P(S(1)=j) (size Mx1)
% Zhat -    estimate of transition probabilities between regimes Z(i,j) = 
%           P(S(t)=j|S(t-1)=i) (size MxM)
% Shat -    estimate of regimes S(t) (size Tx1)
%
% Author:   David Degras, david.degras@umb.edu
%           University of Massachusetts Boston
%
% Date:     August 28, 2018
%
% Reference: "Exploring dynamic functional connectivity of the brain with
%           switching state-space models". D. Degras, C.-M. Ting, and 
%           H. Ombao (2018)
%--------------------------------------------------------------------------




%-------------------------------------------------------------------------%
%                             Preprocessing                               % 
%-------------------------------------------------------------------------%

% Check number of arguments
narginchk(4,9)

% Data dimensions
[N,~] = size(y);

% Initialize optional arguments if needed
if ~exist('opts','var')
    opts = [];
end
if ~exist('control','var')
    control = [];
end
if ~exist('fixed','var')
    fixed = [];
end
if ~exist('equal','var')
    equal = [];
end
if ~exist('scale','var')
    scale = [];
end



%-------------------------------------------------------------------------%
%     Initial parameter estimation for model with switching dynamics      % 
%-------------------------------------------------------------------------%

%  
% This step is used to segment the time series. It also checks the optional
% arguments 'control', 'fixed', 'equal', and 'scale'

fixed_tmp = fixed;
test1 = isstruct(fixed) && isfield(fixed,'C');
test2 = isstruct(equal) && isfield(equal,'C') && equal.C;
if test1
    if test2
        fixed_tmp.C = fixed.C(:,:,1);
    else
        fixed_tmp = rmfield(fixed,'C');
    end
end

[Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,Shat] = ...
    init_dyn(y,M,p,r,opts,control,equal,fixed_tmp,scale); 

% Trivial case: M=1 (no switching). In this case the switching dynamics and
% switching observations models are identical (a standard linear state-space
% model)

if M == 1
    return
end



%-------------------------------------------------------------------------%
%                             Re-estimation                               % 
%-------------------------------------------------------------------------%


[Ahat,Chat,Qhat,Rhat,~,~,~,~] = ...
    reestimate_obs(y,M,p,r,Shat,control,equal,fixed,scale);

% Test compatibility between initial estimates and specified constraints
test = preproc_obs(M,N,p,r,Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,...
    fixed,equal,control,scale); %#ok<NASGU>





        