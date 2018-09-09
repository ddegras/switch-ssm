function [xf,xs,Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,LL] = ... 
    fast_dyn(y,M,p,r,A,C,Q,R,mu,Sigma,S,control,equal,fixed,scale)

%--------------------------------------------------------------------------
%
%       PARAMETER ESTIMATION AND INFERENCE IN STATE-SPACE MODEL 
%           WITH SWITCHING SYNAMICS ASSUMING REGIMES KNOWN 
%
% PURPOSE
% This function estimates model parameters and infers hidden state vectors 
% by the EM algorithm in state-space models with switching dynamics under
% the assumption that the regime (i.e. switching) variables are known.
% The function can be used to fit the model under a trajectory of regimes 
% that is relevant or highly likely. In the case of only one regime (no
% switching), it can also be used to efficiently fit the standard linear
% state-space model.
%
% USAGE
%   [Mf,Ms,Sf,Ss,xf,xs,Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,LL] = ... 
%       switch_dyn(y,M,p,r,A,C,Q,R,mu,Sigma,Pi,Z,control,equal,fixed,scale)
% 
% INPUTS  
% y:    time series data (dimension NxT)
% M:    number of regimes
% p:    order of VAR model for state vector x(t)
% r:    size of state vector x(t)
% A:    Initial estimate of VAR matrices A(l,j) in system equation 
%       x(t,j) = sum(l=1:p) A(l,j) x(t-l,j) + v(t,j), j=1:M (dimension rxrxpxM)  
% C:    Initial estimates of observation matrices C(j) in equation 
%       y(t) = C(j) x(t,j) + w(t), j=1:M (dimension NxrxM)
% Q:    Initial estimate of state noise covariance Cov(v(t,j)) (dimension rxrxM)
% R:    Initial estimate of observation noise covariance Cov(w(t)) (dimension NxN)                  
% mu:   Initial estimate of mean state mu(j)=E(x(t,j)) for t=1:p (dimension rxM) 
% Sigma:  Initial estimate of covariance Sigma(j)=Cov(x(t,j)) for t=1:p (dimension rxrxM)           
% Pi:   Pilot state probability (dimension Mx1)
% Z:    Pilot Markov transition probability matrix (dimension MxM) 
% S:    fixed sequence of regimes S(t), t=1:T
% control:  optional struct variable with fields: 
%       'eps': tolerance for EM termination; defaults to 1e-8
%       'ItrNo': number of EM iterations; dfaults to 100 
%       'beta0': initial inverse temperature parameter for deterministic annealing; default 1 
%       'betarate': decay rate for temperature; default 1 
%       'safe': if true, regularizes variance matrices to be well-conditioned 
%           before taking inverse. If false, no regularization (faster but less safe)
%       'abstol': absolute tolerance for eigenvalues in matrix inversion (only effective if safe = true)
%       'reltol': relative tolerance for eigenvalues in matrix inversion
%           = inverse condition number 
% equal:  optional struct variable with fields:
%       'A': if true, VAR transition matrices A(l,j) are equal across regimes j=1,...,M
%       'C': if true, observation matrices C(j) are equal across regimes
%       'Q': if true, VAR innovation matrices Q(j) are equal across regimes
%       'mu': if true, initial mean state vectors mu(j) are equal across regimes
%       'Sigma': if true, initial variance matrices Sigma(j) are equal across regimes
% fixed:  optional struct variable with fields 'A','C','Q','R','mu','Sigma'.
%   If not empty, each field must contain a matrix with 2 columns, the first for 
%   the location of fixed coefficients and the second for their values. 
% scale:  optional struct variable with fields:
%       'A': upper bound for norm of eigenvalues of A matrices. Must be in (0,1).
%       'C': value of the (euclidean) column norms of the matrices C(j). Must be positive.
%
% OUTPUTS
% Mf:  State probability estimated by switching Kalman Filter
% Ms:  State probability estimated by switching Kalman Smoother
% Sf:  Estimated states (Kalman Filter) 
% Ss:  Estimated states (Kalman Smoother) 
% xf:  Filtered state vector
% xs:  Smoothed state vector
% Ahat:  Estimated system matrix
% Chat:  Estimated observation matrix
% Qhat:  Estimated state noise cov
% Rhat:  Estimated observation noise cov
% muhat:  Estimated initial mean of state vector
% Sigmahat:  Estimated initial variance of state vector 
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
% Version xxx  : June 13 2017
% Reference: [1] S. B. Samdin, C. Ting, H. Ombao, and S. Salleh,
%              “A Unified Estimation Framework for State-Related Changes
%              in Effective Brain Connectivity,” IEEE Trans. Biomed. Eng., 
%              vol. -, no. -, pp. -, 2016.
%            [2] K. Murphy, "Switching Kalman Filters," 1998.
%--------------------------------------------------------------------------





%-------------------------------------------------------------------------%
%                           Initialization                                %
%-------------------------------------------------------------------------%


narginchk(4,15)

% Data dimensions
[N,T] = size(y);

% 'small' state vector: x(t), size r
% 'big' state vector: X(t)=(x(t),...,x(t-p+1)), size p*r
% We assume that the initial vectors x(1),...,x(p) are iid ~ N(mu,Sigma)
% and  mutually independent with S(1),...,S(p).



%@@@@@ Initialize optional arguments if not specified @@@@@%
if ~exist('fixed','var') || isempty(fixed)
    fixed = struct();
end
if ~exist('equal','var') || isempty(equal)
    equal = struct('mu',true,'Sigma',true);
end
if ~exist('control','var') || isempty(control)
    control = struct();
end
if ~exist('scale','var') || isempty(scale)
    scale = struct('A',.999,'C',[]);
else 
    if ~isfield(scale,'A')
        scale.A = .999;
    end
    if ~isfield(scale,'C')
        scale.C = [];
    end
end



%@@@@@ Initialize estimators @@@@@%

if nargin == 4
    [A,C,Q,R,mu,Sigma,Pi,Z,S] = init_dyn(y,M,p,r);
else
    % Set Pi and Z to arbitrary values. These will not be used in the function
    % but must be specified for the initialization function preproc_dyn
    Pi = zeros(M,1);
    Pi(1) = 1;
    Z = eye(M);
    if isfield(fixed,'Pi')
        fixed = rmfield(fixed,'Pi');
    end
    if isfield(fixed,'Z')
        fixed = rmfield(fixed,'Z');
    end
end
    
[Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,~,~,fixed,skip,equal,...
    eps,ItrNo,~,~,safe,abstol,reltol,verbose,scale] = ... 
    preproc_dyn(M,N,p,r,A,C,Q,R,mu,Sigma,Pi,Z,control,equal,fixed,scale);

% Set Markov chain probabilities to zero for Q-function calculation
% (Will not be used elsewhere)
Pihat = zeros(M,1); 
Zhat = zeros(M);

% Initial parameters 'A','C',... are expanded from r-space (x(t)) to
% r-space (X(t)). The new parameters 'Ahat','Chat',... have dimensions:
% Ahat: (p*r)x(p*r)xM,    Chat: NxrxM,    Qhat: rxrxM,  Rhat: NxN,  muhat:
% rxM, Sigmahat: rxrxM. These parameters respect the fixed coefficients and
% equality constraints (either default values or user-specified ones).

% The structure 'fixed' has fields 'fixed.A',... (one field per model
% parameter). Each field is a matrix with two columns containing the
% locations and values of fixed coefficients in the corresponding
% parameter.

% The structure 'skip' specified whether a parameter (A,C,Q...) is entirely
% fixed, in which case its update in the M-step can be skipped altogether

% The structure 'equal' has fields representing equality
% constraints on parameters across regimes j=1:M. By default, equality
% constraints are: Q: false, mu: true, Sigma: true. In other words, only
% the initial parameters mu and Sigma are assumed to be common across
% regimes. If provided, user values will override default values. Note:
% equality constraints are not applicable for C & R (only one parameter by
% definition) and not currently supported for A

% Various control parameters ('eps','ItrNo',...) are set either
% to their default values or to user-specified values through argument 'control'. 


%@@@@ Initialize other quantities @@@@@%

LL = zeros(1,ItrNo); % Log-likelihood
LLflag = 0; % counter for convergence of of log-likelihood 
sum_yy = y * y.'; % sum(t=1:T) y(t)*y(t)'
Shat = S;
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

% Indices and values of fixed coefficients in A(j) = (A(1,j),...,A(M,j)) in
% 2 column-matrix format (needed for projected gradient with A in M-step)
fixed_A = cell(M,1);
if ~isempty(fixed.A)
    Atmp = NaN(r,p*r,M);
    Atmp(fixed.A(:,1)) = fixed.A(:,2);  
    for j = 1:M
        idx = find(~isnan(Atmp(:,:,j)));
        fixed_A{j} = [idx,Atmp(idx)];
        if equal.A
            fixed_A = fixed_A(1);
            break
        end
    end  
end

% Mask of rxr diagonal blocks in a (p*r)x(p*r) matrix
% (used to update Sigma in M-step)
Sigmamask = reshape(find(kron(eye(p),ones(r))),r,r,p);

% Containers for A,C,Q  
Abig = repmat(diag(ones((p-1)*r,1),-r),[1,1,M]);
Cbig = zeros(N,p*r);
Qbig = zeros(p*r,p*r,M);
Sbig = zeros(p*r,p*r,M);


for i = 1:ItrNo
    
   
    
%-------------------------------------------------------------------------%
%                               E-step                                    %
%-------------------------------------------------------------------------%


    Abig(1:r,:,:) = Ahat;
    Cbig(:,1:r) = Chat;
    Qbig(1:r,1:r,:) = Qhat;
    mubig = repmat(muhat,p,1);
    for j = 1:M
        Sbig(:,:,j) = kron(eye(p),Sigmahat(:,:,j));
    end
 
    [xf,xs,L,MP0,Mx0,sum_MCP,sum_MP,sum_MPb,sum_P] = ... 
        kfs_dyn(y,M,p,r,Abig,Cbig,Qbig,Rhat,mubig,Sbig,Shat,...
            safe,abstol,reltol);
 
    sum_xy = xs * y.'; % sum(t=1:T) E(x(t)|y(1:T)) y(t)'

    % Log-likelihood
    LL(i)=L; 
    if verbose
        fprintf('Iteration-%d   Log-likelihood = %g\n',i,LL(i));
    end
    
    % Check if current solution is best to date
    if (i==1 || LL(i) > LLbest)
        LLbest = LL(i);
        Abest = Ahat;
        Cbest = Chat;
        Qbest = Qhat;
        Rbest = Rhat;
        mubest = muhat;
        Sigmabest = Sigmahat;
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



% Unconstrained parameter estimates 
% A = ( sum(t=p+1:T) P(t,t-1|T) ) * ( sum(t=p+1:T) P~(t-1|T) )^{-1}
% Cj = (sum(t=1:T) Wj(t) y(t) xj(t)') * (sum(t=1:T) Wj(t) Pj(t|T))^{-1}
% Qj = (sum(t=2:T) Wj(t) Pj(t) - Aj * sum(t=2:T) Wj(t) Pj(t-1,t)') / sum(t=2:T) Wj(t)
% R = sum(t=1:T) y(t) y(t)' / T - sum(t=1:T) x(t|T) y(t)'
%
% where Wj(t) = P(S(t)=j|y(1:T)), 
% xj(t|T) = E(x(t)|S(t)=j,y(1:T)), 
% x(t|T) = E(x(t)|y(1:T)), 
% Pj(t|T) = E(x(t)x(t)'|S(t)=j,y(1:T)), 
% P~(t-1|T) = E(x(t-1)x(t-1)'|S(t)=j,y(1:T))
% and P(t,t-1|T) = E(x(t)x(t-1)'|y(1:T))

% Stability issues for A and scale constraints for C are handled with a
% projected gradient technique (maximize Q-function under constraint)

    %@@@@@ Update A @@@@@%
    
    % Case: no fixed coefficient constraints
    Ahatold = Ahat;
    if ~skip.A && isempty(fixed.A)
        if equal.A && equal.Q
            sum_Pb = sum(sum_MPb,3);
            sum_CP = sum(sum_MCP,3);
            Ahat = sum_CP / sum_Pb;
            if any(isnan(Ahat(:))|isinf(Ahat(:)))
                Ahat = sum_CP * pinv(sum_Pb);
            end
            Ahat = repmat(Ahat,[1,1,M]);
        elseif equal.A
            % If the A's are all equal but the Q's are not, there is no closed
            % form expression for the A and Q's that maximize the Q function.
            % In this case, fix the Q's and find the best associated A (ECM) 
            lhs = zeros(p*r,p*r);
            rhs = zeros(r,p*r);
            for j=1:M
                Qinv_j = myinv(Qhat(:,:,j));         
                lhs = lhs + kron(sum_MPb(:,:,j),Qinv_j);
                rhs = rhs + Qinv_j * sum_MCP(:,:,j);
            end
            rhs = rhs(:);
            Ahat = reshape(lhs\rhs,r,p*r);
            if any(isnan(Ahat(:))|isinf(Ahat(:)))
                Ahat = reshape(pinv(lhs)*rhs,r,p*r);
            end
            Ahat = repmat(Ahat,[1,1,M]);
        else
            for j=1:M
                A_j = sum_MCP(:,:,j) / sum_MPb(:,:,j);
                if any(isnan(A_j(:)) | isinf(A_j(:)))
                     A_j = sum_MCP(:,:,j) * pinv(sum_MPb(:,:,j));
                end
                Ahat(:,:,j) = A_j;
            end
        end   
    end
    
    % Case: fixed coefficient constraints on A --> Vectorize matrices and
    % solve associated problem after discarding rows associated with fixed
    % coefficients. Recall: there cannot be both fixed coefficient
    % constraints *and* equality constraints on A
    if ~skip.A && ~isempty(fixed.A)
        for j = 1:M
            % Linear indices of free coefficients in A(j)
            free = setdiff(1:p*r^2,fixed_A{j}(:,1));
            free = reshape(free,[],1);
            Qinv_j = myinv(Qhat(:,:,j));
            % Matrix problem min(X) trace(W(-2*B1*X' + X*B2*X')) 
            % (under fixed coefficient constraints) becomes vector problem 
            % min(x) x' kron(B2,W) x - 2 x' vec(W*B1)
            % with X = A(j), x = vec(A(j)), W = Q(j)^(-1), B1 = sum_MCP(j),
            % and B2 = sum_MPb(j) (remove fixed entries in x)
            lhs = kron(sum_MPb(:,:,j),Qinv_j);
            lhs = lhs(free,free);
            rhs = Qinv_j * sum_MCP(:,:,j);            
            rhs = rhs(free);
            A_j = zeros(p*r^2,1);
            A_j(fixed_A{j}(:,1)) = fixed_A{j}(:,2);
            A_j(free) = reshape(lhs\rhs,r,p*r);
            if any(isnan(A_j)|isinf(A_j))
                A_j = pinv(lhs)*rhs;
            end
            Ahat(:,:,j) = reshape(A_j,r,p*r);
        end
    end
    
    % Check eigenvalues of estimate and regularize if less than 'scale.A'.
    % Regularization: algebraic method if no fixed coefficients or all
    % fixed coefficients are zero, projected gradient otherwise        
    if ~skip.A
        for j = 1:M
            % Check eigenvalues
            Abig(1:r,:,j) = Ahat(:,:,j);
            eigval = eig(Abig(:,:,j));
            if any(abs(eigval) > scale.A)
                if verbose
                    warning(['Eigenvalues of A%d greater than %f.',...
                        ' Regularizing.'],j,scale.A)
                end
                % Case: regularize with no fixed coefficients or all fixed
                % coefficients equal to zero
                if isempty(fixed_A{j}) || all(fixed_A{j}(:,2) == 0)
                    c = .999 * scale.A / max(abs(eigval));
                    A_j = reshape(Ahat(:,:,j),[r,r,p]);
                    for l = 1:p
                        A_j(:,:,l) = c^l * A_j(:,:,l);
                    end 
                    Ahat(:,:,j) = reshape(A_j,[r,p*r]);
                else
                % Case: regularize with fixed coefficients constraints  
                    A_j = Ahatold(:,:,j);
                    Q_j = Qhat(:,:,j);
                    sum_MPbj = sum_MPb(:,:,j);
                    sum_MCPj = sum_MCP(:,:,j);
                    % Projected gradient on Q-function holding matrix Q fixed
                    [A_j,~] = PG_A(A_j,sum_MCPj,sum_MPbj,Q_j,scale.A,fixed_A{j});
                    Ahat(:,:,j) = A_j;
                end
            end
            if equal.A
                Ahat = repmat(Ahat(:,:,1),[1,1,M]);
                break
            end 
        end               
            

        % Check that parameter update actually increases Q-function 
        % If not, keep previous parameter estimate 
        Qvalold = Q_dyn(Ahatold,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,p,T,...
            MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,sum_P,sum_xy,sum_yy);
        Qval = Q_dyn(Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,p,T,...
            MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,sum_P,sum_xy,sum_yy);
        if Qval < Qvalold
            Ahat = Ahatold;
        end
    end
 
    
    %@@@@@ Update C @@@@@%
    
    if ~skip.C 
        Chatold = Chat;
        % Case: no fixed coefficient and/or scale constraints on C
        % Calculate estimate in closed form
        if isempty(fixed.C) && isempty(scale.C)
            Chat = sum_xy.' / sum_P;
            if any(isnan(Chat(:)) | isinf(Chat(:)))
                Chat = sum_xy.' * pinv(sum_P);
            end
        else
        % Otherwise: perform constrained estimation by projected gradient
            [Chat,~] = PG_C(Chatold(:,1:r),sum_xy,sum_P,Rhat,scale.C,fixed.C);
        end
    
        % Check that parameter update actually increases Q-function. If
        % not, keep previous parameter estimate. (This is a redundancy
        % check: by design, the above constrained and unconstrained
        % estimates cannot decrease the Q-function)
        Qvalold = Q_dyn(Ahat,Chatold,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,p,T,...
            MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,sum_P,sum_xy,sum_yy);
        Qval = Q_dyn(Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,p,T,...
            MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,sum_P,sum_xy,sum_yy);
        if Qval < Qvalold
            Chat = Chatold;
        end
    end
    
    %@@@@@ Update Q @@@@@%
    
    % Unconstrained solution
    Qhatold = Qhat; 
    if ~skip.Q        
        if equal.Q
            Qtmp = zeros(r,r,M);
            for j=1:M
                A_j = Ahat(1:r,:,j);
                sum_MPj = sum_MP(:,:,j);
                sum_MCPj = sum_MCP(:,:,j);
                sum_MPbj = sum_MPb(:,:,j);                
                Qtmp(:,:,j) = sum_MPj - (sum_MCPj * A_j.') - ...
                    (A_j * sum_MCPj.') + A_j * sum_MPbj * A_j.';
            end
            Qtmp = sum(Qtmp,3) / (T-1);
            Qtmp = 0.5 * (Qtmp + Qtmp.');
            Qhat = repmat(Qtmp,1,1,M);
        else
            for j=1:M
                sum_Mj = sum(Ms(j,2:T));
                sum_MPj = sum_MP(:,:,j);
                sum_MCPj = sum_MCP(:,:,j);
                sum_MPbj = sum_MPb(:,:,j);
                A_j = Ahat(1:r,:,j);
                if sum_Mj > 0
                    Q_j = (sum_MPj - (A_j * sum_MCPj') - (sum_MCPj * A_j') ...
                        + (A_j * sum_MPbj * A_j')) / sum_Mj;
                else
                    Q_j = eye(r);
                end
                Q_j = 0.5 * (Q_j + Q_j');
                Qhat(:,:,j) = Q_j;
            end
        end
    
        % Enforce fixed coefficient constraints
        if ~isempty(fixed.Q)
            Qhat(fixed.Q(:,1)) = fixed.Q(:,2);
        end
        
        % Regularize estimate if needed
        for j = 1:M
            eigval = eig(Qhat(:,:,j));
            if min(eigval) < max(abstol,max(eigval)*reltol)
                if verbose 
                    warning(['Q%d ill-conditioned and/or nearly singular.', ... 
                        ' Regularizing.'],j);
                end
                Qhat(:,:,j) = regfun(Qhat(:,:,j),abstol,reltol);
            end
            if equal.Q
                Qhat = repmat(Qhat(:,:,1),[1,1,M]);
                break
            end
        end
        
        % Apply fixed coefficient constraints
        if ~isempty(fixed.Q)
            Qhat(fixed.Q(:,1)) = fixed.Q(:,2);
        end

        % Check that estimate Ahat increases Q-function. If not, keep
        % estimate from previous iteration
        Qvalold = Q_dyn(Ahat,Chat,Qhatold,Rhat,muhat,Sigmahat,Pihat,Zhat,p,T,...
            MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,sum_P,sum_xy,sum_yy);
        Qval = Q_dyn(Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,p,T,...
            MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,sum_P,sum_xy,sum_yy);
        if Qval < Qvalold
            Qhat = Qhatold;
        end        
    end

    %@@@@@ Update R
    Rhatold = Rhat;
    if ~skip.R
        % Unconstrained solution
        Rhat = (sum_yy - Chat * sum_xy - (Chat * sum_xy)' + ...
            Chat * sum_P * Chat') / T;
        Rhat = 0.5 * (Rhat + Rhat');
        % Apply fixed coefficient constraints
        if ~isempty(fixed.R)
            Rhat(fixed.R(:,1)) = fixed.R(:,2);
        end
        % Check positive definiteness and conditioning of Rhat. Regularize
        % if needed
        eigval = eig(Rhat);
        if min(eigval) < max(abstol,max(eigval)*reltol)
            if verbose
                warning('R ill-conditioned and/or nearly singular. Regularizing.');
            end
            Rhat = regfun(Rhat,abstol,reltol); 
            if ~isempty(fixed.R)
                Rhat(fixed.R(:,1)) = fixed.R(:,2);
            end
        end 
        
        % Check that Rhat increases Q-function. If not, keep estimate from
        % previous iteration
        Qvalold = Q_dyn(Ahat,Chat,Qhat,Rhatold,muhat,Sigmahat,Pihat,Zhat,p,T,...
            MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,sum_P,sum_xy,sum_yy);
        Qval = Q_dyn(Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,p,T,...
            MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,sum_P,sum_xy,sum_yy);
        if Qval < Qvalold
            Rhat = Rhatold;
        end
    end
    
    %@@@@@ Update mu 
    muhatold = muhat;
    if ~skip.mu
        sum_Mx0 = squeeze(sum(reshape(Mx0,[r,p,M]),2)); 
        if equal.mu && equal.Sigma
            muhat = sum(sum_Mx0,2)/p;
            muhat = repmat(muhat,1,M);
        elseif equal.mu
            lhs = zeros(r,r);
            rhs = zeros(r,1);
            for j=1:M
                Sinv_j = myinv(Sigmahat(:,:,j));         
                lhs = lhs + (p * Ms(j,1)) * Sinv_j;
                rhs = rhs + Sinv_j * sum_Mx0(:,j);
            end
            muhat = lhs\rhs;
            if any(isnan(muhat) | isinf(muhat))
                muhat = myinv(lhs)*rhs;
            end
            muhat = repmat(muhat,1,M);        
        else
            muhat = zeros(r,M);
            for j = 1:M
                if Ms(j,1) > 0
                    muhat(:,j) = sum_Mx0(:,j) / Ms(j,1);
                end
            end            
        end
        
        % Apply fixed coefficient constraints
        if ~isempty(fixed.mu)
            muhat(fixed.mu(:,1)) = fixed.mu(:,2);
        end

        % Check that muhat increases Q-function. If not, keep estimate from
        % previous iteration
        Qvalold = Q_dyn(Ahat,Chat,Qhat,Rhat,muhatold,Sigmahat,Pihat,Zhat,p,T,...
            MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,sum_P,sum_xy,sum_yy);
        Qval = Q_dyn(Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,p,T,...
            MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,sum_P,sum_xy,sum_yy);
        if Qval < Qvalold
            muhat = muhatold;
        end
    end
    
    %@@@@@ Update Sigma 
    Sigmahatold = Sigmahat;
    mubig = repmat(muhat,[p,1]);
    if ~skip.Sigma
        % Unconstrained solution
        if equal.Sigma
            Stmp = sum(MP0,3) - (mubig * Mx0') - (Mx0 * mubig') + ...
                (mubig * diag(Ms(:,1)) * mubig'); % dimension (p*r)x(p*r)
            Sigmahat = mean(Stmp(Sigmamask),3); % dimension rxr 
            Sigmahat = 0.5 * (Sigmahat + Sigmahat'); % symmetrize
            Sigmahat = repmat(Sigmahat(:,:,1),[1,1,M]); % replicate
        else
            for j = 1:M
                if Ms(j,1) > 0
                    S_j = MP0(:,:,j) - (mubig(:,j) * Mx0(:,j)') - ...
                        (Mx0(:,j) * mubig(:,j)') + Ms(j,1) * (mubig(:,j) * mubig(:,j)');             
                S_j = mean(S_j(Sigmamask),3) / Ms(j,1); 
                S_j = 0.5 * (S_j + S_j'); 
                else
                    S_j = eye(r);
                end
                Sigmahat(:,:,j) = S_j; 
            end
        end
        
        % Enforce any fixed coefficient constraints
        if ~isempty(fixed.Sigma)
            Sigmahat(fixed.Sigma(:,1)) = fixed.Sigma(:,2);
        end
        
        % Regularize estimate if needed
        for j = 1:M
            eigval = eig(Sigmahat(:,:,j));
            if min(eigval) < max(abstol,max(eigval)*reltol)
                if verbose
                    warning(['Sigma%d ill-conditioned and/or nearly singular.', ...
                        ' Regularizing.'],j);
                end
                Sigmahat(:,:,j) = regfun(Sigmahat(:,:,j),abstol,reltol); 
            end 
            if equal.Sigma
                Sigmahat = repmat(Sigmahat(:,:,1),[1,1,M]);
                break
            end
        end
        % Enforce fixed coefficient constraints 
        if ~isempty(fixed.Sigma)
            Sigmahat(fixed.Sigma(:,1)) = fixed.Sigma(:,2);
        end

        % Check that Sigmahat increases Q-function. If not, keep 
        % parameter estimate from previous iteration
        Qvalold = Q_dyn(Ahat,Chat,Qhat,Rhat,muhat,Sigmahatold,Pihat,Zhat,p,T,...
            MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,sum_P,sum_xy,sum_yy);
        Qval = Q_dyn(Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,p,T,...
            MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,sum_P,sum_xy,sum_yy);
        if Qval < Qvalold
            Sigmahat = Sigmahatold;
        end
    end
    
    %@@@@@ Update Pi
    if ~skip.Pi
        Pihat = Ms(:,1);
    end
    
    %@@@@@ Update Z
    if ~skip.Z
        Zhat = sum_Ms2 ./ repmat(sum(sum_Ms2,2),1,M);
        if ~isempty(fixed.Z)
            Zhat(fixed.Z(:,1)) = fixed.Z(:,2);
        end
    end
   
    
end % END MAIN LOOP




%-------------------------------------------------------------------------%
%                               Output                                    %
%-------------------------------------------------------------------------%

Ahat = reshape(Abest(1:r,:,:),r,r,p,M);
Chat = Cbest(:,1:r);
Qhat = Qbest(1:r,1:r,:);
Rhat = Rbest;
muhat = mubest(1:r,:);
Sigmahat = Sigmabest(1:r,1:r,:);
LL = LL(1:i);

end

