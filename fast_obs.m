function [xf,xs,Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,LL] = ... 
    fast_obs(y,M,p,A,C,Q,R,mu,Sigma,S,control,equal,fixed,scale)
%--------------------------------------------------------------------------
% Title:    Parameter estimation and inference in state-space models with 
%           regime switching (switching observations) assuming regimes known
%
% Function: Infer hidden state vectors and regimes by switching Kalman 
%           filtering/smoothing (aka Hamilton filtering or Kim filtering) 
%           and estimate model parameters by maximum likelihood (EM algorithm).
%
% Usage:    [Mf,Ms,Sf,Ss,xf,xs,Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,LL] = ... 
%               fast_obs(y,M,p,A,C,Q,R,mu,Sigma,S,control,equal,fixed,scale)
% 
% Inputs:  
%       y - Time series (dimension NxT)
%       M - number of regimes
%       p - order of VAR model for state vector 
%       A - Initial estimates of VAR matrices A(l,j) in system equation 
%           x(t,j) = sum(l=1:p) A(l,j) x(t-l,j) + v(t,j), j=1:M (dimension rxrxpxM)  
%       C - Initial estimates of observation matrices C(j) in equation 
%           y(t) = C(j) x(t,j) + w(t), j=1:M (dimension NxrxM)
%       Q - Initial estimates of state noise covariance Cov(v(t,j)) (dimension rxrxM)
%       R - Pilot estimate of observation noise covariance Cov(w(t)) (dimension NxN)                  
%       mu - Pilot estimate of mean state mu(j)=E(x(t,j)) for t=1:p (dimension rxM) 
%       Sigma - Pilot estimate of covariance Sigma(j)=Cov(x(t,j)) for t=1:p (dimension rxrxM)           
%       S - regime sequence (length T)
%       control - optional struct variable with fields: 
%           'eps': tolerance for EM termination; defaults to 1e-8
%           'ItrNo': number of EM iterations; dfaults to 100 
%           'beta0': initial inverse temperature parameter for deterministic annealing; default 1 
%           'betarate': decay rate for temperature; default 1 
%           'safe': if true, regularizes variance matrices to be well-conditioned 
%            before taking inverse. If false, no regularization (faster but less safe)
%           'abstol': absolute tolerance for eigenvalues in matrix inversion (only effective if safe = true)
%           'reltol': relative tolerance for eigenvalues in matrix inversion
%            = inverse condition number (only effective if safe = true)
%       equal - optional struct variable with fields:
%           'A': if true, VAR transition matrices A(l,j) are equal across regimes j=1,...,M
%           'C': if true, observation matrices C(j) are equal across regimes
%           'Q': if true, VAR innovation matrices Q(j) are equal across regimes
%           'mu': if true, initial mean state vectors mu(j) are equal across regimes
%           'Sigma': if true, initial variance matrices Sigma(j) are equal across regimes
%       fixed - optional struct variable with fields 'A','C','Q','R','mu','Sigma'.
%           If not empty, each field must contain a matrix with 2 columns, the first for 
%           the location of fixed coefficients and the second for their values. 
%       scale - optional struct variable with fields:
%           'A': upper bound for norm of eigenvalues of A matrices. Must be in (0,1).
%           'C': value of the (euclidean) column norms of the matrices C(j). Must be positive.
%
% Outputs: 
%       Mf - State probability estimated by switching Kalman Filter
%       Ms - State probability estimated by switching Kalman Smoother
%       Sf - Estimated states (Kalman Filter) 
%       Ss - Estimated states (Kalman Smoother) 
%       xf - Filtered state vector
%       xs - Smoothed state vector
%       Ahat - Estimated system matrix
%       Chat - Estimated observation matrix
%       Qhat - Estimated state noise cov
%       Rhat - Estimated observation noise cov
%       muhat - Estimated initial mean of state vector
%       Sigmahat - Estimated initial variance of state vector 
%       LL  - Log-likelihood

% Variables:
%       T = length of signal
%       N = dimension of observation vector
%       r = dimension of state vector
%       M = number of regimes/states
%
% Author:       David Degras
%               University of Massachusetts Boston
%
% Contributors: Ting Chee Ming, cmting@utm.my
%               Siti Balqis Samdin
%               Centre for Biomedical Engineering, Universiti Teknologi Malaysia.
%               
% Version date: November 3, 2018
%--------------------------------------------------------------------------





%-------------------------------------------------------------------------%
%                           Initialization                                %
%-------------------------------------------------------------------------%


% Data dimensions
[N,T] = size(y);
r = size(A,1);

% x(t,j): state vector for j-th process at time t (size r0)
% x(t) = x(t,1),...,x(t,M): state vector for all processes at time t (size M*r0)
% X(t,j) = x(t,j),...,x(t-p+1,j)): state vector for j-th process at times t,...,t-p+1 (size r=p*r0)
% X(t) = x(t,1),...,x(t,M): state vector for all processes at times t,...,t-p+1 (size M*p*r0)


% We assume that the initial vectors x(1),...,x(1-p+1) are iid ~ N(mu,Sigma)


%@@@@ Initialize optional arguments if not specified

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



%@@@@ Initialize estimators @@@@%
% Set Pi and Z to arbitrary values. These will not be used in the function
% but must be specified for the initialization function preproc_dyn
Pi = zeros(M,1);
Pi(S(1)) = 1;
Z = eye(M);
if isstruct(fixed) && isfield(fixed,'Pi')
    fixed = rmfield(fixed,'Pi');
end
if isstruct(fixed) && isfield(fixed,'Z')
    fixed = rmfield(fixed,'Z');
end

[Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,fixed,equal,eps,...
    ItrNo,~,~,safe,abstol,reltol,verbose,scale] = ... 
    preproc_obs(M,N,p,A,C,Q,R,mu,Sigma,Pi,Z,fixed,equal,control,scale);





%@@@@ Initialize other quantities

LL = zeros(1,ItrNo); % Log-likelihood
LLbest = -Inf;
LLflag = 0; % counter for convergence of of log-likelihood 
sum_yy = y * y.'; % sum(t=1:T) y(t)*y(t)'
Ms = zeros(M,T); % P(S(t)=j), for use in Q-function
% Ms(sub2ind([M,T],S,T)) = 1;
sum_Ms2 = zeros(M);
for j = 1:M
    Ms(j,S == j) = 1;
    for k = 1:M
        sum_Ms2(j,k) = sum(S(1:end-1) == j & S(2:end) == k);
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

% Expanded parameters
Abig = repmat(diag(ones((p-1)*r,1),-r),[1,1,M]);

% Mask of rxr diagonal blocks in a (p*r)x(p*r) matrix
% (used to update Sigma in M-step)
Sigmamask = reshape(find(kron(eye(p),ones(r))),r,r,p);


for i = 1:ItrNo
    
   

%-------------------------------------------------------------------------%
%                     Filtering and smoothing + E-step                    %
%-------------------------------------------------------------------------%

    
[xf,xs,x0,P0,Loglik,sum_CP,sum_MP,sum_Mxy,sum_P,sum_Pb] = ...
    kfs_obs(y,M,p,Ahat,Chat,Qhat,Rhat,S,muhat,Sigmahat,safe,abstol,reltol);    
        
    % Log-likelihood
    LL(i) = Loglik; 
    if verbose
        fprintf('Iteration-%d   Log-likelihood = %g\n',i,LL(i));
    end
        
    % Check if current solution is best to date
    if LL(i) > LLbest
        LLbest = LL(i);
        Abest = Ahat;
        Cbest = Chat;
        Qbest = Qhat;
        Rbest = Rhat;
        mubest = muhat;
        Sigmabest = Sigmahat;
    end
     
    % Monitor convergence of log-likelihood
    if i>1 && (LL(i)-LL(i-1)) < (eps * abs(LL(i-1)))
        LLflag = LLflag + 1;
    else
        LLflag = 0;
    end  
    
    % Terminate EM algorithm if no sufficient reduction in log-likelihood
    % for 10 successive iterations
    if LLflag == 10
        break;
    end
            
        
    
    
%-------------------------------------------------------------------------%
%                               M-step                                    %
%-------------------------------------------------------------------------%



%@@@@ Update parameter estimates @@@@%

% Unconstrained estimates 
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
            sum_Pb_all = sum(sum_Pb,3);
            sum_CP_all = sum(sum_CP,3);
            Ahat = sum_CP_all / sum_Pb_all;
            if any(isnan(Ahat(:))|isinf(Ahat(:)))
                Ahat = sum_CP_all * pinv(sum_Pb_all);
            end
            Ahat = repmat(Ahat,[1,1,M]);
        elseif equal.A
            lhs = zeros(p*r*r);
            rhs = zeros(r,p*r);
            for j = 1:M
                Qinv_j = myinv(Qhat(1:r,1:r,j));
                lhs = lhs + kron(sum_Pb(:,:,j),Qinv_j);
                rhs = rhs + Qinv_j * sum_CP(:,:,j);
            end
            Ahat = lhs\rhs(:);
            if any(isnan(Ahat))|| any(isinf(Ahat))
                Ahat = pinv(lhs) * rhs(:);
            end
            Ahat = repmat(reshape(Ahat,r,p*r),1,1,M);
        else
            for j = 1:M
                A_j = sum_CP(:,:,j) / sum_Pb(:,:,j);
                if any(isnan(A_j(:)) | isinf(A_j(:)))
                    A_j = sum_CP(:,:,j) * pinv(sum_Pb(:,:,j));
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
            mat = kron(sum_Pb(:,:,j),Qinv_j);
            vec = reshape(Qinv_j * sum_CP(:,:,j),p*r^2,1);            
            A_j = zeros(p*r^2,1);
            A_j(fixed_A{j}(:,1)) = fixed_A{j}(:,2);
            A_j(free) = mat(free,free)\vec(free);
            if any(isnan(A_j)|isinf(A_j))
                A_j(free) = pinv(mat(free,free)) * vec(free);
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
                    sum_Pbj = sum_Pb(:,:,j);
                    sum_CPj = sum_CP(:,:,j);
                    % Projected gradient on Q-function holding matrix Q fixed
                    [A_j,~] = PG_A(A_j,sum_CPj,sum_Pbj,Q_j,scale.A,fixed_A{j});
                    Ahat(:,:,j) = A_j;
                end
            end
            if equal.A
                Ahat = repmat(Ahat(:,:,1),[1,1,M]);
                break
            end 
        end
    end
        
            
    %@@@@@ Update C
    Chatold = Chat;
    if equal.C
        Chat = zeros(N,p*r);
        lhs = sum(sum_MP(:,:,:),3);
        rhs = sum(sum_Mxy(:,:),3).';
        Chat(:,1:r) = rhs/lhs;
        Chat = repmat(Chat,1,1,M);
    else
        for j=1:M
            C_j = (sum_Mxy(:,:,j).')/sum_MP(:,:,j);
           if any(isnan(C_j(:)))|| any(isinf(C_j(:)))
                C_j = (sum_Mxy(:,:,j).') * pinv(sum_MP(:,:,j));
           end
        Chat(:,1:r,j) = C_j;
        end
    end
    Chat(fixed.C(:,1)) = fixed.C(:,2);    
    % If scale constraints on C and unconstrained estimate C^ does not
    % satisfy them, calculate constrained C^ by projected gradient holding
    % R^ fixed with unconstrained C^ as starting point
    if ~isempty(scale.C)
        for j=1:M
            nrm = sqrt(sum(Chat(:,:,j).^2));
            valid = (abs(nrm-scale.C) <= 1e-8);
            if ~all(valid)
                C_j = Chatold(:,:,j);
                sum_Mxyj = sum_Mxy(:,:,j);
                sum_MPj = sum_MP(:,:,j);
                Chat(:,:,j) = PG_C(C_j,Rhat,sum_yy,sum_Mxyj,sum_MPj,scale.C);
            end
            if equal.C
                Chat = repmat(Chat(:,:,1),1,1,M);
                break
            end
        end
    end
      
    %@@@@@ Update Q
    Qhatold = Qhat;
    for j=1:M
        A_j = Ahat(1:r,:,j);
        sum_CPj = sum_CP(:,:,j);
        sum_Pj = sum_P(:,:,j);
        sum_Pbj = sum_Pb(:,:,j);
        Q_j = (sum_Pj - (sum_CPj * A_j.') - (A_j * sum_CPj.') + ...
            (A_j * sum_Pbj * A_j'))/(T-1);
        Q_j = 0.5 * (Q_j+Q_j.');
        Qhat(1:r,1:r,j) = Q_j;
    end
    if equal.Q
        Qhat = repmat(mean(Qhat,3),1,1,M);
    end
    Qhat(fixed.Q(:,1)) = fixed.Q(:,2);
    % Regularize Q if needed
    for j = 1:M
        Q_j = Qhat(1:r,1:r,j);
        eigval = eig(Q_j);
        if min(eigval) < max(abstol,max(eigval)*reltol)
            if verbose
                warning('Q%d ill-conditioned and/or nearly singular. Regularizing.',j);
            end
            Qhat(1:r,1:r,j) = regfun(Q_j,abstol,reltol);
        end
    end 
    Qhat(fixed.Q(:,1)) = fixed.Q(:,2);
    % Make sure that update increases Q-function. If not, do not update
    Qvalold = Q_obs(Ahat,Chat,Qhatold,Rhat,muhat,Sigmahat,Pihat,Zhat,p,T,...
        Ms,x0,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,sum_P,sum_Pb,sum_yy);
    Qval = Q_obs(Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,p,T,...
        Ms,x0,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,sum_P,sum_Pb,sum_yy);
    if Qval < Qvalold
        Qhat = Qhatold;
    end
    
    %@@@@@ Update R
    Rhatold = Rhat;
    Rhat = sum_yy;
    for j=1:M
        C_j = Chat(:,1:r,j);
        sum_MPj = sum_MP(:,:,j);
        sum_Mxyj = sum_Mxy(:,:,j);
        Rhat = Rhat -  (C_j*sum_Mxyj) - (C_j*sum_Mxyj)' + (C_j*sum_MPj*C_j');
    end
    Rhat = Rhat / T;
    Rhat = 0.5 * (Rhat+Rhat');
    Rhat(fixed.R(:,1)) = fixed.R(:,2);
    % Regularize R if needed
    eigval = eig(Rhat);
    if min(eigval) < max(abstol,max(eigval)*reltol)
        if verbose
            warning('R ill-conditioned and/or nearly singular. Regularizing.');
        end
        Rhat = regfun(Rhat,abstol,reltol);
        Rhat(fixed.R(:,1)) = fixed.R(:,2);
    end 
    % Make sure that parameter update increases Q-function
    % If not, do not update parameter estimate 
    Qvalold = Q_obs(Ahat,Chat,Qhat,Rhatold,muhat,Sigmahat,Pihat,Zhat,p,T,...
        Ms,x0,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,sum_P,sum_Pb,sum_yy);
    Qval = Q_obs(Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,p,T,...
        Ms,x0,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,sum_P,sum_Pb,sum_yy);
    if Qval < Qvalold
        Rhat = Rhatold;
    end

    %@@@@@ Update mu 
    muhat = reshape(x0,r,p,M); % unconstrained solution
    muhat = repmat(squeeze(mean(muhat,2)),p,1); 
    % Assume E(x(1,j))=E(x(0,j))=...=E(x(1-p+1,j) for j=1:M
    if equal.mu
        muhat = repmat(mean(muhat,2),1,M);
    end
    muhat(fixed.mu(:,1)) = fixed.mu(:,2);
    
    %@@@@@ Update Sigma 
    Sigmahatold = Sigmahat;
    for j = 1:M
        mu_j = muhat(:,j);
        B_j = P0(:,:,j) - x0(:,j) * mu_j.' - ...
            mu_j * x0(:,j).' + (mu_j * mu_j.');        
        S_j = mean(B_j(Sigmamask),3);
        S_j = 0.5 * (S_j+S_j.');
        Sigmahat(:,:,j) = kron(eye(p),S_j);
    end
    if equal.Sigma
        Sigmahat = repmat(mean(Sigmahat,3),1,1,M);
    end
    Sigmahat(fixed.Sigma(:,1)) = fixed.Sigma(:,2);
    % Enforce semi-positive definiteness if needed 
    for j = 1:M
        S_j = Sigmahat(1:r,1:r,j);
        eigval = eig(S_j);
        if ~all(eigval >= 0) 
            if verbose
                warning('Sigma%d non semi-positive definite. Regularizing.',j);
            end
        S_j = regfun(S_j,0,0); 
        Sigmahat(:,:,j) = kron(eye(p),S_j);
        end
    end
    Sigmahat(fixed.Sigma(:,1)) = fixed.Sigma(:,2);
    % Make sure that parameter update increases Q-function
    % If not, do not update parameter estimate 
    Qvalold = Q_obs(Ahat,Chat,Qhat,Rhat,muhat,Sigmahatold,Pihat,Zhat,p,T,...
        Ms,x0,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,sum_P,sum_Pb,sum_yy);
    Qval = Q_obs(Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,p,T,...
        Ms,x0,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,sum_P,sum_Pb,sum_yy);
    if Qval < Qvalold
        Sigmahat = Sigmahatold;
    end
     
    
    
end % END MAIN LOOP



%-------------------------------------------------------------------------%
%                               Output                                    %
%-------------------------------------------------------------------------%

% Return best estimates (i.e. with highest log-likelihood) 
% after reshaping them in compact form
Ahat = reshape(Abest(1:r,:,:),r,r,p,M);
Chat = Cbest(:,1:r,:);
Qhat = Qbest(1:r,1:r,:);
Rhat = Rbest;
muhat = mubest(1:r,:);
Sigmahat = Sigmabest(1:r,1:r,:);
LL = LL(1:i);

end

