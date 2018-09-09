function [Mf,Ms,Sf,Ss,xf,xs,Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,LL] = ... 
    switch_obs(y,M,p,r,A,C,Q,R,mu,Sigma,Pi,Z,control,equal,fixed,scale)

%--------------------------------------------------------------------------
% Title:    Parameter estimation and inference in state-space models with regime switching
%            (switching observations model)
%
% Function: Infer hidden state vectors and regmes by switching Kalman filtering/smoothing
%            (aka Hamilton filtering or Kim filtering) and estimate model parameters by 
%            maximum likelihood (EM algorithm).
%
% Usage:    [Mf,Ms,Sf,Ss,xf,xs,Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,LL] = ... 
%               switch_obs(y,M,p,A,C,Q,R,mu,Sigma,Pi,Z,control,equal,fixed,scale)
%
% Inputs:   y - Time series (size NxT)
%           M - number of regimes
%           p - order of VAR model for state vector 
%           A - Initial estimates of VAR matrices A(l,j) in system equation 
%               x(t,j) = sum(l=1:p) A(l,j) x(t-l,j) + v(t,j), j=1:M (size rxrxpxM)  
%           C - Initial estimates of observation matrices C(j) in equation 
%               y(t) = C(j) x(t,j) + w(t), j=1:M (size NxrxM)
%           Q - Initial estimates of state noise covariance Cov(v(t,j)) (size rxrxM)
%           R - Pilot estimate of observation noise covariance Cov(w(t)) (size NxN)                  
%           mu - Pilot estimate of mean state mu(j)=E(x(t,j)) for t=1:p (size rxM) 
%           Sigma - Pilot estimate of covariance Sigma(j)=Cov(x(t,j)) for t=1:p (size rxrxM)           
%           Pi - Pilot state probability (size Mx1)
%           Z - Pilot Markov transition probability matrix (size MxM) 
%           control - optional struct variable with fields: 
%                   'eps': tolerance for EM termination; defaults to 1e-8
%                   'ItrNo': number of EM iterations; dfaults to 100 
%                   'beta0': initial inverse temperature parameter for deterministic annealing; default 1 
%                   'betarate': decay rate for temperature; default 1 
%                   'safe': if true, regularizes variance matrices to be well-conditioned 
%                    before taking inverse. If false, no regularization (faster but less safe)
%                   'abstol': absolute tolerance for eigenvalues in matrix inversion (only effective if safe = true)
%                   'reltol': relative tolerance for eigenvalues in matrix inversion
%                    = inverse condition number (only effective if safe = true)
%            equal - optional struct variable with fields:
%                   'A': if true, VAR transition matrices A(l,j) are equal across regimes j=1,...,M
%                   'C': if true, observation matrices C(j) are equal across regimes
%                   'Q': if true, VAR innovation matrices Q(j) are equal across regimes
%                   'mu': if true, initial mean state vectors mu(j) are equal across regimes
%                   'Sigma': if true, initial variance matrices Sigma(j) are equal across regimes
%            fixed - optional struct variable with fields 'A','C','Q','R','mu','Sigma'.
%                   If not empty, each field must contain a matrix with 2 columns, the first for 
%                   the location of fixed coefficients and the second for their values. 
%            scale - optional struct variable with fields:
%                   'A': upper bound for norm of eigenvalues of A matrices. Must be in (0,1).
%                   'C': value of the (euclidean) column norms of the matrices C(j). Must be positive.
%                   
% Outputs:  Mf - State probability estimated by switching Kalman Filter (size MxT)
%           Ms - State probability estimated by switching Kalman Smoother (size MxT)
%           Sf - Estimated states (Kalman Filter) 
%           Ss - Estimated states (Kalman Smoother) 
%           xf - Filtered state vector (size rxT)
%           xs - Smoothed state vector (size MxT)
%           Ahat - Estimated system matrix (size rxrxpxM)
%           Chat - Estimated observation matrix (size Nxr)
%           Qhat - Estimated state noise covariance (size rxrxM)
%           Rhat - Estimated observation noise covariance (size NxN)
%           muhat - Estimated initial mean of state vector (size rxM)
%           Sigmahat - Estimated initial variance of state vector (size rxrxM)
%           LL  - Sequence of log-likelihood values
%                    
% Author:   David Degras, david.degras@umb.edu
%           University of Massachusetts Boston
%
% Contributors: Ting Chee Ming, cmting@utm.my
%           Siti Balqis Samdin
%           Centre for Biomedical Engineering, Universiti Teknologi Malaysia.
%               
% Version:  September 4, 2018
%--------------------------------------------------------------------------





%-------------------------------------------------------------------------%
%                           Initialization                                %
%-------------------------------------------------------------------------%

narginchk(4,16);

% Data dimensions
[N,T] = size(y);


% x(t,j): state vector for j-th process at time t (size r)
% x(t) = x(t,1),...,x(t,M): state vector for all processes at time t (size M*r)
% X(t,j) = x(t,j),...,x(t-p+1,j)): state vector for process j at times t,...,t-p+1 (size p*r)
% X(t) = X(t,1),...,X(t,M): state vector for all processes at times t,...,t-p+1 (size M*p*r)
% Assumption: initial vectors x(1),...,x(1-p+1) are iid ~ N(mu,Sigma)


%@@@@@ Initialize optional arguments if not specified

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



%@@@@@ Initialize estimators @@@@@%

if nargin == 4
    [A,C,Q,R,mu,Sigma,Pi,Z,~] = init_obs(y,M,p,r);
end

[Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,fixed,skip,equal,eps,...
    ItrNo,beta0,betarate,safe,abstol,reltol,verbose,scale] = ... 
    preproc_obs(M,N,p,r,A,C,Q,R,mu,Sigma,Pi,Z,fixed,equal,control,scale);

% Initial parameters 'A','C',... are expanded from r-space (x(t)) to
% (p*r)-space (X(t)). The new parameters 'Ahat','Chat',... have dimensions:
% Ahat: (p*r)x(p*r)xM, Chat: Nx(p*r)xM, Qhat: (p*r)x(p*r)xM, Rhat: NxN,
% muhat: (p*r)xM, Sigmahat: (p*r)x(p*r)xM. These parameters respect the
% fixed coefficients and equality constraints (either default values or
% user-specified ones).

% The structure 'fixed' has fields 'fixed.A',... (one field per model
% parameter). Each field is a matrix with two columns containing the
% locations and values of fixed coefficients in the corresponding
% parameter.

% The structure 'equal' has fields 'equal.A',... representing equality
% constraints on parameters across regimes j=1,...,M. By default, equality
% constraints are: A: false, C: false, Q: false, (R: not applicable), mu:
% true, Sigma: true. In other words, only the initial parameters mu and
% Sigma are assumed to be common across regimes. If provided, user values
% will override default values.

% Various control parameters ('eps','ItrNo',...) are set either to their
% default values or to user-specified values through argument 'control'.




%@@@@@ Initialize other quantities @@@@@%

LL = zeros(1,ItrNo); % Log-likelihood
LLbest = -Inf; % best attained log-likelihood 
LLflag = 0; % counter for monitoring progress of log-likelihood 
sum_yy = y * y.'; % sum(t=1:T) y(t)*y(t)'
beta = beta0; % initial temperature for deterministic annealing 

% Indices and values of fixed coefficients in A(j) = (A(1,j),...,A(M,j)) in
% 2 column-matrix format (needed for projected gradient with A in M-step)
fixed_A = cell(M,1);
if ~isempty(fixed.A)
    Atmp = NaN(r,p*r,M);
    Atmp(fixed.A(:,1)) = fixed.A(:,2); 
    for j = 1:M
        idx = find(~isnan(Atmp(:,:,j)));
        fixed_A{j} = [idx,Atmp(idx)];        
    end  
end
% Indices and values of fixed coefficients in C(j) in 2 column-matrix format 
% (needed for projected gradient with C in M-step)
fixed_C = cell(M,1);
if ~isempty(fixed.C)
    Ctmp = NaN(N,r,M);
    Ctmp(fixed.C(:,1)) = fixed.C(:,2);  
    for j = 1:M
        idx = find(~isnan(Ctmp(:,:,j)));
        fixed_C{j} = [idx,Ctmp(idx)];
    end  
end

% Mask of rxr diagonal blocks in a (p*r)x(p*r) matrix
% (used to update Sigma in M-step)
Sigmamask = reshape(find(kron(eye(p),ones(r))),r,r,p);

% Expanded parameters
Abig = repmat(diag(ones((p-1)*r,1),-r),[1,1,M]);
Cbig = zeros(N,p*r,M);
Qbig = zeros(p*r,p*r,M);
% mubig = zeros(p*r,M);
Sbig = zeros(p*r,p*r,M);




for i=1:ItrNo
    
   

%-------------------------------------------------------------------------%
%                                 E-step                                  %
%-------------------------------------------------------------------------%


    % Expand parameters
    Abig(1:r,:,:) = Ahat;
    Cbig(:,1:r,:) = Chat;
    Qbig(1:r,1:r,:) = Qhat;
    mubig = repmat(muhat,[p,1]);
    for j = 1:M
        Sbig(:,:,j) = kron(eye(p),Sigmahat(:,:,j));
    end
    
    % Kim/Hamilton filtering and smoothing
    [Mf,Ms,xf,xs,x0,P0,Loglik,sum_CP,sum_MP,sum_Ms2,sum_Mxy,sum_P,sum_Pb] = ... 
        skfs_obs(y,M,p,Abig,Cbig,Qbig,Rhat,mubig,Sbig,Pihat,Zhat, ...
            beta,safe,abstol,reltol,verbose);
            
    % Log-likelihood
    LL(i) = Loglik; 
    if verbose
        fprintf('Iteration-%d   Log-likelihood = %g\n',i,LL(i));
        Qval = Q_obs(Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,p,T,...
            Ms,x0,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,sum_P,sum_Pb,sum_yy);
        fprintf('Iteration-%d   Q-function = %g (before M-step)\n',i,Qval);
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
        Pibest = Pihat;
        Zbest = Zhat;
        Mfbest = Mf;
        Msbest = Ms;
        xfbest = xf;
        xsbest = xs;
    end
     
    % Monitor progress of log-likelihood
    if i>1 && LL(i)-LL(i-1) < eps * abs(LL(i-1))
        LLflag = LLflag + 1;
    else
        LLflag = 0;
    end  
    % Terminate EM algorithm if no sufficient reduction in log-likelihood
    % for 10 successive iterations
    if LLflag == 10
        break;
    end
        
    % Update inverse temperature parameter (DAEM)
    beta = beta^betarate;
    
        
    
    
%-------------------------------------------------------------------------%
%                               M-step                                    %
%-------------------------------------------------------------------------%




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
    
    % Unconstrained solution
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
   
        % Check that parameter update actually increases Q-function 
        % If not, keep previous parameter estimate 
        Qvalold = Q_obs(Ahatold,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,p,T,...
            Ms,x0,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,sum_P,sum_Pb,sum_yy);
        Qval = Q_obs(Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,p,T,...
            Ms,x0,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,sum_P,sum_Pb,sum_yy);
        if Qval < Qvalold
            Ahat = Ahatold;
        end
    end
    
    %@@@@@ Update C @@@@@%
        
    if ~skip.C
%         Chatold = Chat;
      
        for j = 1:M
            if equal.C                
                sum_MPj = sum(sum_MP(:,:,:),3);
                sum_Mxyj = sum(sum_Mxy(:,:),3);
            else
                sum_MPj = sum_MP(:,:,j);
                sum_Mxyj = sum_Mxy(:,:,j);
            end
            
            % If no fixed coefficient and no scale constraints on C,
            % calculate unconstrained solution
            if isempty(fixed_C{j}) && isempty(scale.C)
                Ctmp = (sum_Mxyj')/sum_MPj;
                if any(isnan(Ctmp(:)) | isinf(Ctmp(:)))
                    Ctmp = (sum_Mxyj') * pinv(sum_MPj);
                end   
                
            % If fixed coefficient constraints but no scale constraints on C,
            % maximizer can be found in closed form (keep R fixed)
            elseif ~isempty(fixed_C{j}) && isempty(scale.C)              
                % Transform matrix trace problem into quadratic form problem
                lhs = kron(sum_MPj,myinv(Rhat));
                rhs = Rhat\(sum_Mxyj');
                % Remove rows/columns in LHS, coefficients in RHS
                % associated with fixed coefficients in C
                idx = setdiff(1:N*r,fixed_C{j}(:,1)); % index of free coefficients in C(j)
                idx = reshape(idx,[],1);
                lhs = lhs(idx,idx);
                rhs = rhs(idx);
                Ctmp = zeros(N,r);
                Ctmp(fixed_C{j}(:,1)) = fixed_C{j}(:,2); % fill in fixed coefficients
                Ctmp(idx) = lhs\rhs;
                if any(isnan(Ctmp(:))|isinf(Ctmp(:)))
                    Ctmp(idx) = pinv(lhs) * rhs;
                end
                
            % If scale constraints on C (and eventually fixed coefficient
            % constraints too), optimize the Q-function under these constraints
            % by projected gradient while holding Rhat fixed
            else
                C_j = Chat(:,:,j); % Chatold(:,:,j);
                [Ctmp,~] = PG_C(C_j,sum_Mxyj,sum_MPj,Rhat,scale.C,fixed_C{j});
            end
            
            if equal.C
                Chat = repmat(Ctmp,1,1,M);
                break
            else
                Chat(:,:,j) = Ctmp;
            end
        end
             
        % Enforce fixed coefficient constraints (redundant because these
        % constraints have already been applied above)
        if ~isempty(fixed.C)
            Ctmp = Chat(:,1:r,:);       
            Ctmp(fixed.C(:,1)) = fixed.C(:,2);
            Chat(:,1:r,:) = Ctmp; 
        end
    end

    
    %@@@@@ Update Q @@@@@%
    
    if ~skip.Q
        % Store previous estimate
        Qhatold = Qhat;
        % Calculate unconstrained solution
        for j=1:M
            A_j = Ahat(:,:,j);
            sum_CPj = sum_CP(:,:,j);
            sum_Pj = sum_P(:,:,j);
            sum_Pbj = sum_Pb(:,:,j);
            Q_j = (sum_Pj - (sum_CPj * A_j.') - (A_j * sum_CPj.') + ...
                (A_j * sum_Pbj * A_j'))/(T-1);
            Qhat(:,:,j) = 0.5 * (Q_j+Q_j.');            
        end
        % Apply equality constraints 
        if equal.Q
            Qhat = repmat(mean(Qhat,3),1,1,M);
        end
        % Apply fixed coefficient constraints (currently, only diagonality
        % constraints are allowed on covariance matrices)
        if ~isempty(fixed.Q)
            Qhat(fixed.Q(:,1)) = fixed.Q(:,2);       
        end
        % Check conditioning and positive definiteness of Q
        % Regularize if needed
        for j = 1:M
            eigval = eig(Qhat(:,:,j));
            if min(eigval) < max(abstol,max(eigval)*reltol)             
                if verbose
                    warning('Q%d ill-conditioned and/or nearly singular. Regularizing.',j);
                end
                Qhat(:,:,j) = regfun(Qhat(:,:,j),abstol,reltol);
            end
        end 
        % Apply fixed coefficient constraints again 
        % (Redundant under current diagonality constraints: whether Qhat(j)
        % has been regularized or not, it remains diagonal)
        if ~isempty(fixed.Q)
            Qhat(fixed.Q(:,1)) = fixed.Q(:,2);       
        end
        % Check that update increases Q-function. If not, do not update
        Qvalold = Q_obs(Ahat,Chat,Qhatold,Rhat,muhat,Sigmahat,Pihat,Zhat,p,T,...
            Ms,x0,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,sum_P,sum_Pb,sum_yy);
        Qval = Q_obs(Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,p,T,...
            Ms,x0,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,sum_P,sum_Pb,sum_yy);
        if Qval < Qvalold
            Qhat = Qhatold;
        end
    end
    
    
    %@@@@@ Update R @@@@@%
    
    if ~skip.R
        Rhatold = Rhat;
        % Unconstrained estimate
        Rhat = sum_yy;
        for j=1:M
            C_j = Chat(:,:,j);
            sum_MPj = sum_MP(:,:,j);
            sum_Mxyj = sum_Mxy(:,:,j);
            Rhat = Rhat -  (C_j*sum_Mxyj) - (C_j*sum_Mxyj)' + (C_j*sum_MPj*C_j');
        end
        Rhat = Rhat / T;
        Rhat = 0.5 * (Rhat+Rhat');
        % Apply fixed coefficient constraints
        if ~isempty(fixed.R)
            Rhat(fixed.R(:,1)) = fixed.R(:,2);  
        end
        % Regularize R if needed
        eigval = eig(Rhat);
        if min(eigval) < max(abstol,max(eigval)*reltol)
            if verbose
                warning('R ill-conditioned and/or nearly singular. Regularizing.');
            end
            Rhat = regfun(Rhat,abstol,reltol);
         end 
        % Apply fixed coefficient constraints again 
        % (Redundant under current diagonality constraints: whether Qhat(j)
        % has been regularized or not, it remains diagonal)
        if ~isempty(fixed.R)
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
    end
    
    
    %@@@@@ Update mu @@@@@%
    
    if ~skip.mu
        % Unconstrained solution
        muhat = reshape(x0,r,p,M); 
        muhat = squeeze(mean(muhat,2)); % size rxM
        % Assume E(x(1,j))=E(x(0,j))=...=E(x(1-p+1,j) for j=1:M
        if equal.mu && equal.Sigma
            muhat = repmat(mean(muhat,2),1,M);
        elseif equal.mu
            lhs = zeros(r);
            rhs = zeros(r,1);
            for j = 1:M
                Sinv_j = myinv(Sigmahat(:,:,j));
                lhs = lhs + Sinv_j;
                rhs = rhs + Sinv_j * muhat(:,j);
            end
            muhat = lhs\rhs;
            if any(isnan(muhat)|isinf(muhat))
                muhat = pinv(lhs) * rhs;
            end
            muhat = repmat(muhat,1,M);  
        end
        % Apply fixed coefficient constraints
        if ~isempty(fixed.mu)
            muhat(fixed.mu(:,1)) = fixed.mu(:,2);
        end
        mubig = repmat(muhat,[p,1]);
    end
    
    %@@@@@ Update Sigma @@@@@%
    
    if ~skip.Sigma
        Sigmahatold = Sigmahat;
        % Unconstrained solution
        for j = 1:M
            mu_j = mubig(:,j);
            B_j = P0(:,:,j) - x0(:,j) * mu_j.' - ...
                mu_j * x0(:,j).' + (mu_j * mu_j.');        
            S_j = mean(B_j(Sigmamask),3);
            Sigmahat(:,:,j) = 0.5 * (S_j+S_j.');            
        end
        % Apply equality constraints
        if equal.Sigma
            Sigmahat = repmat(mean(Sigmahat,3),1,1,M);
        end
        % Apply fixed coefficient constraints
        if ~isempty(fixed.Sigma)
             Sigmahat(fixed.Sigma(:,1)) = fixed.Sigma(:,2);
        end
        % Regularize if needed 
        for j = 1:M
            eigval = eig(Sigmahat(:,:,j));
            if any(eigval < abstol)                
                if verbose
                    warning('Sigma%d nearly singular. Regularizing.',j);
                end
                Sigmahat(:,:,j) = regfun(Sigmahat(:,:,j),abstol,0); 
            end
        end
        % Apply fixed coefficient constraints again 
        % (Redundant for currently supported constraints: diagonal
        % structure)
       if ~isempty(fixed.Sigma)
             Sigmahat(fixed.Sigma(:,1)) = fixed.Sigma(:,2);
        end
        % Make sure that parameter update increases Q-function
        % If not, do not update parameter estimate 
        Qvalold = Q_obs(Ahat,Chat,Qhat,Rhat,muhat,Sigmahatold,Pihat,Zhat,p,T,...
            Ms,x0,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,sum_P,sum_Pb,sum_yy);
        Qval = Q_obs(Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,p,T,...
            Ms,x0,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,sum_P,sum_Pb,sum_yy);
        if Qval < Qvalold
            Sigmahat = Sigmahatold;
        end
    end
    
    %@@@@@ Update Pi @@@@@%
    
    if ~skip.Pi
        Pihat = Ms(:,1);
    end
    
    
    %@@@@@ Update Z @@@@@%
    
    if ~skip.Z
        Zhat = sum_Ms2 ./ repmat(sum(sum_Ms2,2),1,M); 
        if ~isempty(fixed.Z)
            Zhat(fixed.Z(:,1)) = fixed.Z(:,2);
        end
    end
    
    % Evaluate and display Q-function value if required 
    if verbose
        Qval = Q_obs(Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,p,T,...
            Ms,x0,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,sum_P,sum_Pb,sum_yy);
        fprintf('Iteration-%d   Q-function = %g (after M-step)\n',i,Qval);
    end
        
    
end % END MAIN LOOP



%-------------------------------------------------------------------------%
%                               Output                                    %
%-------------------------------------------------------------------------%

% Return best estimates (i.e. with highest log-likelihood) 
% after reshaping them to original size
Ahat = reshape(Abest,r,r,p,M);
Chat = Cbest;
Qhat = Qbest;
Rhat = Rbest;
muhat = mubest;
Sigmahat = Sigmabest;
Pihat = Pibest;
Zhat = Zbest;

Mf = Mfbest;
Ms = Msbest;
[~,Sf] = max(Mf);
[~,Ss] = max(Ms);
xf = xfbest;
xs = xsbest;

LL = LL(1:i);

end

