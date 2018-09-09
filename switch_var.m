function [Mf,Ms,Sf,Ss,Ahat,Qhat,muhat,Sigmahat,Pihat,Zhat,LL] = ... 
    switch_var(y,M,p,A,Q,mu,Sigma,Pi,Z,control,equal,fixed,scale)

%--------------------------------------------------------------------------
% Title:    Parameter estimation and inference in state-space models with 
%           regime switching (switching vector autoregressive)
%
% Purpose:  Infer hidden state vectors and regimes by switching Kalman 
%           filtering/smoothing (aka Hamilton filtering or Kim filtering) 
%           and estimate model parameters by maximum likelihood (EM algorithm).
%
% Usage:    [Mf,Ms,Sf,Ss,xf,xs,Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,LL] = ... 
%               switch_var(y,M,p,A,Q,mu,Sigma,Pi,Z,control,equal,fixed,scale)
% 
%
% Inputs:   y - Time series (size NxT)
%           M - number of regimes (values) for switching variable S(t) 
%           p - order of VAR model for y(t)
%           A - Initial estimate of VAR matrices A(l,j) in model  
%               y(t) = sum(l=1:p) A(l,j) y(t-l) + v(t) conditional on S(t)=j, 
%               j=1:M (size rxrxpxM)  
%           Q - Initial estimate of system noise covariance Q(j)=V(v(t)|S(t)=j), 
%               j=1:M (size rxrxM)
%           mu - Initial estimate of mean state mu(j)=E(y(1)|S(1)=j), j=1:M  
%               (size rxM) 
%           Sigma - Initial estimate of state covariance Sigma(j)=V(y(1)|S(1)=j), 
%                j=1:M (size rxrxM) 
%           Pi - Initial estimate of probability Pi(j)=P(S(1)=j), j=1:M (size Mx1)
%           Z - Initial estimate of transition probabilities Z(i,j) = 
%               P(S(t)=j|S(t-1)=i), i,j=1:M (size MxM) 
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
%            equal - optional struct variable with fields:
%                   'A': if true, estimates of transition matrices A(l,j) 
%                       are equal across regimes j=1,...,M. Default = false
%                   'Q': if true, estimates of innovation matrices Q(j) are
%                       equal across regimes. Default = false
%                   'mu': if true, estimates of initial mean state vectors 
%                       mu(j) are equal across regimes. Default = true
%                   'Sigma': if true, estimates of initial variance matrices 
%                       Sigma(j) are equal across regimes. Default = true
%            fixed - optional struct variable with fields 'A','C','Q','R',
%                   'mu','Sigma'. If not empty, each field must be an array
%                   of the same dimension as the parameter. Numeric values 
%                   in the array are interpreted as fixed coefficients whereas 
%                   NaN's represent free coefficients. For example, a
%                   diagonal structure for 'R' would be specified as
%                   fixed.R = diag(NaN(N,1)). 
%            scale - optional struct variable with fields:
%                   'A': upper bound on norm of eigenvalues of A matrices. 
%                       Must be in (0,1) to guarantee stationarity of state process.
%                   
% Outputs:  Mf - State probability estimated by switching Kalman Filter (size MxT)
%           Ms - State probability estimated by switching Kalman Smoother (size MxT)
%           Sf - Estimated states (Kalman Filter) 
%           Ss - Estimated states (Kalman Smoother) 
%           Ahat - Estimated system matrix (size rxrxpxM)
%           Qhat - Estimated state noise covariance (size rxrxM)
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
% Version:  July 20, 2018
%--------------------------------------------------------------------------





%-------------------------------------------------------------------------%
%                           Initialization                                %
%-------------------------------------------------------------------------%


narginchk(3,13);

% Data dimensions
[r,T] = size(y);

% Initialize optional arguments if not specified
fixed0 = struct('C',eye(r),'R',1e-20 * eye(r));
if exist('fixed','var') && isstruct(fixed)
    fixed.C = fixed0.C;
    fixed.R = fixed0.R;
else
    fixed = fixed0;
end

equal0 = struct('mu',true,'Sigma',true);
if exist('equal','var') && isstruct(equal)
    equal.mu = equal0.mu;
    equal.Sigma = equal0.Sigma;
else
    equal = equal0;
end

if ~exist('control','var')
    control = struct();
end

if ~exist('scale','var')
    scale = struct();
end

% scale0 = struct('A',.999);
% if exist('scale','var') && isstruct(scale) && isfield(scale,'A')
%     scale0 = scale.A;
% end
% scale = scale0;


%@@@@ Initialize estimators @@@@%

if nargin == 3
    [A,~,Q,~,mu,Sigma,Pi,Z,~] = init_dyn(y,M,p,r,[],control,equal,fixed,scale);
end

[Ahat,~,Qhat,~,muhat,Sigmahat,Pihat,Zhat,fixed,skip,equal,...
    eps,ItrNo,beta0,betarate,~,abstol,reltol,verbose,scale] = ... 
    preproc_dyn(M,r,p,r,A,fixed.C,Q,fixed.R,mu,Sigma,Pi,Z,control,equal,fixed,scale);

% The structure 'fixed' has fields 'fixed.A',... (one field per model
% parameter). Each field is a matrix with two columns containing the
% locations and values of fixed coefficients in the corresponding
% parameter.

% The structure 'skip' specified whether a parameter (A,C,Q...) is entirely
% fixed, in which case its update in the M-step can be skipped altogether

% We assume that conditional on S(1:p), the initial vectors y(1),...,y(p)
% are independent and that conditional on S(t)=j (t=1:p), y(t) ~ N(mu(j),Sigma(j))



%@@@@ Initialize other quantities

LL = zeros(1,ItrNo); % Log-likelihood
LLbest = -Inf; % best attained value of log-likelihood 
LLflag = 0; % counter for convergence of of log-likelihood 
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
        if equal.A
            fixed_A = fixed_A(1);
            break
        end
    end  
end

% Container for checking eigenvalues of A 
Abig = diag(ones((p-1)*r,1),-r);
 

for i=1:ItrNo
    
   

%-------------------------------------------------------------------------%
%                               E-step                                    %
%-------------------------------------------------------------------------%


    
    % Kim/Hamilton filtering and smoothing
    [Mf,Ms,L,sum_Ms2] = ...
        skfs_var(y,M,p,Ahat,Qhat,muhat,Sigmahat,Pihat,Zhat,beta);

    % Log-likelihood
    LL(i) = L; 

    % Quantities required for Q-function and for M-step
    sum_MP = zeros(r,r,M);
    sum_MPb = zeros(p*r,p*r,M);
    sum_MCP = zeros(r,p*r,M);
    for t = p+1:T
        P = (y(:,t) * y(:,t)');
        Yb = reshape(y(:,t-1:-1:t-p),p*r,1);
        Pb = (Yb * Yb');
        CP = y(:,t) * Yb';
        for j = 1:M
            sum_MP(:,:,j) = sum_MP(:,:,j) + Ms(j,t) * P;
            sum_MPb(:,:,j) = sum_MPb(:,:,j) + Ms(j,t) * Pb;
            sum_MCP(:,:,j) = sum_MCP(:,:,j) + Ms(j,t) * CP;
        end
    end
    
            
    % Evaluate Q-function before M-step and display LL & Q if required        
    if verbose
        fprintf('Iteration-%d   Log-likelihood = %g\n',i,LL(i));
        Qval = Q_var(Ahat,Qhat,muhat,Sigmahat,Pihat,Zhat,p,T,Ms,...
            sum_MCP,sum_MP,sum_MPb,sum_Ms2,y);
        fprintf('Q-function before M-step = %g\n',Qval);
    end
       
    % Check if current solution is best to date
    if (LL(i) > LLbest)
        LLbest = LL(i);
        Abest = Ahat;
        Qbest = Qhat;
        mubest = muhat;
        Sigmabest = Sigmahat;
        Pibest = Pihat;
        Zbest = Zhat;
        Mfbest = Mf;
        Msbest = Ms;
    end
     
    % Monitor progress of log-likelihood
    if (i>1 &&  LL(i)-LL(i-1) < eps * abs(LL(i-1)))
        LLflag = LLflag + 1;
    else
        LLflag = 0;
    end  
    % Terminate EM algorithm if no sufficient increase in log-likelihood
    % for 10 iterations in a row
    if LLflag == 10
        break;
    end
        
    % Update inverse temperature parameter (DAEM)
    beta = beta^betarate;
    
    
    
%-------------------------------------------------------------------------%
%                               M-step                                    %
%-------------------------------------------------------------------------%



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
            Qinv_j = myinv(Qhat(:,:,j));
            % Matrix problem min(X) trace(W(-2*B1*X' + X*B2*X')) 
            % (under fixed coefficient constraints) becomes vector problem 
            % min(x) x' kron(B2,W) x - 2 x' vec(W*B1)
            % with X = A(j), x = vec(A(j)), W = Q(j)^(-1), B1 = sum_MCP(j),
            % and B2 = sum_MPb(j) (remove fixed entries in x)
            lhs = kron(sum_MPb(:,:,j),Qinv_j);
            lhs = lhs(free,:);
            rhs = Qinv_j * sum_MCP(:,:,j);            
            rhs = rhs(free);
            A_j = reshape(lhs\rhs,r,p*r);
            if any(isnan(A_j(:))|isinf(A_j(:)))
                A_j = reshape(pinv(lhs)*rhs,r,p*r);
            end
            Ahat(:,:,j) = A_j;
        end
    end
    
    % Check eigenvalues of estimate and regularize if needed      
    if ~skip.A
        for j = 1:M
            % Check eigenvalues
            Abig(1:r,:,:) = Ahat(:,:,j);
            eigval = eig(Abig);
            if any(abs(eigval) > scale.A)
                if verbose
                    warning(['Eigenvalues of A%d greater than %f.',...
                        ' Regularizing.'],j,scale.A)
                end
                % Case: regularize with no fixed coefficients constraints
                if isempty(fixed_A{j})
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
        Qvalold = Q_var(Ahatold,Qhat,muhat,Sigmahat,Pihat,Zhat,p,T,Ms,...
            sum_MCP,sum_MP,sum_MPb,sum_Ms2,y);
        Qval = Q_var(Ahat,Qhat,muhat,Sigmahat,Pihat,Zhat,p,T,Ms,...
            sum_MCP,sum_MP,sum_MPb,sum_Ms2,y);
        if Qval < Qvalold
            Ahat = Ahatold;
        end
    end
 
  
    
    %@@@@@ Update Q @@@@@%
    
    % Unconstrained solution
    if ~skip.Q        
        Qhatold = Qhat; 
        Qhat = zeros(r,r,M);
        sum_M = sum(Ms(:,p+1:T),2);
        for j=1:M
            if sum_M(j) == 0
                Qhat(:,:,j) = eye(r);
                continue
            end                
            A_j = Ahat(:,:,j);
            sum_MPj = sum_MP(:,:,j);
            sum_MCPj = sum_MCP(:,:,j);
            sum_MPbj = sum_MPb(:,:,j);                
            Q_j = (sum_MPj - (sum_MCPj * A_j.') - ...
                (A_j * sum_MCPj.') + A_j * sum_MPbj * A_j.') / sum_M(j);
            Qhat(:,:,j) = 0.5 * (Q_j + Q_j');
        end
        if equal.Q
            Qtmp = zeros(r);
            for j = 1:M
                Qtmp = Qtmp + (sum_M(j)/(T-p)) * Qhat(:,:,j);
            end
            Qhat = repmat(Qtmp,1,1,M);
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

        % Check that estimate Qhat increases Q-function. If not, keep
        % estimate from previous iteration
        Qvalold = Q_var(Ahat,Qhatold,muhat,Sigmahat,Pihat,Zhat,p,T,Ms,...
            sum_MCP,sum_MP,sum_MPb,sum_Ms2,y);
        Qval = Q_var(Ahat,Qhat,muhat,Sigmahat,Pihat,Zhat,p,T,Ms,...
            sum_MCP,sum_MP,sum_MPb,sum_Ms2,y);
        if Qval < Qvalold
            Qhat = Qhatold;
        end        
    end


    
    %@@@@@ Update mu 
    if ~skip.mu       
        muhatold = muhat;
        if equal.mu && equal.Sigma
            muhat = repmat(mean(y(:,1:p),2),1,M);
        elseif equal.mu
            lhs = zeros(r,r);
            rhs = zeros(r,1);
            for j=1:M
                Sinv_j = myinv(Sigmahat(:,:,j));         
                lhs = lhs + Sinv_j;
                rhs = rhs + Sinv_j * y(:,1:p) * Ms(j,1:p)';
            end
            muhat = lhs\rhs;
            if any(isnan(muhat) | isinf(muhat))
                muhat = pinv(lhs)*rhs;
            end
            muhat = repmat(muhat,1,M);        
        else
            muhat = zeros(r,M);
            for j = 1:M
                sum_Mj = sum(Ms(j,1:p));                
                if sum_Mj > 0
                    muhat(:,j) = (y(:,1:p) * Ms(j,1:p)') / sum_Mj;
                end
            end            
        end
        
        % Apply fixed coefficient constraints
        if ~isempty(fixed.mu)
            muhat(fixed.mu(:,1)) = fixed.mu(:,2);
        end

        % Check that muhat increases Q-function. If not, keep estimate from
        % previous iteration
        Qvalold = Q_var(Ahat,Qhat,muhatold,Sigmahat,Pihat,Zhat,p,T,Ms,...
            sum_MCP,sum_MP,sum_MPb,sum_Ms2,y);
        Qval = Q_var(Ahat,Qhat,muhat,Sigmahat,Pihat,Zhat,p,T,Ms,...
            sum_MCP,sum_MP,sum_MPb,sum_Ms2,y);
        if Qval < Qvalold
            muhat = muhatold;
        end
    end
    
    %@@@@@ Update Sigma 
    if ~skip.Sigma
        Sigmahatold = Sigmahat;
        Sigmahat = zeros(r,r,M);
        % sum(t=1:p) P(S(t)=j|y(1:T))
        sum_M = sum(Ms(:,1:p),2);
        for j = 1:M
            if sum_M(j) == 0
                Sigmahat(:,:,j) = eye(r);
                continue
            end
            B_j = y(:,1:p) - repmat(muhat(:,j),1,p);
            S_j = (B_j * diag(Ms(j,1:p)) * B_j') / sum_M(j);
            Sigmahat(:,:,j) = 0.5 * (S_j + S_j'); 
        end
        if equal.Sigma
            Stmp = zeros(r);
            for j = 1:M
                Stmp = Stmp + (sum_M(j)/p) * Sigmahat(:,:,j);
            end
            Sigmahat = repmat(Stmp,1,1,M);
        end
        
        % The above estimates of Sigma(j) have rank p at most (VAR order). 
        % If p < r (time series dimension), they are not invertible --> 
        % set off-diagonal terms to zero
        if p < r
            for j = 1:M
                Sigmahat(:,:,j) = diag(diag(Sigmahat(:,:,j)));
            end
        end
        
        % Enforce fixed coefficient constraints
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
        Qvalold = Q_var(Ahat,Qhat,muhat,Sigmahatold,Pihat,Zhat,p,T,Ms,...
            sum_MCP,sum_MP,sum_MPb,sum_Ms2,y);
        Qval = Q_var(Ahat,Qhat,muhat,Sigmahat,Pihat,Zhat,p,T,Ms,...
            sum_MCP,sum_MP,sum_MPb,sum_Ms2,y);
        if Qval < Qvalold
            Sigmahat = Sigmahatold;
        end
    end
    
    %@@@@@ Update Pi
    if ~skip.Pi
        Pihat = Ms(:,1);
        % For numerical accuracy
        Pihat = Pihat / sum(Pihat);
     end
    
    %@@@@@ Update Z
    if ~skip.Z
        Zhat = sum_Ms2 ./ repmat(sum(sum_Ms2,2),1,M);        
        if ~isempty(fixed.Z)
            Zhat(fixed.Z(:,1)) = fixed.Z(:,2);
        end
    end
    
    % Evaluate and display Q-function if required 
    if verbose
        Qval = Q_var(Ahat,Qhat,muhat,Sigmahat,Pihat,Zhat,p,T,Ms,...
            sum_MCP,sum_MP,sum_MPb,sum_Ms2,y);
        fprintf('Q-function after M-step  = %g\n',Qval);
    end


    
end % END MAIN LOOP



%-------------------------------------------------------------------------%
%                               Output                                    %
%-------------------------------------------------------------------------%

% Return best estimates (i.e. with highest log-likelihood) 
% after reshaping them in compact form

Ahat = reshape(Abest,r,r,p,M);
Qhat = Qbest;
muhat = mubest;
Sigmahat = Sigmabest;
Pihat = Pibest;
Zhat = Zbest;
Mf = Mfbest;
Ms = Msbest;
[~,Sf] = max(Mf);
[~,Ss] = max(Ms);
LL = LL(1:i);

end

