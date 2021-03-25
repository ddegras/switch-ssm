function [outpars,LL] = fast_var(y,M,p,S,control,equal,fixed,scale)

%--------------------------------------------------------------------------
%
%       PARAMETER ESTIMATION AND INFERENCE IN SWITCHING VECTOR  
%           AUTOREGRESSIVE (VAR) MODEL ASSUMING REGIMES KNOWN 
%
% PURPOSE
% This function estimates model parameters and infers hidden state vectors 
% by the EM algorithm in switching vector autoregressive models under
% the assumption that the regimes (i.e. switching variables) are known.
% The function can be used to fit the model under a trajectory of regimes 
% that is relevant or highly likely. In the case of only one regime (no
% switching), it can also be used to fit a standard (Gaussian) linear SSM
%
% USAGE
%   [outpars,LL] = fast_var(y,M,p,S,control,equal,fixed,scale)
% 
% INPUTS  
% y:    time series data (dimension NxT)
% M:    number of regimes
% p:    order of VAR model for observation vector y(t)
% S:    fixed sequence of regimes S(t), t=1:T
% control: optional struct variable with fields: 
%       'abstol': absolute tolerance for eigenvalues. Default = 1e-8
%       'reltol': relative tolerance for eigenvalues. Default = 1e-8
% equal:  optional struct variable with fields:
%       'A': if true, VAR transition matrices A(l,j) are equal across regimes j=1,...,M
%       'Q': if true, VAR innovation matrices Q(j) are equal across regimes
%       'mu': if true, initial mean state vectors mu(j) are equal across regimes
%       'Sigma': if true, initial variance matrices Sigma(j) are equal across regimes
% fixed:  optional struct variable with fields 'A','Q','mu','Sigma'.
%   If not empty, each field must contain an array of the same dimensions as 
% the corresponding parameter. Array entries with numerical values indicate 
% the fixed coefficients, entries with NaN indicate free coefficients
% scale:  optional struct variable with field:
%       'A': upper bound for norm of eigenvalues of A matrices. Must be in (0,1).
%
% OUTPUTS
% Ahat:  Estimated system matrix
% Qhat:  Estimated state noise cov
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
% Date : March 13, 2021
% Reference: K. Murphy, "Switching Kalman Filters," 1998.
%--------------------------------------------------------------------------





%-------------------------------------------------------------------------%
%                           Initialization                                %
%-------------------------------------------------------------------------%


narginchk(4,8)

% Check that time series and regime history have same length
assert(size(y,2) == numel(S))

% Data dimensions
[N,T] = size(y);
% 'small' state vector: x(t), size r
% 'big' state vector: X(t)=(x(t),...,x(t-p+1)), size p*r
% We assume that the initial vectors x(1),...,x(p) are iid ~ N(mu,Sigma)
% and  mutually independent with S(1),...,S(p).

% Center the data
y = y - mean(y,2);

% Initialize optional arguments if not specified
if ~exist('fixed','var')
    fixed = []; 
end
if ~exist('equal','var')
    equal = []; 
end
if ~exist('control','var')
    control = []; 
end
if ~exist('scale','var')
    scale = []; 
end


% Pilot estimate
Pi = zeros(M,1);
Pi(S(1)) = 1;
pars = struct('A',zeros(N,N,p,M), 'C',eye(N), 'Q',zeros(N,N,M), ...
    'R',1e-10*eye(N), 'mu',zeros(N,M), 'Sigma',repmat(eye(N),1,1,M), ...
    'Pi',Pi, 'Z',eye(M));
if ~isempty(fixed) && isstruct(fixed)
    parname = {'A','Q','mu','Sigma'};
    for i = 1:4
        name = parname{i};
        if isfield(fixed,name) && ~isempty(fixed.(name))
            idx = ~isnan(fixed.(name));
            pars.(name)(idx) = fixed.(name)(idx);
        end
    end
    if isfield(fixed,'C')
        fixed = rmfield(fixed,'C');
    end
    if isfield(fixed,'R')
        fixed = rmfield(fixed,'R');
    end
end

[pars,control,equal,fixed,scale,skip] = ... 
    preproc_dyn(M,N,p,N,pars,control,equal,fixed,scale);

       

%-------------------------------------------------------------------------%
%                    Maximum Likelihood Estimation                        %
%-------------------------------------------------------------------------%



% Calculate required quantities for log-likelihood function
Ms = zeros(M,T);
sum_MP = zeros(N,N,M);
sum_MPb = zeros(p*N,p*N,M);
sum_MCP = zeros(N,p*N,M);
sum_Ms2 = eye(M);
for j = 1:M
    idxj = find(S == j);
    Ms(j,idxj) = 1;
    idxj = idxj(idxj > p);
    sum_MP(:,:,j) = y(:,idxj) * y(:,idxj).';
    for l1 = 1:p
        idx1 = (l1-1)*N+1:l1*N;
        sum_MCP(:,idx1,j) = y(:,idxj) * y(:,idxj-l1).';
        for l2 = 1:l1
            idx2 = (l2-1)*N+1:l2*N;            
            sum_MPb(idx1,idx2,j) = y(:,idxj-l1) * y(:,idxj-l2).';
            sum_MPb(idx2,idx1,j) = sum_MPb(idx1,idx2,j).';
        end
    end    
end

% Maximize (log)likelihood
outpars = M_var(pars,Ms,sum_MCP,sum_MP,sum_MPb,sum_Ms2,y,...
    control,equal,fixed,scale,skip);

% Calculate log-likelihood
LL = - 0.5 * N * T * log(2*pi);
for j = 1:M
    idx = (S(1:p) == j);
    if any(idx)
        nj = sum(idx);
        cholS = chol(outpars.Sigma(:,:,j),'lower');
        e = cholS\(y(:,idx) - outpars.mu(:,j));
        LL = LL - nj * sum(log(diag(cholS))) - 0.5 * sum(e(:).^2);
    end
    A_j = outpars.A(:,:,j);
    Q_j = outpars.Q(:,:,j);
    cholQ = chol(Q_j,'lower');
    sum_Mj = sum(Ms(j,p+1:end));
    sum_MPj = sum_MP(:,:,j);
    sum_MPbj = sum_MPb(:,:,j);
    sum_MCPj = sum_MCP(:,:,j);  
    LL = LL - sum_Mj * sum(log(diag(cholQ))) - ...
        0.5 * trace(Q_j\(sum_MPj - (A_j * sum_MCPj.') - (sum_MCPj * A_j.') + ...
            (A_j * sum_MPbj * A_j.')));
end



% Postprocess MLE 
outpars.A = reshape(outpars.A,N,N,p,M);
outpars = rmfield(outpars,{'C','R','Pi','Z'});







%@@@@@ Estimate A @@@@@%
 

% If (i) A is fixed but not entirely OR (ii) A(1) = ... = A(M) and the Q(j)
% are not constrained to be equal, then the estimation of A and Q is
% coupled

% if skip.A
%     Ahat = reshape(fixed.A(:,2),N,N,p,M);
% else
%     % Pilot estimate of A 
%     if equal.A
%         Ahat = sum(sum_CP,3) / sum(sum_Pb,3);
%         if any(isinf(Ahat(:)) | isnan(Ahat(:)))
%             Ahat = sum(sum_CP,3) * pinv(sum(sum_Pb,3));
%         end
%         Ahat = repmat(Ahat,1,1,M); 
%     else 
%         Ahat = zeros(N,p*N,M);
%         for j = 1:M
%             A_j = sum_CP(:,:,j) / sum_Pb(:,:,j);
%             if any(isinf(A_j(:)) | isnan(A_j(:)))
%                 A_j = sum_CP(:,:,j) * pinv(sum_Pb(:,:,j));
%             end
%             Ahat(:,:,j) = A_j;
%         end
%     end
%     
%     % Build pilot estimate of Q as needed
%     if (equal.A && ~equal.Q) || ~isempty(fixed.A)
%         Qhat = zeros(r,r,M);
%         for j = 1:M
%             A_j = Ahat(:,:,j);
%             idx = find(S == j);
%             idx = idx(idx > p);
%             Qhat(:,:,j) = y(:,idx) * y(:,idx).' - ...
%                 (A_j * sum_CP(:,:,j)') - (sum_CP(:,:,j) * A_j') ...
%                 + A_j * sum_Pb(:,:,j) * A_j';
%             Qhat(:,:,j) = Qhat(:,:,j) / numel(idx);
%             Qhat(:,:,j) = 0.5 * (Qhat(:,:,j) + Qhat(:,:,j)');
%         end
%     end
% 
%     % Re-estimate A if equality constraints on A but not on Q
%     if  equal.A && ~equal.Q
%         lhs = zeros(p*r^2,p*r^2);
%         rhs = zeros(r,p*r);
%         for j=1:M
%             Qinv_j = myinv(Qhat(:,:,j));         
%             lhs = lhs + kron(sum_Pb(:,:,j),Qinv_j);
%             rhs = rhs + Qinv_j * sum_CP(:,:,j);
%         end
%         rhs = rhs(:);
%         Ahat = lhs\rhs;
%         if any(isnan(Ahat(:))|isinf(Ahat(:)))
%             Ahat = pinv(lhs)*rhs;
%         end
%         Ahat = reshape(Ahat,r,p*r);
%         Ahat = repmat(Ahat,[1,1,M]);
%     end
% 
%     % Case: fixed coefficient constraints on A --> Vectorize matrices and
%     % solve associated problem after discarding rows associated with fixed
%     % coefficients. Recall: there cannot be both fixed coefficient
%     % constraints *and* equality constraints on A
%     if ~isempty(fixed.A)
%         fixed_A = NaN(size(Ahat));
%         fixed_A(fixed.A(:,1)) = fixed.A(:,2);
%         for j = 1:M
%             % Linear indices of free coefficients in A(j)
%             free = (isnan(fixed_A(:,:,j)));
%             % Quadratic part
%             Qinv_j = myinv(Qhat(:,:,j));
%             lhs = kron(sum_Pb(:,:,j),Qinv_j);
%             lhs = lhs(free,free);
%             % Linear part
%             rhs = Qinv_j * sum_CP(:,:,j);            
%             rhs = rhs(free);
%             % vector solution 
%             a_j = lhs\rhs; 
%             if any(isnan(a_j)|isinf(a_j))
%                 a_j = pinv(lhs)*rhs;
%             end     
%             % Fill solutions (free coefficients) in A 
%             A_j = zeros(N,p*N);
%             A_j(free) = a_j;
%             Ahat(:,:,j) = A_j;
%         end
%         % Fill in fixed coefficients
%         Ahat(fixed.A(:,1)) = fixed.A(:,2);
%     end
%     
%     % Check eigenvalues of estimate and regularize if needed 
%     Abig = diag(ones((p-1)*r,1),-r);
%     for j = 1:M
%         % Check eigenvalues
%         Abig(1:r,:) = Ahat(:,:,j);
%         eigval = eig(Abig);
%         if any(abs(eigval) > scale.A)
%             if verbose
%                 warning(['Eigenvalues of A%d greater than %f.',...
%                     ' Regularizing.'],j,scale.A)
%             end
%             c = .999 * scale.A / max(abs(eigval));
%             A_j = reshape(Ahat(:,:,j),[r,r,p]);
%             for l = 1:p
%                 A_j(:,:,l) = c^l * A_j(:,:,l);
%             end 
%             Ahat(:,:,j) = reshape(A_j,[r,p*r]);
% 
%         end
%         if equal.A
%             Ahat = repmat(Ahat(:,:,1),[1,1,M]);
%             break
%         end 
%     end               
% end
% 
%    
% %@@@@@ Update Q @@@@@%
% 
% % Unconstrained solution
% if skip.Q
%     Qhat = fixed.Q; 
% else
%     Qhat = zeros(r,r,M);
%     for j=1:M
%         if sum_M(j) > 0
%             A_j = Ahat(:,:,j);
%             sum_Pj = sum_P(:,:,j);
%             sum_CPj = sum_CP(:,:,j);
%             sum_Pbj = sum_Pb(:,:,j);                
%             Q_j = (sum_Pj - (sum_CPj * A_j.') - ...
%                 (A_j * sum_CPj.') + A_j * sum_Pbj * A_j.') / sum_M(j);
%             Qhat(:,:,j) = 0.5 * (Q_j + Q_j.');
%         end
%      end
%     if equal.Q
%         w = sum_M/(T-p);
%         Qhat = reshape(Qhat,r*r,M) * w;
%         Qhat = reshape(Qhat,r,r);
%         Qhat = 0.5 * (Qhat + Qhat.');
%         Qhat = repmat(Qhat,1,1,M);
%     end
%     
%     % Enforce fixed coefficient constraints
%     if ~isempty(fixed.Q)
%         Qhat(fixed.Q(:,1)) = fixed.Q(:,2);
%     end
% 
%     % Regularize estimate if needed
%     for j = 1:M
%         eigval = eig(Qhat(:,:,j));
%         if min(eigval) < max(abstol,max(eigval)*reltol)
%             if verbose 
%                 warning(['Q%d ill-conditioned and/or nearly singular.', ... 
%                     ' Regularizing.'],j);
%             end
%             Qhat(:,:,j) = regfun(Qhat(:,:,j),abstol,reltol);
%         end
%         
%         if equal.Q
%             Qhat = repmat(Qhat(:,:,1),[1,1,M]);
%             break
%         end
%     end
% 
%     % Re-apply fixed coefficient constraints
%     if ~isempty(fixed.Q)
%         Qhat(fixed.Q(:,1)) = fixed.Q(:,2);
%     end
% end
% 
% 
% %@@@@@ Update mu @@@@@%
% if skip.mu
%     muhat = fixed.mu;
% else
%     muhat = repmat(mean(y(:,1:p),2),[1,M]);
%     if ~isempty(fixed.mu)
%         muhat(fixed.mu(:,1)) = fixed.mu(:,2);
%     end
% end
% 
% 
% %@@@@@ Update Sigma @@@@@%
% if skip.Sigma
%     Sigmahat = fixed.Sigma;
% else
%     Sigmahat = diag(mean((y(:,1:p)-muhat(:,S(1))).^2,2));
%     Sigmahat = repmat(Sigmahat,1,1,M);
% 
%     % Apply fixed coefficient constraints if any
%     if ~isempty(fixed.Sigma)
%         Sigmahat(fixed.Q(:,1)) = fixed.Sigma(:,2);
%     end
% 
%     % Regularize estimate if needed
%     eigval = eig(Sigmahat(:,:,1));
%     if min(eigval) < max(abstol,max(eigval)*reltol)
%         if verbose
%             warning(['Sigma ill-conditioned and/or nearly singular.', ...
%                 ' Regularizing.']);
%         end
%         Sigmahat = regfun(Sigmahat(:,:,1),abstol,reltol); 
%         Sigmahat = repmat(Sigmahat,[1,1,M]);
% 
%         % Enforce fixed coefficient constraints 
%         if ~isempty(fixed.Sigma)
%             Sigmahat(fixed.Q(:,1)) = fixed.Sigma(:,2);
%         end
%     end 
% end






