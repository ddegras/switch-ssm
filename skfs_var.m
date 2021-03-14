function [Mf,Ms,L,sum_Ms2] = skfs_var(x,M,p,pars,beta)

%--------------------------------------------------------------------------
%
%               SWITCHING KALMAN FILTER AND SMOOTHER 
%                       IN SWITCHING VAR MODEL 
%
% PURPOSE
% This is not meant to be directly called by the user. It is called by
% functions 'switch_var' to complete the E step of the EM algorithm. 
% Model: conditional on S(t)=j, 
% x(t) ~ N(mu(j),Sigma(j)) if t <= p
% x(t) = sum(l=1:p) A(l,j) x(t-l) + v(t) if t > p
% with x(1),...,x(p) independent given S(1),...,S(p)
% 
% USAGE
% [Mf,Ms,L,sum_Ms2] = skfs_var(x,M,p,pars,beta)
%
% REFERENCES
% K. P. Murphy (1998) "Switching Kalman Filters", Technical Report
%
%--------------------------------------------------------------------------

% Model dimensions
[r,T] = size(x); 

A = reshape(pars.A,r,r,p,M); Q = pars.Q; mu = pars.mu; 
Sigma = pars.Sigma; Pi = pars.Pi; Z = pars.Z;
Zt = Z';

% Remove warnings when inverting singular matrices
warning('off','MATLAB:singularMatrix');
warning('off','MATLAB:illConditionedMatrix');

% Filtered and smoothed regime probabilities
Mf = zeros(M,T);        % P(S(t)=j|x(1:t))
Ms = zeros(M,T);        % P(S(t)=j|x(1:T))
sum_Ms2 = zeros(M,M);   % sum(t=2:T) P(S(t-1)=i,S(t)=j|y(1:T))
if M == 1
    Mf = ones(1,T);
    Ms = ones(1,T);
    sum_Ms2 = T-1;
end

%  P(x(t)|S(t)=j)
Lp = zeros(M,T);

% Log-likelihood log(P(x(t)|x(1:t-1)))
L = zeros(1,T);  

% Cholesky decomposition of Q(j)
cholQ = zeros(r,r,M);
for j = 1:M
    cholQ(:,:,j) = chol(Q(:,:,j),'lower');
end

% Constants involving determinant of Q(j)
logSqrtDetQ = zeros(M,1);
for j = 1:M
    logSqrtDetQ(j) = sum(log(diag(cholQ(:,:,j)))) + (r/2) * log((2*pi));
end


%-------------------------------------------------------------------------%
%                        Switching Kalman Filter                          %
%-------------------------------------------------------------------------%   



% Initialize filter
for t = 1:p   
    for j = 1:M
        Lp(j,t) = mvnpdf(x(:,t)',mu(:,j)',Sigma(:,:,j));
    end   
    if t == 1
        Acc = Pi(:) .* Lp(:,t);   
    else
        Acc = (Zt * Mf(:,t-1)) .* Lp(:,t);
    end   
    if any(isnan(Acc)) 
        Acc(isnan(Acc)) = 0;
    end
    L(t) = log(sum(Acc)); 
    
    if all(Acc == 0)
        Acc = ones(M,1);
    elseif any(isinf(Acc))
        idx = isinf(Acc);
        Acc(idx) = 1;
        Acc(~idx) = 0;
    end
    Mf(:,t) = Acc / sum(Acc);     
end


% Calculate predictive probabilities Lp(j,t) = P(x(t)|S(t)=j,x(1:t-1))
for j = 1:M
    xp = zeros(r,T-p);
    for k = 1:p
        xp = xp + A(:,:,k,j) * x(:,p+1-k:end-k);
    end
    e = cholQ(:,:,j)\(x(:,p+1:end) - xp);
    Lp(j,p+1:T) = exp(-0.5 * sum(e.^2) - logSqrtDetQ(j));
end
clear e xp


% Calculate log-likelihood and filtered probabilities 
for t=p+1:T    
    Acc = Lp(:,t) .* (Zt * Mf(:,t-1));    
    if any(isnan(Acc)) 
        Acc(isnan(Acc)) = 0;
    end
    L(t) = log(sum(Acc));
    
    if all(Acc == 0)
        Acc = ones(M,1); 
    elseif any(isinf(Acc))
        idx = isinf(Acc);
        Acc(idx) = 1;
        Acc(~idx) = 0;
    end
    Mf(:,t) = Acc / sum(Acc);    
end
clear Lp


% Handle infinite values in log-likelihood
test = isinf(L);
if any(test)
    if all(test)
        L = -Inf;
    else
        L(test) = min(L(~test));        
    end
end 

% Add log-likelihoods
L = sum(L);

if M == 1
    return
end


%-------------------------------------------------------------------------%
%                        Switching Kalman Smoother                        %
%-------------------------------------------------------------------------%
    


% Initialize smoother at time T
Ms(:,T) = Mf(:,T);

for t = T-1:-1:1  
    % P(S(t)=j|S(t+1)=k,y(1:T))
    U = Mf(:,t) .* Z; %@@@@ uses implicit expansion
    U = U ./ sum(U);  %@@@@ uses implicit expansion
    U(isnan(U)) = 0;
    
    % P(S(t)=j,S(t+1)=k|y(1:T))
    Ms2 = U .* (Ms(:,t+1)'); %@@@@ uses implicit expansion
    if all(Ms2(:) == 0)
        Ms2 = (1/M^2) * ones(M);
    end
    if beta < 1
        Ms2 = Ms2.^beta; % DAEM
    end
    Ms2 = Ms2 / sum(Ms2(:)); % for numerical accuracy
    sum_Ms2 = sum_Ms2 + Ms2;
    
    % P(S(t)=j|y(1:T))
    Ms(:,t) = sum(Ms2,2);    
end


