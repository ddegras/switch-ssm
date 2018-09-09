function [Mf,Ms,L,sum_Ms2] = skfs_var(x,M,p,A,Q,mu,Sigma,Pi,Z,beta)

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
% [Mf,Ms,xf,xs,L,MP0,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,sum_P] = ... 
%     skfs_var(y,M,p,r,A,Q,mu,Sigma,Pi,Z,beta,safe,abstol,reltol)
%
% REFERENCES
% K. P. Murphy (1998) "Switching Kalman Filters", Technical Report
%
%--------------------------------------------------------------------------

% Model dimensions
[r,T] = size(x); 
% Size of 'small' state vector x(t): r
% Size of 'big' state vector X(t) = (x(t),...,x(t-p+1)): p * r

% Shrink input array A if needed
if size(A,1) == p*r
    A = A(1:r,:,:); 
end

% Remove warnings when inverting singular matrices
warning('off','MATLAB:singularMatrix');
warning('off','MATLAB:illConditionedMatrix');

% Filtered and smoothed regime probabilities
Mf = zeros(M,T);        % P(S(t)=j|x(1:t))
Ms = zeros(M,T);        % P(S(t)=j|x(1:T))
sum_Ms2 = zeros(M,M);   % sum(t=2:T) P(S(t-1)=i,S(t)=j|y(1:T))





%-------------------------------------------------------------------------%
%                        Switching Kalman Filter                          %
%-------------------------------------------------------------------------%   


% Initialize filter
Acc = zeros(M,1);
Px = zeros(M,1);

% P(x(1),S(1)=j)
for j=1:M    
    Acc(j) = Pi(j) * mvnpdf(x(:,1)',mu(:,j)',Sigma(:,:,j));   
end
if all(Acc == 0)
    Acc = eps * ones(M,1);
end

Mf(:,1) = Acc / sum(Acc);   % P(S(1)=j|y(1))
L = log(sum(Acc));          % log(P(y(1)))

if p > 1
    for t = 2:p
        % P(x(t)|S(t)=j)
        for j = 1:M
            Px(j) = mvnpdf(x(:,t)',mu(:,j)',Sigma(:,:,j));
        end
        
        % P(x(t),S(t-1)=i,S(t)=j|x(1:t-1))
        % = P(x(t)|S(t)=j) * P(S(t)=j|S(t-1)=i) * P(S(t-1)=i|x(1:t-1))
        Acc = diag(Mf(:,t-1)) * Z * diag(Px);
        if all(Acc(:) == 0)
            Acc = eps * ones(M,M);
        end
        
        % Log-likelihood
        % P(x(t)|x(1:t-1)) = sum(i,j) P(x(t),S(t-1)=i,S(t)=j|x(1:t-1))
        L = L + log(sum(Acc(:))); 
        
        % Filtered occupancy probability of state j at time t
        Mf2 = Acc / sum(Acc(:)); % P(S(t-1)=i,S(t)=j|x(1:t))
        Mf(:,t) = sum(Mf2).';    % P(S(t)=j|x(1:t))      
    end
end


% MAIN LOOP

Lp = zeros(M,1);

for t=p+1:T    
    Xtm1 = reshape(x(:,t-1:-1:t-p),p*r,1);
    % P(x(t)|x(1:t-1),S(t)=j)
    for j=1:M                 
        e = x(:,t) - A(:,:,j) * Xtm1;
        Lp(j) = mvnpdf(e',[],Q(:,:,j));        
    end
    % P(x(t),S(t)=j|x(1:t-1))
    % = P(x(t)|S(t)=j,x(1:t-1)) * sum(i=1:M) {P(S(t)=j|S(t-1)=i) * ...
    % P(S(t-1)=i|x(1:t-1))}
    Acc = Lp .* (Z' * Mf(:,t-1));    
    if all(Acc == 0)
        Acc = eps * ones(M,1);
    end
    
    % P(x(t)|x(1:t-1))
    L = L + log(sum(Acc));
    
    % P(S(t)=j|x(1:t))
    Mf(:,t) = Acc/sum(Acc);    
end

 

%-------------------------------------------------------------------------%
%                        Switching Kalman Smoother                        %
%-------------------------------------------------------------------------%
    


% Initialize smoother at time T
Ms(:,T) = Mf(:,T);

for t = T-1:-1:1
    
    % P(S(t)=j|S(t+1)=k,y(1:T))
    U = diag(Mf(:,t)) * Z; 
    U = U ./ repmat(sum(U),M,1); % scaling
    U(isnan(U)) = 0;
    
    % P(S(t)=j,S(t+1)=k|y(1:T))
    Ms2 = U * diag(Ms(:,t+1)); 
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


