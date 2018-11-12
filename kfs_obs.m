function [xf,xs,x0,P0,Loglik,sum_CP,sum_MP,sum_Mxy,sum_P,sum_Pb] = ...
    kfs_obs(y,M,p,r,A,C,Q,R,S,mu,Sigma,safe,abstol,reltol)


% Model dimensions
[N,T] = size(y);

% Notations for state vector
% x(t,j): length r (time t, regime j)
% x(t) = (x(t,1),...,x(t,M)): length M*r (time t, all regimes) 
% X(t,j) = (x(t,j),...,x(t-p+1,j)): length p*r (all lags, regime j)
% X(t) = (X(t,1),...,X(t,M)): length M*p*r (all lags, all regimes)
    
% Mask for predictive covariance matrix
tmp = zeros(p*r); tmp(1:r,1:r) = 1;
mask_Vp = (kron(eye(M),tmp) == 1);

% Mask for filtering covariance matrix
mask_Vf = (kron(eye(M),ones(p*r)) == 1);

% Reshape parameter estimates
Atmp = kron(eye(M),diag(ones((p-1)*r,1),-r));
tmp = zeros(p*r); tmp(1:r,:) = 1;
mask_A = (kron(eye(M),tmp) == 1);
Atmp(mask_A) = A(:);
Asmall = zeros(M*r,M*p*r); % used for smoothing, size Mr x Mpr
mask_A = (kron(eye(M),ones(r,p*r)) == 1);
Asmall(mask_A) = A(:); 
Ctmp = zeros(N,M*p*r,M);
for j = 1:M
    idx = (j-1)*p*r+1:(j-1)*p*r+r;
    Ctmp(:,idx,j) = C(:,:,j);       
end
Qtmp = zeros(M*p*r);
Qtmp(mask_Vp) = Q(:);
mutmp = repmat(mu,p,1);
Sigmatmp = zeros(M*p*r);
tmp = repmat(reshape(Sigma,r*r,M),p,1);
mask_Sigma = (kron(eye(M*p),ones(r)) == 1);
Sigmatmp(mask_Sigma) = tmp;
A = Atmp;       % size Mpr x Mpr (used for filtering)
C = Ctmp;       % size N x Mpr x M
Q = Qtmp;       % size Mpr x Mpr
mu = mutmp(:);  % size Mpr x 1
Sigma = Sigmatmp; % size Mpr x Mpr

% Remove warnings when inverting singular matrices
warning('off','MATLAB:singularMatrix');
warning('off','MATLAB:nearlySingularMatrix');
warning('off','MATLAB:illConditionedMatrix');

% Declare Kalman filter variables
xp = zeros(M*r,T);      % E(x(t)|y(1:t-1))
Vp = zeros(M*r^2,T);    % V(x(t,j)|y(1:t-1)), j=1:M
Loglik = zeros(T,1);    % P(y(t)|y(1:t-1))
xf = zeros(M*p*r,T);    % E(X(t)|y(1:t))
Vf = zeros(M*(p*r)^2,T); % V(X(t,j)|y(1:t)), j=1:M

% Declare Kalman smoothing variables
xs = zeros(M*r,T);      % E(x(t)|y(1:T))




%-------------------------------------------------------------------------%
%                        Switching Kalman Filter                          %
%-------------------------------------------------------------------------%   



% Indices of x(t) (size Mr) in X(t) (size Mpr)
indt = zeros(p*r,M);
indt(1:r,:) = 1;
indt = find(indt);

% Constant for likelihood calculation
cst = -N * log(2*pi);


% FILTERING LOOP
for t = 1:T

    % Prediction of x(t)
    if t == 1
        xpt = mu;
        Vpt = Sigma;
    else
        xpt = A * xf(:,t-1); 
        Vpt = A * Vft * A.' + Q; % here Vft = V(X(t-1)|y(1:t-1))
        Vpt = 0.5 * (Vpt + Vpt.');
    end
    xp(:,t) = xpt(indt);
    Vp(:,t) = Vpt(mask_Vp);

    % Prediction error for y(t)
    Ct = C(:,:,S(t));
    e = y(:,t) - Ct * xpt;
    CVp = Ct * Vpt; % Ct * V(x(t)|y(1:t-1)) 
    Ve = CVp * Ct.' + R; % Variance of prediction error         
    Ve = 0.5 * (Ve+Ve.');

    % Check that variance matrix is positive definite and well-conditioned
    if safe
        Ve = regfun(Ve,abstol,reltol);
    end
           
    % Filtering update 
    K = (CVp.') / Ve; % Kalman gain matrix
    xf(:,t) = xpt + K * e; % E(X(t)|y(1:t))
    Vft = Vpt - K * CVp; % V(X(t)|y(1:t))
    Vf(:,t) = Vft(mask_Vf);
    
    % Log-likelihood P(y(t)|y(1:t-1))
    try 
        % Choleski decomposition
        L = chol(Ve,'lower'); 
        Loglik(t) = cst - sum(log(diag(L))) - 0.5 * norm(L\e)^2;
%        Loglik(t) = mvnpdf(e',[],Ve); slower
    catch
        Loglik(t) = -Inf;
    end
end 

% Replace infinite or undefined values of log-likelihood by very small
% value
is_singular_Ve = isnan(Loglik) | isinf(Loglik);
if any(is_singular_Ve)
    Loglik(is_singular_Ve) = min(Loglik(~is_singular_Ve)) + log(eps);
end
    
Loglik = sum(Loglik);




%-------------------------------------------------------------------------%
%                        Switching Kalman Smoother                        %
%-------------------------------------------------------------------------%
    
    

% Mask for r x r diagonal blocks in Mr x Mr matrix
mask_Mr = (kron(eye(M),ones(r)) == 1);

% Initialize smoother at time T
xst = xf(:,T);
xs(:,T) = xst(indt); 
Vst = zeros(M*p*r);     % diag(V(X(T,j)|y(1:T))), j=1:M
Vst(mask_Vf) = Vf(:,T);
Vptp1 = zeros(M*r);
PT = Vst + (xst * xst.');
sum_CP = zeros(M*r,M*p*r);      % sum(t=2:T) E(x(t)X(t-1)'|y(1:T))
sum_MP = zeros(M*p*r,M*p*r,M);  % sum(t:S(t)=j) E(X(t)X(t)'|y(1:T)), j=1:M
sum_MP(:,:,S(T)) = PT;


% To calculate smoothed means, covariances, and cross-covariances for all
% components X(t,j), j=1:M, simultaneously, relevant predicted and filtered
% covariance  matrices are expressed in *block-diagonal* form

for t=T-1:-1:1
    
    % Store smoothed mean & variance from previous iteration
    xstp1 = xst; % E(X(t+1)|y(1:T))
    Vstp1 = Vst; % diag(V(X(t+1,j)|y(1:T))), j=1:M
    
    % Predicted and filtered mean and variance (for faster access)
    xptp1 = xp(:,t+1);
    Vptp1(mask_Mr) = Vp(:,t+1); % diag(V(x(t+1,j)|y(1:t))), j=1:M
    xft = xf(:,t);
    Vft(mask_Vf) = Vf(:,t);     % diag(V(X(t+1,j)|y(1:t))), j=1:M
    
    % Kalman smoother gain J(t) 
    % J(t) = V(X(t)|y(1:t)) * A' * V(x(t+1)|y(1:t))^{-1}
    J = Vft * (Asmall.') / Vptp1;
    if any(isnan(J(:))) || any(isinf(J(:)))
        J = Vft * Asmall.' * pinv(Vptp1);
    end

    % Smoothed mean vector E(X(t)|y(1:T))
    xst = xft + J * (xstp1(indt) - xptp1);
    xs(:,t) = xst(indt);
    
    % Smoothed covariance matrix diag(V(X(t,j)|y(1:T))), j=1:M
    Vst = Vft + J * (Vstp1(indt,indt) - Vptp1) * J.';
    
    % Smoothed cross-covariance matrix diag(Cov(x(t+1,j),X(t,j)|y(1:T))), j=1:M 
    % Equation (20) of "Derivation of Kalman filtering and smoothing equations"
    % by B. M. Yu, K. V. Shenoy, M. Sahani. Technical report, 2004.
    % Cov(x(t+1),X(t)|y(1:T)) = V(x(t+1)|y(1:T)) * J(t)'
    CVst = Vstp1(indt,indt) * J.'; 
    
    % Required quantities for M step  
    P = Vst + (xst * xst.');
    sum_MP(:,:,S(t)) = sum_MP(:,:,S(t)) + P;
    CP = CVst + (xstp1(indt) * xst.'); 
    sum_CP = sum_CP + CP;
    
end % end t loop



%-------------------------------------------------------------------------%
%                            Process outputs                              %
%-------------------------------------------------------------------------%


% E(x(t,j)|y(1:t)), t=1:T,j=1:M
xf = reshape(xf,[p*r,M,T]);
xf = xf(1:r,:,:);

% E(x(t,j)|y(1:T)), t=1:T,j=1:M
xs = reshape(xs,[r,M,T]);

% E(X(1,j)|y(1:T)), j=1:M
x0 = reshape(xst,[p*r,M]); 

% E(X(1,j)X(1,j)'|y(1:T)), j=1:M
P0 = reshape(P(mask_Vf),[p*r,p*r,M]);

% sum(t=2:T) E(x(t,j)X(t-1,j)'|y(1:T)), j=1:M
mask_CP = (kron(eye(M),ones(r,p*r)) == 1);
sum_CP = reshape(sum_CP(mask_CP),[r,p*r,M]); 

% sum(t:S(t=j)) E(x(t,j)x(t,j)'|y(1:T)), j=1:M
sum_MPcopy = sum_MP;
sum_MP = zeros(r,r,M);
for j = 1:M
    idx = (j-1)*p*r+1:(j-1)*p*r+r;
    sum_MP(:,:,j) = sum_MPcopy(idx,idx,j);
end

% sum(t:S(t)=j) E(x(t,j)|y(1:T))y(t)', j=1:M
sum_Mxy = zeros(r,N,M);
for j = 1:M
    idx = (S == j);
    sum_Mxy(:,:,j) = squeeze(xs(:,j,idx)) * y(:,idx).';
end

% sum(t=2:T) E(x(t,j)x(t,j)'|y(1:T)), j=1:M
sum_P = sum(sum_MPcopy,3) - P; 
sum_P = reshape(sum_P(mask_Vp),r,r,M);

% sum(t=1:T-1) E(X(t,j)X(t,j)'|y(1:T)), j=1:M
sum_Pb = sum(sum_MPcopy,3) - PT; 
sum_Pb = reshape(sum_Pb(mask_Vf),[p*r,p*r,M]);


end % END FUNCTION
