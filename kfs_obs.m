function [xf,xs,x0,P0,Loglik,sum_CP,sum_MP,sum_Mxy,sum_P,sum_Pb] = ...
    kfs_obs(y,M,p,r,A,C,Q,R,S,mu,Sigma,safe,abstol,reltol)


% Model dimensions
[N,T] = size(y);

% Dimension of state vector x(t) = r
% Dimension of state vector X(t,j) = (x(t,j),...,x(t-p+1,j)) = p * r
% (regime j, all lags)
% Dimension of state vector X(t) = (X(t,1),...,X(t,M)) = M * p * r 
% (all regimes, all lags)     

% % Reshape parameter estimates
% Atmp = squeeze(mat2cell(A(1:r,:,:),r,p*r,repelem(1,M)));
% Asmall = blkdiag(Atmp{:});
% Atmp = squeeze(mat2cell(A,p*r,p*r,repelem(1,M)));
% A = blkdiag(Atmp{:});
% Ctmp = C;
% C = zeros(N,M*p*r,M);
% for j = 1:M
%     ind = (j-1)*p*r+1:j*p*r;
%     C(:,ind,j) = Ctmp(:,:,j);
% end
% Qtmp = squeeze(mat2cell(Q,p*r,p*r,repelem(1,M)));
% Q = blkdiag(Qtmp{:});
% mu = mu(:);
% Stmp = squeeze(mat2cell(Sigma,p*r,p*r,repelem(1,M)));
% Sigma = blkdiag(Stmp{:});
% 
% clear Atmp Ctmp Qtmp Stmp

% Reshape parameter estimates
Atmp = zeros(M*p*r); 
Asmall = zeros(M*r,M*p*r);
Ctmp = zeros(N,M*p*r,M);
Qtmp = zeros(M*p*r); 
Stmp = zeros(M*p*r);

for j = 1:M
    ind = (j-1)*p*r+1:j*p*r;
    ind2 = (j-1)*r+1:j*r;
    Atmp(ind,ind) = A(:,:,j);       % for filtering 
    Asmall(ind2,ind) = A(1:r,:,j);  % for smoothing
    Ctmp(:,ind,j) = C(:,:,j);       
    Q(ind,ind) = Q(:,:,j);
    Stmp(ind,ind) = Sigma(:,:,j);
end

mu = mu(:);
A = Atmp;
C = Ctmp;
Q = Qtmp;
Sigma = Stmp;

clear Atmp Ctmp Qtmp Stmp


cst = N * log(2*pi);

% Remove warnings when inverting singular matrices
warning('off','MATLAB:singularMatrix');
warning('off','MATLAB:illConditionedMatrix');

% Declaring Kalman filter variables
xp = zeros(M*r,T); % E(x(t)|y(1:t-1))
Vp = zeros(M*r,M*r,T); % V(x(t)|y(1:t-1))
Loglik = zeros(T,1); % P(y(t)|y(1:t-1))
xf = zeros(M*r,T); % E(X(t)|y(1:t))
Vf = zeros(M*(p*r)^2,T); % V(X(t,l)|y(1:t))

% Declaring Kalman smoothing variables
xs = zeros(M*r,T); % E(x(t)|y(1:T))
% Vst = zeros(M*r,M*r); % V(x(t)|y(1:T))
% CVst = zeros(M*r,M*r); % Cov(x(t+1),x(t)|y(1:T))
% P = zeros(r,r,M); % E(x(t)x(t)'|y(1:T))

% Other outputs
% P0 = zeros(r,r,M); % E(x(1,j)x(1,j)'|y(1:T)) 
sum_CP = zeros(M*r,M*p*r); % sum(t=2:T) E(x(t)X(t-1)'|y(1:T))
sum_MP = zeros(M*p*r,M*p*r,M); % sum(t=2:T) P(S(t)=j) E(x(t,j)x(t,j)'|y(1:T))
% sum_P = zeros(r0,r0,M); % sum(t=2:T) E(x(t,l)x(t,l)'|y(1:T))
% sum_Pb = zeros(r,r,M); % sum(t=2:T) E(X(t-1,l)X(t-1,l)'|y(1:T))
% sum_Mxy = zeros(r0,N,M); % sum(t=1:T) P(S(t)=j) E(x(t,j)|y(1:T)) y(t)'


%-------------------------------------------------------------------------%
%                        Switching Kalman Filter                          %
%-------------------------------------------------------------------------%   



% Mask for predictive covariance matrix
tmp = zeros(p); tmp(1) = 1;
mask_Vp = (kron(eye(M),kron(tmp,ones(r))) == 1);

% Mask for filtering covariance matrix
mask_Vf = (kron(eye(M),ones(p*r)) == 1);

% Mask for r x r diagonal blocks in Mr x Mr matrix
mask_Mr = find(kron(eye(M),ones(r)));

% Indices of x(t,1),...,x(t,M) (size Mr) in X(t) (size Mpr)
indt = zeros(p*r,M);
indt(1:r,:) = 1;
indt = find(indt);

% Container for Mr x Mr block-diagonal matrix
Vtmp = zeros(M*r);


% FILTERING LOOP
for t = 1:T

    % Prediction of x(t)
    if t == 1
        xpt = mu;
        Vpt = Sigma;
    else
        xpt = A * xf(:,t-1); 
        Vpt = A * Vf(:,:,t-1) * A.' + Q;
        Vpt = 0.5 * (Vpt + Vpt.');
    end
    xp(:,t) = xpt(indt);
    Vtmp(mask_Mr) = Vpt(mask_Vp); %#ok<*FNDSB>
    Vp(:,:,t) = Vtmp;      
   
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
    xf(:,t) = xpt + K * e; % E(x(t)|y(1:t))
    Vft = Vpt - K * CVp; % V(X(t)|y(1:t))
    Vf(:,t) = Vft(mask_Vf);
    
    % Log-likelihood P(y(t)|y(1:t-1))
    try 
        % Choleski decomposition
        U = chol(Ve); 
        Loglik(t) = -0.5 * (cst + sum(log(diag(U))) + norm(U\e)^2);
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
    
    

% Set off-diagonal elements of V(X(t)|y(1:t),S(1:t)) to zero
Vf( repmat( (kron(eye(M),ones(p*r)) == 0), [1,1,T]) ) = 0; %@@@@@@@@

% Mask of diagonal rxr blocks for array of dimensions Mr x Mr x M
rrmask = (kron(eye(M),ones(p*r)) == 1);

% Initialize smoother at time T
xst = xf(:,T);
xs(:,T) = xst;
Vst = Vf(:,:,T);
% Vst(~rrmask) = 0;
PT = Vst + (xst * xst.');
sum_MP(:,:,S(T)) = PT;


for t=T-1:-1:1
    
    % Store smoothed mean & variance from previous iteration
    xstp1 = xst; % E(x(t+1)|y(1:T))
    Vstp1 = Vst; % V(x(t+1)|y(1:T))
    
    % Predicted and filtered mean and variance (for faster access)
    xptp1 = xp(:,t+1);
    Vptp1 = Vp(:,:,t+1);
    xft = xf(:,t);
    Vft = Vf(:,:,t);
    
    % Smoothed mean and variance of x(t), smoothed cross-covariance of
    % x(t+1) & x(t)   
    % Kalman smoother gain 
    % J(t) = V(x(t)|y(1:t)) * A' * V(x(t+1)|y(1:t))^{-1}
    J = Vft * (Asmall.') / Vptp1;
    if any(isnan(J(:))) || any(isinf(J(:)))
        J = Vft * Asmall.' * pinv(Vptp1);
    end

    % E(x(t)|y(1:T))
    xst = xft + J * (xstp1(indt) - xptp1);
    xs(:,t) = xst(indt);
    
    % V(x(t)|y(1:T))
    Vst = Vft + J * (Vstp1(indt,indt) - Vptp1) * J.';
    Vst(~rrmask) = 0;
    % Cov(x(t+1),x(t)|y(1:T)) = V(x(t+1)|S(t+1)=k,y(1:T)) * J(t)'
    % Equation (20) of "Derivation of Kalman filtering and smoothing equations"
    % by B. M. Yu, K. V. Shenoy, M. Sahani. Technical report, 2004.
    CVst = Vstp1(indt,indt) * J.'; 
    
    % Required quantities for M step  
    P = Vst + (xst * xst.');
    sum_MP(:,:,S(t)) = sum_MP(:,:,S(t)) + P;
    CP = CVst + (xstp1(indt) * xst.'); 
    sum_CP = sum_CP + CP;
    
end % end t loop

P0 = P; % E(X(1)X(1)'|y(1:T))
sum_P = sum(sum_MP,3) - P0; % sum(t=2:T) E(x(t)x(t)'|y(1:T))
sum_Pb = sum(sum_MP,3) - PT; % sum(t=1:T-1) E(x(t)x(t)'|y(1:T))

% Post-process output quantities
% x0 = reshape(xs(:,1),p*r,M); % E(X(1,l)|y(1:T))
x0 = reshape(xst,[p*r,M]);
xf = reshape(xf,[r,M,T]);
xs = reshape(xs,[r,M,T]);

P0 = reshape(P0(rrmask),[p*r,p*r,M]);
r0rmask = (kron(eye(M),ones(r,p*r)) == 1);
sum_CP = reshape(sum_CP(r0rmask),[r,p*r,M]);
tmp = zeros(p*r); tmp(1:r,1:r) = 1; 
r0r0mask = (kron(eye(M),tmp) == 1);
sum_P = reshape(sum_P(r0r0mask),[r,r,M]);
sum_Pb = reshape(sum_Pb(rrmask),[p*r,p*r,M]);
sum_Mxy = zeros(r,N,M);
sum_MPcopy = sum_MP;
sum_MP = zeros(r,r,M);
for j = 1:M
    ind1 = (j-1)*p*r+1:(j-1)*p*r+r;
    sum_MP(:,:,j) = sum_MPcopy(ind1,ind1,j);
    ind2 = (S == j);
    sum_Mxy(:,:,j) = xs(:,ind2) * y(:,ind2).';
end

end % END FUNCTION

