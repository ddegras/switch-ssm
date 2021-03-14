function [xf,xs,L,MP0,Mx0,sum_MCP,sum_MP,sum_MPb,sum_P] = ... 
    kfs_dyn(y,M,p,r,pars,S,safe,abstol,reltol)


A = pars.A; C = pars.C; Q = pars.Q; R = pars.R; mu = pars.mu; 
Sigma = pars.Sigma; 

% Model dimensions
[N,T] = size(y); 
% Size of 'small' state vector x(t): r
% Size of 'big' state vector X(t) = (x(t),...,x(t-p+1)): p * r

% To remove warnings when inverting singular matrices
warning('off','MATLAB:singularMatrix');
warning('off','MATLAB:illConditionedMatrix');

% Declaring Kalman filter variables
xp = zeros(r,T);	% E(x(t)|y(1:t-1),S(t-1)=i,S(t)=j)
Vp = zeros(r,r,T); % V(x(t)|y(1:t-1),S(t-1)=i)
xf = zeros(p*r,T);      % E(x(t)|y(1:t))
Vf = zeros(p*r,p*r,T); % V(x(t)|y(1:t),S(t-1)=i,S(t)=j)

% Declaring Kalman smoothing variables
xs = zeros(r,T);      % E(x(t)|y(1:T))
% Vst = zeros(p*r,p*r); % V(X(t)|y(1:T))
% CVst = zeros(r,p*r);  % Cov(x(t+1),X(t)|y(1:T)) 

% Other outputs
sum_MCP = zeros(r,p*r,M); % sum(t=2:T,S(t)=j) E(x(t)X(t-1)'|y(1:T))
sum_MP = zeros(r,r,M);  % sum(t=2:T,S(t)=j) E(x(t)x(t)'|y(1:T))
sum_MPb = zeros(p*r,p*r,M); % sum(t=2:T,S(t)=j) E(X(t-1)X(t-1)'|y(1:T))
% sum_P = zeros(r,r)    % sum(t=1:T) E(x(t)x(t)'|y(1:T))
MP0 = zeros(p*r,p*r,M); % P(S(1)=j|y(1:T)) * E(X(1)X(1)'|y(1:T))
Mx0 = zeros(p*r,M);     % P(S(1)=j|y(1:T)) * E(X(1)|y(1:T))

% Log-likelihood
L = zeros(1,T);
% Constant for likelihood calculation
cst = - N/2 * log(2*pi);

% Expand matrices
Abig = repmat(diag(ones((p-1)*r,1),-r),[1,1,M]);
Abig(1:r,:,:) = A;
Cbig = zeros(N,p*r);
Cbig(:,1:r) = C;
Qbig = zeros(p*r,p*r,M);
Qbig(1:r,1:r,:) = Q;


%-------------------------------------------------------------------------%
%                        Switching Kalman Filter                          %
%-------------------------------------------------------------------------%   



% MAIN LOOP
for t=1:T
    
    St = S(t);
    
    % Prediction of x(t)
    if t == 1         
        xpt = repmat(mu(:,St),p,1);
        Vpt = kron(eye(p),Sigma(:,:,St));
    else
        xpt = Abig(:,:,St) * xf(:,t-1); 
        Vpt = Abig(:,:,St) * Vf(:,:,t-1) * Abig(:,:,St).' + Qbig(:,:,St);
    end
    % Store predictions
    xp(:,t) = xpt(1:r);
    Vp(:,:,t) = Vpt(1:r,1:r);  
  
    % Prediction error for y(t)
    e = y(:,t) - C * xpt(1:r);
    Ve = C * Vpt(1:r,1:r) * C.' + R; % Variance of prediction error
%     Ve = 0.5 * (Ve+Ve.');
    % Check that variance matrix is positive definite and well-conditioned
    if safe
        Ve = regfun(Ve,abstol,reltol);
    end

%     % Filtering update 
%     CVp = C * Vpt;
%     K = (CVp.') / Ve; % Kalman gain matrix
%     xf(:,t) = xpt + K * e;       % E(X(t)|S(t-1)=i,S(t)=j,y(1:t))
%     Vf(:,:,t) = Vpt - K * CVp;   % V(X(t)|S(t-1)=i,S(t)=j,y(1:t))
%     if t == T
%       % Cov(x(t),x(t-1)|S(t-1)=i,S(t)=j,y(1:t))
%         CVf = (I - K*C) * A(:,:,j) * Vf(:,:,t-1); 
%     end
 

    [Lchol,err] = chol(Ve,'lower');             
    if ~err % case: Ve definite positive
        LinvCVp = (Lchol\Cbig) * Vpt;
        Linve = Lchol\e;
        % Log-Likelihood 
        L(t) = cst - sum(log(diag(Lchol))) - 0.5 * sum(Linve.^2);
        % Filtering update
        xf(:,t) = xpt + LinvCVp.' * Linve;         % E(X(t)|S(t-1)=i,S(t)=j,y(1:t))
        Vf(:,:,t) = Vpt - (LinvCVp.' * LinvCVp);   % V(X(t)|S(t-1)=i,S(t)=j,y(1:t))
    else
        L(t) = -Inf;
        xf(:,t) = xpt;
        Vf(:,:,t) = Vpt;
    end

end % end t loop  

idx = isinf(L);
if any(idx)
    L(idx) = min(L(~idx)) + log(eps);
end
L = sum(L);
    





%-------------------------------------------------------------------------%
%                        Switching Kalman Smoother                        %
%-------------------------------------------------------------------------%
    


% Initialize smoother at time T
xs(:,T) = xf(1:r,T);
Vst = Vf(:,:,T);
St = S(T);
sum_MP(:,:,St) = (Vst(1:r,1:r) + (xs(:,T) * xs(:,T).'));

for t = T-1:-1:1
    
    % Store relevant vectors/matrices from previous iteration
    Vstp1 = Vst(1:r,1:r); % V(x(t+1)|S(t+1),y(1:T))

    % Shorthand
    St = S(t);
    Stp1 = S(t+1);
    
    % Smoothed mean and variance of x(t), smoothed cross-covariance of
    % x(t+1) & X(t) given S(t)=j and S(t+1)=k 
    % Kalman smoother gain 
    % J(t) = V(X(t)|S(t)=j,y(1:t)) * A_k' * V(x(t+1)|S(t)=j,y(1:t))^{-1}
    J = Vf(:,:,t) * A(:,:,Stp1).' / Vp(:,:,t+1);
    if any(isnan(J(:))) || any(isinf(J(:)))
        J = Vf(:,:,t) * A(:,:,Stp1).' * pinv(Vp(:,:,t+1));
    end        
    % E(X(t)|y(1:T))
    xst = xf(:,t) + J * (xs(:,t+1) - xp(:,t+1)); 
    xs(:,t) = xst(1:r);
    % V(X(t)|y(1:T))
    Vst = Vf(:,:,t) + J * (Vstp1 - Vp(:,:,t+1)) * J.';  
    % Cov(x(t+1),X(t)|y(1:T)) = V(x(t+1)|y(1:T)) * J(t)'
    % Equation (20) of "Derivation of Kalman filtering and smoothing equations"
    % by B. M. Yu, K. V. Shenoy, M. Sahani. Technical report, 2004.
    CVst = Vstp1 * J.';  
       
    % Required quantities for M step 
    % E(X(t)X(t)'|y(1:T))
    P = Vst + (xst * xst.');
    % E(x(t+1)X(t)'|y(1:T))
    CP = CVst(1:r,:) + xs(1:r,t+1) * xst.';    
    if t > 1
        sum_MP(:,:,St) = sum_MP(:,:,St) + P(1:r,1:r);
    end
    sum_MPb(:,:,Stp1) = sum_MPb(:,:,Stp1) + P;
    sum_MCP(:,:,Stp1) = sum_MCP(:,:,Stp1) + CP;
    
end % end t loop

Mx0(:,S(1)) = xst;
MP0(:,:,S(1)) = P;
sum_P = sum(sum_MP,3) + P(1:r,1:r);
xf = xf(1:r,:); 


