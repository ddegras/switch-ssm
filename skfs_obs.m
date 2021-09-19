function [Mf,Ms,xf,xs,x0,P0,L,sum_CP,sum_MP,sum_Ms2,sum_Mxy,sum_P,sum_Pb] = ...
    skfs_obs(y,M,p,pars,beta,safe,abstol,reltol)
% test
A = pars.A; C = pars.C; Q = pars.Q; R = pars.R; mu = pars.mu; 
Sigma = pars.Sigma; Pi = pars.Pi; Z = pars.Z;

% Model dimensions
[N,T] = size(y);
r = size(A,1);  
% Size of 'small' state vector x(t,j): r
% Size of state vector X(t,j) = (x(t,j),...,x(t-p+1,j)): p * r (all lags, 1 regime j)
% Size of all concatenated state vectors X(t) = (X(t,1),...,X(t,M)): M * p * r (all lags, all regimes)  


% Reshape parameter estimates
% Atmp = zeros(M*p*r,M*p*r); 
Atmp = kron(eye(M),diag(ones((p-1)*r,1),-r));
Asmall = zeros(M*r,M*p*r);   
% Ctmp = zeros(N,M*p*r,M); 
Qtmp = zeros(M*p*r,M*p*r); 
Stmp = zeros(M*p*r,M*p*r);
for j = 1:M
    idx1 = (j-1)*p*r + (1:r);
    idx2 = (j-1)*p*r + (1:p*r);
    idx3 = (j-1)*r + (1:r);   
    Atmp(idx1,idx2) = A(:,:,j);   % for filtering 
    Asmall(idx3,idx2) = A(:,:,j); % for smoothing
    Qtmp(idx1,idx1) = Q(:,:,j);
    Stmp(idx2,idx2) = kron(eye(p),Sigma(:,:,j));
end
mu = repmat(mu,p,1);
mu = mu(:);
A = Atmp;
Q = Qtmp;
Sigma = Stmp; 
clear Atmp Qtmp Stmp


% Remove warnings when inverting singular matrices
warning('off','MATLAB:singularMatrix');
warning('off','MATLAB:nearlySingularMatrix');
warning('off','MATLAB:illConditionedMatrix');

% Declaring Kalman filter variables
xp = zeros(M*r,M,T); % E(X(t)|y(1:t-1),S(t-1)=i)        
Vp = zeros(M*r^2,M,T); % V(X(t)|y(1:t-1),S(t-1)=i)      
Lp = zeros(M,M); % P(y(t)|y(1:t-1),S(t)=j,S(t-1)=i)
% L % P(y(t)|y(1:t-1))
xf = zeros(M*r,T); % E(X(t)|y(1:t))                     
xf1 = zeros(M*p*r,M,T); % E(X(t)|y(1:t),S(t)=j)
xf2 = zeros(M*p*r,M,M); % E(X(t)|y(1:t),S(t-1)=i,S(t)=j)
Vf1 = zeros((M*p*r)^2,T); % V(X(t)|y(1:t),S(t)=j)
Vf2 = zeros(M*p*r,M*p*r,M,M); % V(X(t)|y(1:t),S(t-1)=i,S(t)=j)
Mf = zeros(M,T); % P(S(t)=j|y(1:t))
Mf2 = zeros(M,M); % P(S(t-1)=i,S(t)=j|y(1:t))

% Declaring Kalman smoothing variables
% xs = zeros(M*p*r,T); % E(X(t,l)|y(1:T))
xs = zeros(M*r,T); % E(x(t,l)|y(1:T))                   
xs2 = zeros(M*p*r,M,M); % E(X(t,l)|y(1:T),S(t)=j,S(t+1)=k)
Vs2 = zeros(M*p*r,M*p*r,M,M); % V(X(t,l)|y(1:T),S(t)=j,S(t+1)=k)
CVs2 = zeros(M*r,M*p*r,M,M); % Cov(x(t+1,l),X(t,l)|y(1:T),S(t)=j,S(t+1)=k)
MCP = zeros(M*r,M*p*r,M); % P(S(t)=j|y(1:T)) * E(x(t+1)X(t)'|y(1:T),S(t)=j)
Ms = zeros(M,T); % P(S(t)=j|y(1:T))
MP = zeros(M*p*r,M*p*r,M); % E(X(t)X(t)'|y(1:T),S(t)=j)

% Other outputs
% P0 = zeros(r,r,M); % E(X(1,l)X(1,l)'|y(1:T)) 
sum_CP = zeros(M*r,M*p*r); % sum(t=1:T-1) E(x(t+1,l)X(t,l)'|y(1:T))
% sum_MP = zeros(r,r,M); % sum(t=1:T) P(S(t)=j|y(1:T)) * E(x(t)x(t)'|S(t)=j,y(1:T))
sum_Ms2 = zeros(M,M); % sum(t=2:T) P(S(t-1)=i,S(t)=j|y(1:T))
sum_Mxy = zeros(r,N,M); % sum(t=1:T) P(S(t)=j|y(1:T)) * E(x(t,j)|S(t)=j,y(1:T)) * y(t)'
% sum_P = zeros(r,r,M); % sum(t=2:T) E(x(t,j)x(t,j)'|y(1:T))
% sum_Pb = zeros(r,r,M); % sum(t=1:T-1) E(x(t,j)x(t,j)'|y(1:T))

% Note: final dimension of above outputs may be different


% Masks for accessing diagonal blocks of matrices/arrays

% Mask for V(x(t,1)),...,V(x(t,M)) in V(X(t)) 
tmp = zeros(p*r); tmp(1:r,1:r) = 1; 
mask_Vp = (kron(eye(M),tmp) == 1);

% Mask for V(X(t,1)|S(t)=1),...,V(X(t,M)|S(t)=1), ..., V(X(t,M)|S(t)=M)
% in Mpr x Mpr x M array V(X(t)|S(t)=1), ..., V(X(t)|S(t)=M)
mask_Vf = repmat(kron(eye(M),ones(p*r)) == 1,[1,1,M]);

% Indices of x(t,1),...,x(t,M) (size Mr) in X(t) (size Mpr)
indt = zeros(p*r,M);
indt(1:r,:) = 1;
indt = find(indt);
mask_xX = reshape(indt,r,M); 

% Constant for likelihood calculation
cst = - N / 2 * log(2*pi);




%-------------------------------------------------------------------------%
%                        Switching Kalman Filter                          %
%-------------------------------------------------------------------------%   


% Initialize filter
Acc = zeros(M,1); 
Vf1t = zeros(M*p*r,M*p*r,M);

for j=1:M
    C_j = C(:,:,j);
    idx = mask_xX(:,j);
    e = y(:,1) - C_j * mu(idx);
    Ve = C_j * Sigma(idx,idx) * C_j.' + R;
    if safe
        Ve = regfun(Ve,abstol,reltol);
    end
    Ve = 0.5 * (Ve+Ve.');    
    xf1(:,j,1) = mu;
    xf1(idx,j,1) = xf1(idx,j,1) + Sigma(idx,idx) * C_j.' * (Ve\e);
    Vf1t(:,:,j) = Sigma;
    Vf1t(idx,idx,j) = Vf1t(idx,idx,j) - ...
        Sigma(idx,idx) * C_j.' * (Ve\C_j) * Sigma(idx,idx);
    Acc(j) = Pi(j) * mvnpdf(e.',[],Ve);
end

Vf1(:,1) = Vf1t(mask_Vf);

if all(Acc == 0)
%     if verbose
%         warning('Kalman filter: outlying observation at time point 1'); 
%     end
    Acc = eps * ones(M,1);
end
Mf(:,1) = Acc / sum(Acc);   % P(S(1)=j|y(1))
L = log(sum(Acc));          % log(P(y(1)))
Vhat = zeros(M*p*r,M*p*r,M);





% FILTERING LOOP
for t = 2:T

    % Store filtered variance from previous iteration
    Vf1tm1 = Vf1t; % V(X(t-1)|S(t-1),y(1:t-1))
    
    for i = 1:M % S(t-1)=i
        
        % Prediction of x(t)
        xp_i = A * xf1(:,i,t-1);            % E(X(t)|y(1:t-1),S(t-1)=i)
        Vp_i = A * Vf1tm1(:,:,i) * A.' + Q; % V(X(t)|y(1:t-1),S(t-1)=i)
%         Vp_i = 0.5 * (Vp_i + Vp_i.');
       
        % Store predictions
        xp(:,i,t) = xp_i(indt);     
        Vp(:,i,t) = Vp_i(mask_Vp);
        
        for j = 1:M % S(t)=j      
   
            % Prediction of y(t)
            C_j = C(:,:,j);
            idx = mask_xX(:,j);
            e = y(:,t) - C_j * xp_i(idx); 
            CVp = C_j * Vp_i(idx,:); % C_j * V(X(t)|y(1:t-1),S(t-1)=i) 
            Ve = CVp(:,idx) * C_j.' + R; % Variance of prediction error         

            % Check that variance matrix is positive definite and well-conditioned
            if safe
                Ve = regfun(Ve,abstol,reltol);
            end
           
            % Choleski decomposition
            [Lchol,err] = chol(Ve,'lower'); 
            
            if ~err % case: Ve definite positive
                LinvCVp = Lchol\CVp;
                Linve = Lchol\e;
                % Predictive Likelihood L(i,j,t) = P(y(t)|y(1:t-1),S(t)=j,S(t-1)=i)
                Lp(i,j) = exp(cst - sum(log(diag(Lchol))) - 0.5 * sum(Linve.^2));
                % Filtering update
                xf2(:,i,j) = xp_i + LinvCVp.' * Linve;         % E(X(t)|S(t-1)=i,S(t)=j,y(1:t))
                Vf2(:,:,i,j) = Vp_i - (LinvCVp.' * LinvCVp);   % V(X(t)|S(t-1)=i,S(t)=j,y(1:t))
            else
                Lp(i,j) = 0;
                xf2(:,i,j) = xp_ij;
                Vf2(:,:,i,j) = Vp_ij;
            end
            
%             % Filtering update
%             K = (CVp.') / Ve; % Kalman gain matrix
%             xf2(:,i,j) = xp_i + K * e; 
%             Vf2(:,:,i,j) = Vp_i - K * CVp; % V(X(t)|S(t-1)=i,S(t)=j,y(1:t))
% 
%             % Predictive Likelihood L(i,j,t) = P(y(t)|y(1:t-1),S(t)=j,S(t-1)=i)
% %             Lp(i,j) = mvnpdf(e.',[],Ve); slower
%             try 
%                 % Choleski decomposition
%                 Lchol = chol(Ve,'lower'); 
%                 Lp(i,j) = exp(cst - sum(log(diag(Lchol))) - 0.5 * norm(Lchol\e)^2);
%             catch
%                 Lp(i,j) = 0;
%             end
  
            % P(S(t-1)=i,S(t)=j|y(1:t)) (up to multiplicative constant)
            Mf2(i,j) = Lp(i,j) * Z(i,j) * Mf(i,t-1); % P(y(t),S(t-1)=i,S(t)=j|y(1:t-1))

         end  % end j loop

    end % end i loop
  
    % Numerical control
    if all(Mf2(:) == 0)
        Mf2 = eps * ones(M);
    end
    
    % Update log-likelihood
    % P(y(t)|y(1:t-1)) = sum(i,j) P(y(t)|S(t-1)=i,S(t)=j,y(1:t-1)) *
    % P(S(t)=j|S(t-1)=i) * P(S(t-1)=i|y(t-1))
    L = L + log(sum(Mf2(:)));
    
    % Filtered occupancy probability of state j at time t
    Mf2 = Mf2 / sum(Mf2(:)); % P(S(t-1)=i,S(t)=j|y(1:t))
    Mf(:,t) = sum(Mf2).'; % P(S(t)=j|y(1:t))      

    % Weights of state components
    W = Mf2 ./ (Mf(:,t).'); % P(S(t-1)=i|S(t)=j,y(1:t))
    W(isnan(W)) = 0;
  
    % Collapse M^2 distributions (x(t)|S(t-1:t),y(1:t)) to M (x(t)|S(t),y(1:t))
    for j = 1:M
        xhat =  xf2(:,:,j) * W(:,j);
        for i = 1:M
            m = xf2(:,i,j) - xhat;
            Vhat(:,:,i) = W(i,j) * (Vf2(:,:,i,j) + (m*m.'));
        end
        xf1(:,j,t) = xhat;          % E(X(t)|S(t)=j,y(1:t))
        Vf1t(:,:,j) = sum(Vhat,3);  % V(X(t)|S(t)=j,y(1:t))        
    end    
    % Store filtered variance
    Vf1(:,t) = Vf1t(mask_Vf);
    
    % Collapse M distributions (x(t)|S(t),y(1:t)) to 1 (x(t)|y(1:t))
    xf(:,t) = xf1(indt,:,t) * Mf(:,t); % E(x(t,l)|y(1:t)) 
  
end % end t loop  
  



%-------------------------------------------------------------------------%
%                        Switching Kalman Smoother                        %
%-------------------------------------------------------------------------%

    
% Reshape predicted variance
Vp = reshape(Vp,(M*r)^2,T);

% Redefine mask for container for predicted variance: Mr x Mr x M
mask_Vp = (repmat(kron(eye(M),ones(r)),[1,1,M]) == 1);


% Initialize smoother at time T
Ms(:,T) = Mf(:,T);
xs(:,T) = xf(:,T);
xs1 = xf1(:,:,T);
Vs1 = zeros(M*p*r,M*p*r,M);
Vs1(mask_Vf) = Vf1(:,T);
for j = 1:M
    idx = mask_xX(:,j);
    sum_Mxy(:,:,j) = Ms(j,T) * xs1(idx,j) * y(:,T).';
    MP(:,:,j) = Ms(j,T) * (Vs1(:,:,j) + (xs1(:,j) * xs1(:,j).'));
end
sum_MP = MP;
PT = squeeze(sum(MP,3)); % E(x(T,l)x(T,l)'|y(1:T))
% Vhat = zeros(M*p*r,M*p*r,M);


for t=T-1:-1:1
    
    % Store smoothed mean & variance from previous iteration
    xs1tp1 = xs1; % E(X(t+1,l)|S(t+1)=k,y(1:T))
    Vs1tp1 = Vs1; % V(X(t+1,l)|S(t+1)=k,y(1:T))

    % Predicted and filtered mean and variance (for faster access)
    xp1tp1 = xp(:,:,t+1);
    Vp1tp1 = zeros(M*r,M*r,M); % diag(V(x(t+1,l)|S(t)=j,y(1:t)),l=1:M), j=1:M
    Vp1tp1(mask_Vp) = Vp(:,t+1);
    xf1t = xf1(:,:,t);
    Vf1t = zeros(M*p*r,M*p*r,M); % diag(V(X(t,l)|S(t)=j,y(1:t)),l=1:M), j=1:M
    Vf1t(mask_Vf) = Vf1(:,t);

    
    % Smoothed mean and variance of X(t), smoothed cross-covariance of
    % x(t+1) & X(t) given S(t)=j and S(t+1)=k 
    for j = 1:M
     
        % Kalman smoother gain matrix
        % J(t) = V(X(t,l)|S(t)=j,y(1:t)) * A(l)' * V(x(t+1,l)|S(t)=j,y(1:t))^{-1}
        % with A(l) = (A(1,l),...,A(M,l))     
        J = Vf1t(:,:,j) * Asmall.' / Vp1tp1(:,:,j);
        
        for k = 1:M
            % E(X(t,l)|S(t)=j,S(t+1)=k,y(1:T))
            xs2(:,j,k) = xf1t(:,j) + J * (xs1tp1(indt,k) - xp1tp1(:,j)); 
            % V(X(t,l)|S(t)=j,S(t+1)=k,y(1:T))
            Vs2(:,:,j,k) = Vf1t(:,:,j) + ...
                J * (Vs1tp1(indt,indt,k) - Vp1tp1(:,:,j)) * J.';
            % Indexing by (indt,indt) to extract diagonal blocks
            % V(x(t+1,l)|y(1:T),S(t+1)=k) is okay b/c off-diagonal blocks
            % are already set to zero with code line Vs1(~mask_Vf) = 0;
            
            % Cov(x(t+1,l),x(t,l)|S(t)=j,S(t+1)=k,y(1:T)) = V(x(t+1,l)|S(t+1)=k,y(1:T)) * J(t)'
            % Equation (20) of "Derivation of Kalman filtering and smoothing equations"
            % by B. M. Yu, K. V. Shenoy, M. Sahani. Technical report, 2004.
            CVs2(:,:,j,k) = Vs1tp1(indt,indt,k) * J.';              
        end
            
    end 
    
    % Smoothed probability distribution of S(t)
    U = diag(Mf(:,t)) * Z; % P(S(t)=j|S(t+1)=k,y(1:T))
    U = U ./ repmat(sum(U),M,1); % scaling
    U(isnan(U)) = 0;    
    Ms2 = U * diag(Ms(:,t+1)); % P(S(t)=j,S(t+1)=k|y(1:T))
    if all(Ms2(:) == 0)
        Ms2 = (1/M^2) * ones(M);
    end
    if beta < 1
        Ms2 = Ms2.^beta; % DAEM
    end
    Ms2 = Ms2 / sum(Ms2(:)); % for numerical accuracy
    sum_Ms2 = sum_Ms2 + Ms2;
    Ms(:,t) = sum(Ms2,2); % P(S(t)=j|y(1:T))
    W = Ms2 ./ repmat(Ms(:,t),1,M); % P(S(t+1)=k|S(t)=j,y(1:T)) 
    W(isnan(W)) = 0;   
    
    % Collapse M^2 distributions (x(t)|S(t)=j,S(t+1)=k) to M (x(t)|S(t)=j)      
    xs2p = permute(xs2,[1,3,2]); % @@@@@@@
    for j = 1:M
%         xs1(:,j) = squeeze(xs2(:,j,:)) * W(j,:).'; % E(x(t)|S(t)=j,y(1:T))
        xhat = xs2p(:,:,j) * W(j,:).'; % @@@@@@@@
        for k = 1:M
            m = xs2(:,j,k) - xhat;
            Vhat(:,:,k) = W(j,k) * (Vs2(:,:,j,k) + (m*m.'));
        end
        xs1(:,j) = xhat;
        Vs1(:,:,j) = sum(Vhat,3); % V(X(t)|S(t)=j,y(1:T))
    end
    Vs1(~mask_Vf) = 0;
            
    % Collapse M distributions (x(t)|S(t)=j) to 1 (x(t))
%     xs(:,t) = xs1 * Ms(:,t); % E(x(t)|y(1:T))
    xs(:,t) = xs1(indt,:) * Ms(:,t); % E(x(t)|y(1:T))
    
    % Required quantities for M step  
    for j = 1:M       
%         idx = mask_xX(:,j);
        sum_Mxy(:,:,j) = sum_Mxy(:,:,j) + ...
            Ms(j,t) * xs1(mask_xX(:,j),j) * y(:,t).';
        MP(:,:,j) = Ms(j,t) * (Vs1(:,:,j) + (xs1(:,j) * xs1(:,j).'));
        for k = 1:M
            % Use approximation E(x(t+1)|S(t)=j,S(t+1)=k,y(1:T)) ~= E(x(t+1)|S(t+1)=k,y(1:T))
            MCP(:,:,j,k) = Ms2(j,k) * (CVs2(:,:,j,k) + ...
                (xs1tp1(indt,k) * xs2(:,j,k).')); 
        end            
    end
    sum_CP = sum_CP + sum(sum(MCP,4),3);       
    sum_MP = sum_MP + MP;
    
    
end % end t loop

P0 = sum(MP,3);                 % E(X(1)X(1)'|y(1:T))
sum_P = sum(sum_MP,3) - P0;     % sum(t=2:T) E(X(t)X(t)'|y(1:T))
sum_Pb = sum(sum_MP,3) - PT;    % sum(t=1:T-1) E(X(t)X(t)'|y(1:T))

% Post-process output quantities
x0 = reshape(xs1 * Ms(:,1),[p*r,M]);
xf = reshape(xf,[r,M,T]);
xs = reshape(xs,[r,M,T]);

P0 = reshape(P0(mask_Vf(:,:,1)),[p*r,p*r,M]);
mask_CP = (kron(eye(M),ones(r,p*r)) == 1);
sum_CP = reshape(sum_CP(mask_CP),[r,p*r,M]);
tmp = zeros(p*r); tmp(1:r,1:r) = 1; 
mask_P = kron(eye(M),tmp) == 1;
sum_P = reshape(sum_P(mask_P),[r,r,M]);
sum_Pb = reshape(sum_Pb(mask_Vf(:,:,1)),[p*r,p*r,M]);
sum_MPcopy = sum_MP;
sum_MP = zeros(r,r,M);
for j = 1:M
    ind1 = (j-1)*p*r+1:(j-1)*p*r+r;
    sum_MP(:,:,j) = sum_MPcopy(ind1,ind1,j);
end


end % END FUNCTION

