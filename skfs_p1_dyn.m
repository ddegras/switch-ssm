function [Mf,Ms,xf,xs,L,MP0,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,sum_P] = ... 
    skfs_p1_dyn(y,M,~,r,pars,beta,safe,abstol,reltol)

%--------------------------------------------------------------------------
%
%               SWITCHING KALMAN FILTER AND SMOOTHER 
%           IN STATE-STATE MODEL WITH SWITCHING DYNAMICS
%
% PURPOSE
% This is not meant to be directly called by the user. It is called by
% the function 'switch_dyn' to complete the E step of the EM algorithm
% 
% USAGE
% [Mf,Ms,xf,xs,L,MP0,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,sum_P] = ... 
%     skfs_p1_dyn(y,M,~,r,A,C,Q,R,mu,Sigma,Pi,Z,beta,safe,abstol,reltol)
%
% REFERENCES
% C. J. Kim (1994) "Dynamic Linear Models with Markov-Switching", Journal of
% Econometrics 60, 1-22.
% K. P. Murphy (1998) "Switching Kalman Filters", Technical Report
%
%--------------------------------------------------------------------------

% Model dimensions
[N,T] = size(y); 
p = size(pars.A,2) / size(pars.A,1);
% Size of 'small' state vector x(t): r
% Size of 'big' state vector X(t) = (x(t),...,x(t-p+1)): p * r

A = pars.A; C = pars.C; Q = pars.Q; R = pars.R; mu = pars.mu; 
Sigma = pars.Sigma; Pi = pars.Pi; Z = pars.Z;


% Remove warnings when inverting singular matrices
warning('off','MATLAB:singularMatrix');
warning('off','MATLAB:nearlySingularMatrix');
warning('off','MATLAB:illConditionedMatrix');

% Declare Kalman filter variables
% xp = zeros(p*r,M,M,T);	% E(x(t)|y(1:t-1),S(t-1)=i,S(t)=j)
% Vp = zeros(p*r,p*r,M,M,T); % V(x(t)|y(1:t-1),S(t-1)=i)
xp = zeros(r,M,M,T);	% E(x(t)|y(1:t-1),S(t-1)=i,S(t)=j)  @@@@ reduce memory footprint
Vp = zeros(r,r,M,M,T); % V(x(t)|y(1:t-1),S(t-1)=i,S(t)=j)   @@@@ reduce memory footprint
xf = zeros(r,T);      % E(x(t)|y(1:t))
xf1 = zeros(r,M,T);	% E(X(t)|y(1:t),S(t)=j)
xf2 = zeros(r,M,M);   % E(X(t)|y(1:t),S(t-1)=i,S(t)=j)
Vf1 = zeros(r,r,M,T); % V(X(t)|y(1:t),S(t)=j)
Vf2 = zeros(r,r,M,M); % V(X(t)|y(1:t),S(t-1)=i,S(t)=j)
% CVf2 = zeros(r,r,M,M); % Cov(x(t),x(t-1)|y(1:t),S(t-1)=i,S(t)=j)
Lp = zeros(M,M);        % P(y(t)|y(1:t-1),S(t)=j,S(t-1)=i)
Mf = zeros(M,T);        % P(S(t)=j|y(1:t))
% Mf2 = zeros(M,M);       % P(S(t-1)=i,S(t)=j|y(1:t))

% Declare Kalman smoothing variables
xs = zeros(r,T);      % E(x(t)|y(1:T))
xs2 = zeros(r,M,M);   % E(X(t)|y(1:T),S(t)=j,S(t+1)=k)
Vs2 = zeros(r,r,M,M); % V(X(t)|y(1:T),S(t)=j,S(t+1)=k)
CVs1 = zeros(r,r,M);  % Cov(x(t+1),X(t)|y(1:T),S(t+1)=k) 
CVs2 = zeros(r,r,M,M); % Cov(x(t+1),X(t)|y(1:t),S(t)=j,S(t+1)=k)
Ms = zeros(M,T);        % P(S(t)=j|y(1:T))

% Other outputs
sum_Ms2 = zeros(M,M);   % sum(t=2:T) P(S(t-1)=i,S(t)=j|y(1:T))
sum_MCP = zeros(r,r,M); % sum(t=2:T) P(S(t)=j|y(1:T)) * E(x(t)X(t-1)'|S(t)=j,y(1:T))
sum_MP = zeros(r,r,M);  % sum(t=2:T) P(S(t)=j|y(1:T)) * E(x(t)x(t)'|S(t)=j,y(1:T))
sum_MPb = zeros(r,r,M); % sum(t=2:T) P(S(t)=j|y(1:T)) * E(X(t-1)X(t-1)'|S(t)=j,y(1:T))
% sum_P = zeros(r,r)    % sum(t=1:T) E(x(t)x(t)'|S(t)=j,y(1:T))
MP0 = zeros(r,r,M); % P(S(1)=j|y(1:T)) * E(X(1)X(1)'|S(t)=j,y(1:T))
Mx0 = zeros(r,M);     % P(S(1)=j|y(1:T)) * E(X(1)|S(t)=j,y(1:T))



% Auxiliary quantities
cst = - N / 2 * log(2*pi);




%-------------------------------------------------------------------------%
%                        Switching Kalman Filter                          %
%-------------------------------------------------------------------------%   


% Initialize filter
Acc = zeros(M,1);
for j=1:M
    e = y(:,1) - C * mu(:,j);
    Sigma_j =  Sigma(:,:,j);
    Ve = C * Sigma_j * C.' + R;
    if safe
        Ve = regfun(Ve,abstol,reltol);
    end
    Lchol = chol(Ve,'lower');            
    LinvCVp = (Lchol\C) * Sigma_j;
    Linve = Lchol\e;
    Acc(j) = Pi(j) * exp(cst - sum(log(diag(Lchol))) - 0.5 * sum(Linve.^2));
    xf1(:,j,1) = mu(:,j) + LinvCVp.' * Linve; 
    Vf1(:,:,j,1) = Sigma_j - (LinvCVp.' * LinvCVp);      
end

if all(Acc == 0)
    Acc = eps * ones(M,1);
end
Mf(:,1) = Acc / sum(Acc);   % P(S(1)=j|y(1))
xf(:,1) = xf1(:,:,1) * Mf(:,1); % E(x(1)|y(1))
L = log(sum(Acc));          % log(P(y(1)))

Vhat = zeros(p*r,p*r,M);

% MAIN LOOP
for t=2:T

    for i=1:M

        for j=1:M      
            
            % Prediction of x(t)
            xp_ij = A(:,:,j) * xf1(:,i,t-1); 
            Vp_ij = A(:,:,j) * Vf1(:,:,i,t-1) * A(:,:,j).' + Q(:,:,j); 
         
            % Store predictions
            xp(:,i,j,t) = xp_ij;
            Vp(:,:,i,j,t) = Vp_ij;

            % Prediction error for y(t)
            e = y(:,t) - C * xp_ij; 
            CVp = C * Vp_ij;
            Ve = CVp * C.' + R; % Variance of prediction error
%             Ve = 0.5 * (Ve+Ve.');
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
                xf2(:,i,j) = xp_ij + LinvCVp.' * Linve;         % E(X(t)|S(t-1)=i,S(t)=j,y(1:t))
                Vf2(:,:,i,j) = Vp_ij - (LinvCVp.' * LinvCVp);   % V(X(t)|S(t-1)=i,S(t)=j,y(1:t))
            else
                Lp(i,j) = 0;
                xf2(:,i,j) = xp_ij;
                Vf2(:,:,i,j) = Vp_ij;
            end
           
         end  % end j loop

    end % end i loop
  
    % P(S(t-1)=i,S(t)=j|y(1:t)) (up to multiplicative constant)
    Mf2 = Lp .* Z .* Mf(:,t-1); % P(y(t),S(t-1)=i,S(t)=j|y(1:t-1))
    if all(Mf2(:) == 0)
        Mf2 = eps * ones(M,M);
    end
    
    % Update log-likelihood
    % P(y(t)|y(1:t-1)) = sum(i,j) P(y(t)|S(t-1)=i,S(t)=j,y(1:t-1)) *
    % P(S(t)=j|S(t-1)=i) * P(S(t-1)=i|y(t-1))
    L = L + log(sum(Mf2(:))); 
    
    % Filtered occupancy probability of state j at time t
    Mf2 = Mf2 / sum(Mf2(:)); % P(S(t-1)=i,S(t)=j|y(1:t))
    Mf(:,t) = sum(Mf2).';    % P(S(t)=j|y(1:t))      

    % Weights of state components
    W = Mf2 ./ (Mf(:,t).');
    W(isnan(W)) = 0;
  
    % Collapse M^2 distributions (X(t)|S(t-1:t),y(1:t)) to M (X(t)|S(t),y(1:t))
    for j = 1:M
        xhat = xf2(:,:,j) * W(:,j);
        for i = 1:M
            m = xf2(:,i,j) - xhat;
            Vhat(:,:,i) = W(i,j) * (Vf2(:,:,i,j) + (m*m.'));
        end 
        % Filtered density of x(t) given state j
        xf1(:,j,t) = xhat;   % E(X(t)|S(t)=j,y(1:t))     (Eq. 11)
        Vf1(:,:,j,t) = sum(Vhat,3); % V(X(t)|S(t)=j,y(1:t))   (Eq. 12)
    end
  
    % Collapse M distributions (X(t)|S(t),y(1:t)) to 1 (X(t)|y(1:t))
    xf(:,t) = xf1(:,:,t) * Mf(:,t); % E(X(t)|y(1:t))
  
end % end t loop  
  






%-------------------------------------------------------------------------%
%                        Switching Kalman Smoother                        %
%-------------------------------------------------------------------------%
    

% Initialize smoother at time T
Ms(:,T) = Mf(:,T);
xs(:,T) = xf(:,T);
xsb = zeros(r,M); 
xs1 = xf1(:,:,T);
Vs1 = Vf1(:,:,:,T);
MCP = zeros(r,r,M); 
MP = zeros(r,r,M); 
MPb = zeros(r,r,M);
for j = 1:M
    sum_MP(:,:,j) = Ms(j,T) * (Vs1(:,:,j) + (xs1(:,j) * xs1(:,j).'));
end
Vsb = zeros(p*r,p*r,M);
CVhat = zeros(r,p*r,M); 

for t = T-1:-1:1
    
    % Store relevant vectors/matrices from previous iteration
    xs1tp1 = xs1;        % E(x(t+1)|S(t+1),y(1:T))
    Vs1tp1 = Vs1;    % V(x(t+1)|S(t+1),y(1:T))

    % Predicted and filtered mean and variance (for faster access)
    xptp1 = xp(:,:,:,t+1);
    Vptp1 = Vp(:,:,:,:,t+1);
    xf1t = xf1(:,:,t);
    Vf1t = Vf1(:,:,:,t);

    % Smoothed mean and variance of x(t), smoothed cross-covariance of
    % x(t+1) & X(t) given S(t)=j and S(t+1)=k 
    for j = 1:M      
        for k = 1:M
            % Kalman smoother gain 
            % J(t) = V(X(t)|S(t)=j,y(1:t)) * A_k' * V(x(t+1)|S(t)=j,y(1:t))^{-1}
            J = Vf1t(:,:,j) * A(:,:,k).' / Vptp1(:,:,j,k);
            if any(isnan(J(:))) || any(isinf(J(:)))
                J = Vf1t(:,:,j) * A(:,:,k).' * pinv(Vptp1(:,:,j,k));
            end    
            % E(X(t)|S(t)=j,S(t+1)=k,y(1:T))
            xs2(:,j,k) = xf1t(:,j) + J * (xs1tp1(:,k) - xptp1(:,j,k)); 
            % V(X(t)|S(t)=j,S(t+1)=k,y(1:T))
            Vs2(:,:,j,k) = Vf1t(:,:,j) + J * (Vs1tp1(:,:,k) - Vptp1(:,:,j,k)) * J.'; 
            % Cov(x(t+1),X(t)|S(t)=j,S(t+1)=k,y(1:T)) = V(x(t+1)|S(t+1)=k,y(1:T)) * J(t)'
            % Equation (20) of "Derivation of Kalman filtering and smoothing equations"
            % by B. M. Yu, K. V. Shenoy, M. Sahani. Technical report, 2004.
            CVs2(:,:,j,k) = Vs1tp1(:,:,k) * J.';  
        end
    end    
    
    % Smoothed probability distribution of S(t)
    U = diag(Mf(:,t)) * Z; % P(S(t)=j|S(t+1)=k,y(1:T))
    U = U ./ sum(U); % scaling
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
    W = Ms2 ./ Ms(:,t); % P(S(t+1)=k|S(t)=j,y(1:T)) 
    W(isnan(W)) = 0;
    
    % Collapse M^2 distributions to M 
    xs2p = permute(xs2,[1,3,2]);
    for j = 1:M
        xhat = xs2p(:,:,j) * W(j,:).';
%         xs1(:,j) = squeeze(xs2(:,j,:)) * W(j,:)';  @@@@@@@@
%         xs1(:,j) = W(j,:) * squeeze(xs2(:,j,:)); % @@@@@@@@@@@@
        for k = 1:M
            m = xs2(:,j,k) - xhat;
            Vhat(:,:,k) = W(j,k) * (Vs2(:,:,j,k) + (m*m.'));
        end
        xs1(:,j) = xhat; % E(X(t)|S(t)=j,y(1:T))
        Vs1(:,:,j) = sum(Vhat,3); % V(X(t)|S(t)=j,y(1:T))
    end
    % Cov(x(t+1),X(t)|S(t+1)=k,y(1:T))
    % B/c of approximation E(x(t+1)|S(t)=j,S(t+1)=k,y(1:T)) ~= E(x(t+1)|S(t+1)=k,y(1:T)),
    % Cov(x(t+1),X(t)|S(t+1)=k,y(1:T)) ~= sum(j=1:M) Cov(x(t+1),X(t)|S(t)=j,S(t+1)=k,y(1:T)) * U(j,k)
    % with U(j,k) = P(S(t)=j|S(t+1)=k,y(1:T))
    for k = 1:M  
        for j = 1:M
            CVhat(:,:,j) = U(j,k) * CVs2(:,:,j,k); 
        end
        CVs1(:,:,k) = sum(CVhat,3);
    end    
    % V(X(t)|S(t+1)=k,y(1:T))
    for k = 1:M
        xsb(:,k) = xs2(:,:,k) * U(:,k); % E(X(t)|S(t+1)=k,y(1:T))
        for j = 1:M
            m = xs2(:,j,k) - xsb(:,k);
            Vhat(:,:,j) = U(j,k) * (Vs2(:,:,j,k) + (m*m.'));
        end
        Vsb(:,:,k) = sum(Vhat,3);
    end

    % Collapse M distributions to 1 
    xs(:,t) = xs1 * Ms(:,t); % E(X(t)|y(1:T))
        
    % Required quantities for M step 
    for j=1:M
        % P(S(t)=j|y(1:T)) * E(x(t)x(t)'|S(t)=j,y(1:T))
        MP(:,:,j) = Ms(j,t) * (Vs1(:,:,j) + (xs1(:,j) * xs1(:,j).'));   
        % P(S(t+1)=j|y(1:T)) * E(X(t)X(t)'|S(t+1)=j,y(1:T))
        MPb(:,:,j) = Ms(j,t+1) * (Vsb(:,:,j) + (xsb(:,j) * xsb(:,j).')); 
        % P(S(t)=j|y(1:T)) * E(x(t+1)X(t)'|S(t+1)=j,y(1:T))
        MCP(:,:,j) = Ms(j,t+1) * (CVs1(:,:,j) + xs1tp1(:,j) * xsb(:,j).');    
    end
    if t > 1
        sum_MP = sum_MP + MP;
    end
    sum_MPb = sum_MPb + MPb;
    sum_MCP = sum_MCP + MCP;
    
end % end t loop

for j = 1:M
    Mx0(:,j) = Ms(j,1) * xs1(:,j);
    MP0(:,:,j) = Ms(j,1) * (Vs1(:,:,j) + (xs1(:,j) * xs1(:,j)'));
end
sum_P = sum(sum_MP,3) + sum(MP,3);


