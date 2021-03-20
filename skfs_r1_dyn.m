function [Mf,Ms,xf,xs,L,MP0,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,sum_P] = ... 
    skfs_r1_dyn(y,M,p,r,pars,beta,safe,abstol,reltol)

%--------------------------------------------------------------------------
%
%               SWITCHING KALMAN FILTER AND SMOOTHER 
%           IN STATE-STATE MODEL WITH SWITCHING DYNAMICS
%
% PURPOSE
% This is not meant to be directly called by the user. It is called by
% functions 'switch_dyn' and 'fast_dyn' to complete the E step of the EM
% algorithm. SPECIAL CASE r = 1
% 
% USAGE
% [Mf,Ms,xf,xs,L,MP0,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,sum_P] = ... 
%     skfs_r1_dyn(y,M,p,r,A,C,Q,R,mu,Sigma,Pi,Z,beta,safe,abstol,reltol)
%
% REFERENCES
% C. J. Kim (1994) "Dynamic Linear Models with Markov-Switching", Journal of
% Econometrics 60, 1-22.
% K. P. Murphy (1998) "Switching Kalman Filters", Technical Report
%
%--------------------------------------------------------------------------

% Model dimensions
[N,T] = size(y); 
% Size of 'small' state vector x(t): r
% Size of 'big' state vector X(t) = (x(t),...,x(t-p+1)): p * r

A = pars.A; C = pars.C; Q = pars.Q; R = pars.R; mu = pars.mu; 
Sigma = pars.Sigma; Pi = pars.Pi; Z = pars.Z;


% Remove warnings when inverting singular matrices
warning('off','MATLAB:singularMatrix');
warning('off','MATLAB:nearlySingularMatrix');
warning('off','MATLAB:illConditionedMatrix');

% Declare Kalman filter variables
% xp = zeros(p,M,M,T);	% E(x(t)|y(1:t-1),S(t-1)=i,S(t)=j)
% Vp = zeros(p,p,M,M,T); % V(x(t)|y(1:t-1),S(t-1)=i)
xp = zeros(M,M,T);	% E(x(t)|y(1:t-1),S(t-1)=i,S(t)=j)  @@@@ reduce memory footprint
Vp = zeros(M,M,T); % V(x(t)|y(1:t-1),S(t-1)=i)          @@@@ reduce memory footprint
% xf = zeros(p,T);      % E(X(t)|y(1:t))
xf1 = zeros(p,M,T);	% E(X(t)|y(1:t),S(t)=j)
xf2 = zeros(p,M,M);   % E(X(t)|y(1:t),S(t-1)=i,S(t)=j)
Vf1 = zeros(p,p,M,T); % V(X(t)|y(1:t),S(t)=j)
Vf2 = zeros(p,p,M,M); % V(X(t)|y(1:t),S(t-1)=i,S(t)=j)
% CVf2 = zeros(r,r,M,M); % Cov(x(t),x(t-1)|y(1:t),S(t-1)=i,S(t)=j)
Lp = zeros(M,M);        % P(y(t)|y(1:t-1),S(t)=j,S(t-1)=i)
Mf = zeros(M,T);        % P(S(t)=j|y(1:t))
% Mf2 = zeros(M,M);       % P(S(t-1)=i,S(t)=j|y(1:t))

% Declare Kalman smoothing variables
xs = zeros(1,T);      % E(X(t)|y(1:T))
xs2 = zeros(p,M,M);   % E(X(t)|y(1:T),S(t)=j,S(t+1)=k)
Vs2 = zeros(p,p,M,M); % V(X(t)|y(1:T),S(t)=j,S(t+1)=k)
CVs1 = zeros(1,p,M);  % Cov(x(t+1),X(t)|y(1:T),S(t+1)=k) 
CVs2 = zeros(1,p,M,M); % Cov(x(t+1),X(t)|y(1:t),S(t)=j,S(t+1)=k)
Ms = zeros(M,T);        % P(S(t)=j|y(1:T))

% Other outputs
sum_Ms2 = zeros(M,M);   % sum(t=2:T) P(S(t-1)=i,S(t)=j|y(1:T))
sum_MCP = zeros(1,p,M); % sum(t=2:T) P(S(t)=j|y(1:T)) * E(x(t)X(t-1)'|S(t)=j,y(1:T))
sum_MP = zeros(1,1,M);  % sum(t=2:T) P(S(t)=j|y(1:T)) * E(x(t)x(t)'|S(t)=j,y(1:T))
sum_MPb = zeros(p,p,M); % sum(t=2:T) P(S(t)=j|y(1:T)) * E(X(t-1)X(t-1)'|S(t)=j,y(1:T))
% sum_P = zeros(r,r)    % sum(t=1:T) E(x(t)x(t)'|S(t)=j,y(1:T))
MP0 = zeros(p,p,M); % P(S(1)=j|y(1:T)) * E(X(1)X(1)'|S(t)=j,y(1:T))
Mx0 = zeros(p,M);     % P(S(1)=j|y(1:T)) * E(X(1)|S(t)=j,y(1:T))




% Expand matrices
Abig = repmat(diag(ones((p-1)*r,1),-r),[1,1,M]);
Abig(1,:,:) = A;
Cbig = zeros(N,p);
Cbig(:,1) = C;
Qbig = zeros(p,p,M);
Qbig(1,1,:) = Q(:);

% Auxiliary quantities
CCt = C * C';
% RinvC = R\Cbig;
% CtRinvC = dot(C,RinvC(:,1)); 
RinvC = R\C; % @@@@@@@@@
CtRinvC = dot(C,RinvC); 
cst = - N / 2 * log(2*pi);

mu = mu(:);
Sigma = Sigma(:);


%-------------------------------------------------------------------------%
%                        Switching Kalman Filter                          %
%-------------------------------------------------------------------------%   


% Initialize filter
Acc = zeros(M,1);
for j=1:M
    S_j = Sigma(j) * eye(p);
    e = y(:,1) - C * mu(j);
    Ve = S_j(1) * CCt + R;
    if safe
        Ve = regfun(Ve,abstol,reltol);
    end
    xf1(:,j,1) = mu(j) + S_j * Cbig.' * (Ve\e);
    Vf1(:,:,j,1) = S_j - S_j * Cbig.' * (Ve\Cbig) * S_j;
    Acc(j) = Pi(j) * mvnpdf(e.',[],Ve);   
end

if all(Acc == 0)
    Acc = eps * ones(M,1);
end
Mf(:,1) = Acc / sum(Acc);   % P(S(1)=j|y(1))
L = log(sum(Acc));          % log(P(y(1)))

Vhat = zeros(p,p,M);

% MAIN LOOP
for t=2:T

    for i=1:M

        for j=1:M      
            
            % Prediction of x(t)
            xp_ij = Abig(:,:,j) * xf1(:,i,t-1); 
            Vp_ij = Abig(:,:,j) * Vf1(:,:,i,t-1) * Abig(:,:,j).' + Qbig(:,:,j); 
        
            % Store predictions
            xp(i,j,t) = xp_ij(1);
            Vp(i,j,t) = Vp_ij(1);

            % Prediction error for y(t)
            e = y(:,t) - C * xp_ij(1);
            Ve = Vp_ij(1) * CCt + R; % Variance of prediction error
            % Check that variance matrix is positive definite and well-conditioned
            if safe
                Ve = regfun(Ve,abstol,reltol);
            end
            
            % Filtering update 
%             CVp = C * Vp_ij;
%             K = (CVp.') / Ve; % Kalman gain matrix
%             xf2(:,i,j) = xp_ij + K * e;         % E(X(t)|S(t-1)=i,S(t)=j,y(1:t))
            xf2(:,i,j) = xp_ij + ...
                (dot(RinvC,e)/(1+Vp_ij(1)*CtRinvC)) * Vp_ij(:,1);
%             Vf2(:,:,i,j) = Vp_ij - K * CVp;   % V(X(t)|S(t-1)=i,S(t)=j,y(1:t))
            Vf2(:,:,i,j) = Vp_ij - ...
                (CtRinvC/(1+Vp_ij(1)*CtRinvC)) * Vp_ij(:,1) * Vp_ij(1,:);
%             Vf2(:,:,i,j) = Vp_ij - ...
%                 ((RinvC.'*C)/(1+Vp_ij(1)*CtRinvC)) * Vp_ij(:,1) * Vp_ij(1,:);
%             if t == T
%               % Cov(x(t),x(t-1)|S(t-1)=i,S(t)=j,y(1:t))
%                 CVf2(:,:,i,j) = (I - K*C) * A(:,:,j) * Vf1(:,:,i,t-1); 
%             end
 
            % Predictive Likelihood L(i,j,t) = P(y(t)|y(1:t-1),S(t)=j,S(t-1)=i)
            % Choleski decomposition
            [Lchol,err] = chol(Ve,'lower'); 
            if ~err % case: Ve definite positive
                Lp(i,j) = exp(cst - sum(log(diag(Lchol))) - 0.5 * norm(Lchol\e)^2);
            else
                Lp(i,j) = 0;
            end
%             Lp(i,j) = mvnpdf(e.',[],Ve);  

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
    W = Mf2 ./ Mf(:,t)';
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
        
end % end t loop
  

% Collapse M distributions (X(t)|S(t),y(1:t)) to 1 (X(t)|y(1:t))
xf = sum(squeeze(xf1(1,:,:)) .* Mf); % E(X(t)|y(1:t))





%-------------------------------------------------------------------------%
%                        Switching Kalman Smoother                        %
%-------------------------------------------------------------------------%
    

% Initialize smoother at time T
Ms(:,T) = Mf(:,T);
xs(:,T) = xf(:,T);
xsb = zeros(p,M); 
xs1 = xf1(:,:,T);
Vs1 = Vf1(:,:,:,T);
Asmall = reshape(A,p,M);  
MCP = zeros(1,p,M); 
MP = zeros(1,1,M); 
MPb = zeros(p,p,M);
for j = 1:M
    sum_MP(j) = Ms(j,T) * (Vs1(1,1,j) + xs1(1,j)^2);
end
Vsb = zeros(p,p,M);
CVhat = zeros(r,p,M); 

for t = T-1:-1:1
    
    % Store relevant vectors/matrices from previous iteration
    xs1tp1 = xs1(1,:); % E(X(t+1)|S(t+1),y(1:T))
    Vs1tp1 = Vs1(1,1,:); % V(X(t+1)|S(t+1),y(1:T))

    % Predicted and filtered mean and variance (for faster access)
    xptp1 = xp(:,:,t+1);
    Vptp1 = Vp(:,:,t+1);
    xf1t = xf1(:,:,t);
    Vf1t = Vf1(:,:,:,t);

    % Smoothed mean and variance of x(t), smoothed cross-covariance of
    % x(t+1) & X(t) given S(t)=j and S(t+1)=k 
    for j = 1:M      
        for k = 1:M
            % Kalman smoother gain 
            % J(t) = V(X(t)|S(t)=j,y(1:t)) * A_k' * V(x(t+1)|S(t)=j,y(1:t))^{-1}
            J = Vf1t(:,:,j) * Asmall(:,k) / Vptp1(j,k);                  
            % E(X(t)|S(t)=j,S(t+1)=k,y(1:T))
            xs2(:,j,k) = xf1t(:,j) + J * (xs1tp1(k) - xptp1(j,k)); 
            % V(X(t)|S(t)=j,S(t+1)=k,y(1:T))
            Vs2(:,:,j,k) = Vf1t(:,:,j) + J * (Vs1tp1(k) - Vptp1(j,k)) * J.'; 
            % Cov(x(t+1),X(t)|S(t)=j,S(t+1)=k,y(1:T)) = V(x(t+1)|S(t+1)=k,y(1:T)) * J(t)'
            % Equation (20) of "Derivation of Kalman filtering and smoothing equations"
            % by B. M. Yu, K. V. Shenoy, M. Sahani. Technical report, 2004.
            CVs2(:,:,j,k) = Vs1tp1(k) * J.';  
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
    for j = 1:M
        xhat = squeeze(xs2(:,j,:)) * W(j,:)'; % E(X(t)|S(t)=j,y(1:T)) @@@@@@@@
%         xs1(:,j) = W(j,:) * squeeze(xs2(:,j,:)); % @@@@@@@@@@@@
        for k = 1:M
            m = xs2(:,j,k) - xhat;
            Vhat(:,:,k) = W(j,k) * (Vs2(:,:,j,k) + (m*m.'));
        end
        xs1(:,j) = xhat;
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
    xs(t) = xs1(1,:) * Ms(:,t); % E(X(t)|y(1:T))
        
    % Required quantities for M step 
    for j=1:M
        % P(S(t)=j|y(1:T)) * E(x(t)x(t)'|S(t)=j,y(1:T))
        MP(j) = Ms(j,t) * (Vs1(1,1,j) + xs1(1,j).^2);   
        % P(S(t+1)=j|y(1:T)) * E(X(t)X(t)'|S(t+1)=j,y(1:T))
        MPb(:,:,j) = Ms(j,t+1) * (Vsb(:,:,j) + (xsb(:,j) * xsb(:,j).')); 
        % P(S(t)=j|y(1:T)) * E(x(t+1)X(t)'|S(t+1)=j,y(1:T))
        MCP(:,:,j) = Ms(j,t+1) * (CVs1(1,:,j) + xs1tp1(1,j) * xsb(:,j).');    
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

sum_P = sum(sum_MP(:)) + sum(MP(:));

 

