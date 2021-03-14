function [Mf,Ms,xf,xs,L,MP0,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,sum_P] = ... 
    skfs_p1r1_dyn(y,M,~,~,pars,beta,safe,abstol,reltol)

%--------------------------------------------------------------------------
%
%               SWITCHING KALMAN FILTER AND SMOOTHER 
%           IN STATE-STATE MODEL WITH SWITCHING DYNAMICS
%
% PURPOSE
% This is not meant to be directly called by the user. It is called by
% functions 'switch_dyn' and 'fast_dyn' to complete the E step of the EM
% algorithm. ******* SPECIAL CASE p = r = 1 *******
% 
% USAGE
% [Mf,Ms,xf,xs,L,MP0,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,sum_P] = ... 
%     skfs_p1r1_dyn(y,M,~,~,A,C,Q,R,mu,Sigma,Pi,Z,beta,safe,abstol,reltol)
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
% xp = zeros(p*r,M,M,T);	% E(x(t)|y(1:t-1),S(t-1)=i,S(t)=j)
% Vp = zeros(p*r,p*r,M,M,T); % V(x(t)|y(1:t-1),S(t-1)=i)
xp = zeros(M,M,T);	% E(x(t)|y(1:t-1),S(t-1)=i,S(t)=j)   @@@@ reduce memory footprint
Vp = zeros(M,M,T);  % V(x(t)|y(1:t-1),S(t-1)=i,S(t)=j)   @@@@ reduce memory footprint
% xf = zeros(1,T);    % E(X(t)|y(1:t))
xf1 = zeros(M,T);	% E(X(t)|y(1:t),S(t)=j)
xf2 = zeros(M,M);   % E(X(t)|y(1:t),S(t-1)=i,S(t)=j)
Vf1 = zeros(M,T);   % V(X(t)|y(1:t),S(t)=j)
% Vf2 = zeros(M,M);   % V(X(t)|y(1:t),S(t-1)=i,S(t)=j)
% CVf2 = zeros(r,r,M,M); % Cov(x(t),x(t-1)|y(1:t),S(t-1)=i,S(t)=j)
Lp = zeros(M,M);    % P(y(t)|y(1:t-1),S(t)=j,S(t-1)=i)
Mf = zeros(M,T);    % P(S(t)=j|y(1:t))
% Mf2 = zeros(M,M);   % P(S(t-1)=i,S(t)=j|y(1:t))

% Declare Kalman smoothing variables
xs = zeros(1,T);      % E(X(t)|y(1:T))
Ms = zeros(M,T);        % P(S(t)=j|y(1:T))

% Other outputs
sum_Ms2 = zeros(M,M);   % sum(t=2:T) P(S(t-1)=i,S(t)=j|y(1:T))
sum_MCP = zeros(M,1); % sum(t=2:T) P(S(t)=j|y(1:T)) * E(x(t)X(t-1)'|S(t)=j,y(1:T))
% sum_MP = zeros(1,1,M);  % sum(t=2:T) P(S(t)=j|y(1:T)) * E(x(t)x(t)'|S(t)=j,y(1:T))
sum_MPb = zeros(M,1); % sum(t=2:T) P(S(t)=j|y(1:T)) * E(X(t-1)X(t-1)'|S(t)=j,y(1:T))
% sum_P = zeros(r,r)    % sum(t=1:T) E(x(t)x(t)'|S(t)=j,y(1:T))
% MP0 = zeros(M,1); % P(S(1)=j|y(1:T)) * E(X(1)X(1)'|S(t)=j,y(1:T))
% Mx0 = zeros(M,1);     % P(S(1)=j|y(1:T)) * E(X(1)|S(t)=j,y(1:T))

% Reshape parameters
A = squeeze(A)'; % size 1xM
Q = squeeze(Q)'; % size 1xM

% Auxiliary quantities 
CCt = C * C';
RinvC = R\C;
CtRinvC = (C')*RinvC; 
cst = - N / 2 * log(2*pi);



%-------------------------------------------------------------------------%
%                        Switching Kalman Filter                          %
%-------------------------------------------------------------------------%   


% Initialize filter
Acc = zeros(M,1);
for j=1:M
    S_j = Sigma(j);
    e = y(:,1) - C * mu(:,j);
    Ve = S_j * CCt + R;
    Ve = 0.5 * (Ve+Ve.');    
    if safe
        Ve = regfun(Ve,abstol,reltol);
    end
    xf1(j,1) = mu(j) + S_j * C.' * (Ve\e);
    Vf1(j,1) = S_j - S_j^2 * C.' * (Ve\C) ;
    Acc(j) = Pi(j) * mvnpdf(e.',[],Ve);   % P(y(1),S(1)=j)
end

if all(Acc == 0)
    Acc = eps * ones(M,1);
end
Mf(:,1) = Acc / sum(Acc);   % P(S(1)=j|y(1))
% xf(1) = dot(xf1(:,1),Acc); % E(x(1)|y(1))
L = log(sum(Acc));          % log(P(y(1)))


% MAIN LOOP
for t=2:T

    % Prediction of x(t)
    xpt = xf1(:,t-1) * A;
    Vpt = Vf1(:,t-1) * (A.^2) + Q;
    
    % Store predictions
    xp(:,:,t) = xpt;
    Vp(:,:,t) = Vpt;
 
    % Filtered variance
    Vf2 = 1 ./ (CtRinvC + 1./Vpt);
    
    for i=1:M

        for j=1:M      
            
            % Prediction error for y(t)
            e = y(:,t) - C * xpt(i,j);
            Ve = Vpt(i,j) * CCt + R; % Variance of prediction error
%             Ve = 0.5 * (Ve+Ve.');
            % Check that variance matrix is positive definite and well-conditioned
            if safe
                Ve = regfun(Ve,abstol,reltol);
            end
            
            % Filtered mean 
            % E(X(t)|S(t-1)=i,S(t)=j,y(1:t))
%             xf2(i,j) = xpt(i,j) + Vpt(i,j) * (C' * (Ve\e)); % slow
            xf2(i,j) = xpt(i,j) + ...
                (Vpt(i,j)/(1+Vpt(i,j)*CtRinvC)) * (RinvC' * e); % fast
 
            % Predictive Likelihood L(i,j,t) = P(y(t)|y(1:t-1),S(t)=j,S(t-1)=i)
            % Choleski decomposition
            [Lchol,err] = chol(Ve,'lower'); 
            if ~err % case: Ve definite positive
                Lp(i,j) = exp(cst - sum(log(diag(Lchol))) - 0.5 * norm(Lchol\e)^2);
            else % case: Ve not definite positive
                Lp(i,j) = 0;
            end
            
        end  % end j loop

    end % end i loop
    
    % P(S(t-1)=i,S(t)=j|y(1:t)) (up to multiplicative constant)
    % Calculated: P(y(t),S(t-1)=i,S(t)=j|y(1:t-1))
    Mf2 = Lp .* Z .* Mf(:,t-1);
    
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
    W = Mf2 ./ (Mf(:,t)');
    W(isnan(W)) = 0;
  
    % Collapse M^2 distributions (X(t)|S(t-1:t),y(1:t)) to M (X(t)|S(t),y(1:t))
    xhat = sum(xf2 .* W); 
    xf1(:,t) = xhat(:);  % E(X(t)|S(t)=j,y(1:t)) j=1:M
%     xhat = repmat(xhat,M,1);    
    Vhat = sum(W .* (Vf2 + (xf2 - xhat).^2));
    Vf1(:,t) = Vhat(:); % V(X(t)|S(t)=j,y(1:t)), j=1:M
    
  
end % end t loop  
  

% Collapse M distributions (X(t)|S(t),y(1:t)) to 1 (X(t)|y(1:t))
xf = sum(xf1 .* Mf); % E(X(t)|y(1:t))





%-------------------------------------------------------------------------%
%                        Switching Kalman Smoother                        %
%-------------------------------------------------------------------------%
    

% Initialize smoother at time T
Ms(:,T) = Mf(:,T);
xs(T) = xf(T);
xs1 = xf1(:,T);
Vs1 = Vf1(:,T);
sum_MP = Ms(:,T) .* (Vs1 + xs1.^2);

for t = T-1:-1:1
    
    % Store relevant vectors/matrices from previous iteration
    xs1tp1 = xs1; % E(X(t+1)|S(t+1),y(1:T))
    Vs1tp1 = Vs1; % V(X(t+1)|S(t+1),y(1:T))

    % Predicted and filtered mean and variance (for faster access)
    xptp1 = xp(:,:,t+1);
    Vptp1 = Vp(:,:,t+1);
    xf1t = xf1(:,t);
    Vf1t = Vf1(:,t);

    % Smoothed mean and variance of x(t), smoothed cross-covariance of
    % x(t+1) & X(t) given S(t)=j and S(t+1)=k 
    % Kalman smoother gain 
    % J(t) = V(X(t)|S(t)=j,y(1:t)) * A_k' * V(x(t+1)|S(t)=j,S(t+1)=k,y(1:t))^{-1}
    J = (Vf1t * A) ./ Vptp1;
    % E(X(t)|S(t)=j,S(t+1)=k,y(1:T))
    xs2 = xf1t + J .* (xs1tp1' - xptp1);
    % V(X(t)|S(t)=j,S(t+1)=k,y(1:T))
    Vs2 = Vf1t + J.^2 .* (Vs1tp1' - Vptp1);
    % Cov(x(t+1),X(t)|S(t)=j,S(t+1)=k,y(1:T)) = V(x(t+1)|S(t+1)=k,y(1:T)) * J(t)'
    % Equation (20) of "Derivation of Kalman filtering and smoothing equations"
    % by B. M. Yu, K. V. Shenoy, M. Sahani. Technical report, 2004.
    CVs2 = J .* Vs1tp1';
    
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
    % E(X(t)|S(t)=j,y(1:T)), j=1:M
    xs1 = sum(W .* xs2,2); 
    % V(X(t)|S(t)=j,y(1:T)), j=1:M
    Vs1 = sum(W .* (Vs2 + (xs2 - xs1).^2),2);
    % Cov(x(t+1),X(t)|S(t+1)=k,y(1:T))
    % B/c of approximation E(x(t+1)|S(t)=j,S(t+1)=k,y(1:T)) ~= E(x(t+1)|S(t+1)=k,y(1:T)),
    % Cov(x(t+1),X(t)|S(t+1)=k,y(1:T)) ~= sum(j=1:M) Cov(x(t+1),X(t)|S(t)=j,S(t+1)=k,y(1:T)) * U(j,k)
    % with U(j,k) = P(S(t)=j|S(t+1)=k,y(1:T))
    CVs1 = sum(U .* CVs2)';
    
    xsb = sum(U .* xs2); 
    Vsb = sum(U .* (Vs2 + (xs2 - xsb).^2)); 
    xsb = xsb(:); % E(X(t)|S(t+1)=k,y(1:T)), k=1:M
    Vsb = Vsb(:); % V(X(t)|S(t+1)=k,y(1:T)), k=1:M
    
    % Collapse M distributions to 1 
    xs(t) = xs1.' * Ms(:,t); % E(X(t)|y(1:T))
        
    % Required quantities for M step 
    % P(S(t)=j|y(1:T)) * E(x(t)x(t)'|S(t)=j,y(1:T)), j=1:M
    MP = Ms(:,t) .* (Vs1 + xs1.^2);
    % P(S(t+1)=j|y(1:T)) * E(X(t)X(t)'|S(t+1)=j,y(1:T)), j=1:M
    MPb = Ms(:,t+1) .* (Vsb + xsb.^2);
    % P(S(t)=j|y(1:T)) * E(x(t+1)X(t)'|S(t+1)=j,y(1:T)), j=1:M    
    MCP = Ms(:,t+1) .* (CVs1 + xs1tp1 .* xsb);
    
    if t > 1
        sum_MP = sum_MP + MP;
    end
    sum_MPb = sum_MPb + MPb;
    sum_MCP = sum_MCP + MCP;
    
end % end t loop



Mx0 = (Ms(:,1) .* xs1)';
MP0 = reshape(Ms(:,1) .* (Vs1 + xs1.^2),1,1,M);

sum_P = sum(sum_MP+MP);
sum_MCP = reshape(sum_MCP,1,1,M);
sum_MP = reshape(sum_MP,1,1,M);
sum_MPb = reshape(sum_MPb,1,1,M);


