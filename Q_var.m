% Q-function of EM algorithm (Conditional expectation of complete-data
% log-likelihood given observed data) for switching VAR model

% Initial state vectors x(1),...,x(p) independent conditional on S(1:p)
% and x(t)|S(t)=j ~ N(mu(j),Sigma(j)) 

function Qval = Q_var(pars,Ms,sum_MCP,sum_MP,sum_MPb,sum_Ms2,x)
    
M = numel(pars.Pi);
[N,T] = size(x);
p = size(pars.A,2) / N;

% A/Q contribution
Q_AQ = -0.5*(T-p)*N*log(2*pi);
for j=1:M
    sum_Mj = sum(Ms(j,p+1:T));
    A_j = pars.A(:,:,j);
    Q_j = pars.Q(:,:,j);
    try
        Qj_chol = chol(Q_j);
    catch
        Qval = Inf;
        return
    end
    sum_MPj = sum_MP(:,:,j);
    sum_MPbj = sum_MPb(:,:,j);
    sum_MCPj = sum_MCP(:,:,j);
    Q_AQ = Q_AQ - sum_Mj * sum(log(diag(Qj_chol))) - ...
         0.5 * trace(Q_j \ (sum_MPj - 2 * sum_MCPj * A_j.' ...
         + A_j * sum_MPbj * A_j.'));
end

% mu/Sigma contribution
Q_muSigma = -0.5*p*N*log(2*pi);
for j = 1:M
    sum_Mj = sum(Ms(j,1:p));
    if sum_Mj == 0
        continue
    end
    try 
        Sigmaj_chol = chol(pars.Sigma(:,:,j));
    catch
        Qval = Inf;
        return
    end
    xc = Sigmaj_chol \ ...
        ((x(:,1:p) - repmat(pars.mu(:,j),1,p)) * diag(sqrt(Ms(j,1:p))));
    Q_muSigma = Q_muSigma - sum_Mj * sum(log(diag(Sigmaj_chol))) ...
        - 0.5 * sum(xc(:).^2);     
end

% Pi/Z contribution
logPi = log(pars.Pi);
logPi(isinf(logPi)) = 0;
Q_Pi = dot(Ms(:,1),logPi); 
logZ = log(pars.Z(:));
logZ(isinf(logZ)) = 0;
Q_Z = dot(sum_Ms2(:),logZ);

% Total
Qval = Q_AQ + Q_muSigma + Q_Pi + Q_Z;

