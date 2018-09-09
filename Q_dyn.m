% Q-function of EM algorithm (Conditional expectation of complete-data
% log-likelihood given observed data) for switching dynamics model

% Initial state vector X(t)=(x(1),...,x(2-p)) ~
% N((mu_j,...,mu_j),diag(S_j,...,S_j)) conditional on S(1)=j

function Qval = Q_dyn(A,C,Q,R,mu,Sigma,Pi,Z,p,T,...
    MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,sum_P,sum_xy,sum_yy)

    M = numel(Pi);
    N = size(R,1);
    r = size(A,2)/p;

%     % Reduce parameters
%     A = A(1:r,:,:);
%     C = C(:,1:r);
%     Q = Q(1:r,1:r,:);

    % Expand parameters
    mu = repmat(mu,p,1);
    Stmp = zeros(p*r,p*r,M);
    for j = 1:M
        Stmp(:,:,j) = kron(eye(p),Sigma(:,:,j));
    end
    Sigma = Stmp;
    
    % C/R contribution
    [R_chol,err] = cholcov(R);
    if err 
        Qval = Inf;
        return
    end        
    Q_CR = -0.5*T*N*log(2*pi) - T * sum(log(diag(R_chol))) ...
        - 0.5*trace(R\(sum_yy - 2 * C * sum_xy + C * sum_P * C.')); 
     
    % A/Q contribution
    Q_AQ = -0.5*(T-1)*r*log(2*pi);
    for j=1:M
        sum_Mj = sum(Ms(j,2:T));
        if sum_Mj == 0
            continue
        end
        A_j = A(:,:,j);
        Q_j = Q(:,:,j);
        [Qj_chol,err] = cholcov(Q_j);
        if err
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
    Q_muSigma = -0.5*p*r*log(2*pi);
    for j = 1:M
        if Ms(j,1) == 0
            continue
        end
        mu_j = mu(:,j);
        Sigma_j = Sigma(:,:,j);
        [Sigmaj_chol,err] = cholcov(Sigma_j);
        if err
            Qval = Inf;
            return
        end
        Mx0j = Mx0(:,j);
        Q_muSigma = Q_muSigma - Ms(j,1) * sum(log(diag(Sigmaj_chol))) ...
            - 0.5 * trace(Sigma_j \ ...
            (MP0(:,:,j) - 2 * (Mx0j * mu_j') + Ms(j,1) * (mu_j * mu_j')));
        
    end
    
    % Pi/Z contribution
    logPi = log(Pi);
    logPi(isinf(logPi)) = 0;
    Q_Pi = dot(Ms(:,1),logPi); 
    logZ = log(Z(:));
    logZ(isinf(logZ)) = 0;
    Q_Z = dot(sum_Ms2(:),logZ);

    % Total
    Qval = Q_AQ + Q_CR + Q_muSigma + Q_Pi + Q_Z;
end
