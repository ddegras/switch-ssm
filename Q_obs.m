function Qval = Q_obs(pars,Ms,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,...
    sum_P,sum_Pb,sum_yy,x0)

% Q-function of EM algorithm (Conditional expectation of complete-data
% log-likelihood given observed data) for switching observations model


% Model and data dimensions
[r,N,M] = size(sum_Mxy);
p = size(sum_Pb,1) / size(sum_P,1);
T = size(Ms,2);

% Expand parameters mu and Sigma
mu = repmat(pars.mu,[p,1]);
Stmp = zeros(p*r,p*r,M);
for j = 1:M
    Stmp(:,:,j) = kron(eye(p),pars.Sigma(:,:,j));
end
Sigma = Stmp;

% C/R contribution
R = pars.R;
Q_CR = -0.5*T*N*log(2*pi) - 0.5*T*log(det(R)) - 0.5 * trace(R\sum_yy);
for j=1:M
    C_j = pars.C(:,:,j);
    sum_Mxyj = sum_Mxy(:,:,j);
    sum_MPj = sum_MP(:,:,j);
    Q_CR = Q_CR - 0.5 * trace(R \ ...
        (-2 * C_j * sum_Mxyj + (C_j * sum_MPj * C_j.'))); 
end

% A/Q contribution
Q_AQ = -0.5*(T-1)*M*p*r*log(2*pi);
for j = 1:M
    A_j = pars.A(:,:,j);
    Q_j = pars.Q(:,:,j);
    sum_Pj = sum_P(:,:,j);
    sum_Pbj = sum_Pb(:,:,j);
    sum_CPj = sum_CP(:,:,j);
    Q_AQ = Q_AQ - 0.5 * (T-1) * log(det(Q_j)) ...
        - 0.5 * trace(Q_j \ ...          
        (sum_Pj - 2 * sum_CPj * A_j.' + (A_j * sum_Pbj * A_j.')));
end

% mu/Sigma contribution
Q_muSigma = -0.5*M*p*r*log(2*pi);
for j = 1:M
    mu_j = mu(:,j);
    Sigma_j = Sigma(:,:,j);
    P0_j = P0(:,:,j);
    Q_muSigma = Q_muSigma - 0.5 * log(det(Sigma_j)) ...
        - 0.5 * trace(Sigma_j \ ...
        (P0_j - 2 * x0(:,j) * mu_j.' + (mu_j * mu_j.')));
end

% Pi/Z contribution
logPi = log(pars.Pi);
logPi(isinf(logPi)) = 0;
Q_Pi = dot(Ms(:,1),logPi); 
logZ = log(pars.Z(:));
logZ(isinf(logZ)) = 0;
Q_Z = dot(sum_Ms2(:),logZ);

% Total
Qval = Q_AQ + Q_CR + Q_muSigma + Q_Pi + Q_Z;



