function [y,S,x] = simulate_obs(pars,T)

%--------------------------------------------------------------------------
%
%           SIMULATE DATA ACCORDING TO STATE-SPACE MODEL 
%                   WITH MARKOV-SWITCHING OBSERVATIONS
%
%
% PURPOSE
% Simulate processes x(t), y(t), and S(t) (t=1:T) in model
%
%           y(t) = C(S(t)) x(t,S(t)) + w(t)
%           x(t,j) = A(1,j) x(t-1,j) + ... + A(p,j) x(t-p,j) + v(t,j) 
%
% where w(1),...,w(T) iid ~ N(0,R), v(1,j),...,v(T,j) are iid ~ N(0,Q(j))
% for j=1:M, and S(t),t=1:T, is a Markov chain with M states (initial 
% probabilities Pi(j) and transition probability matrix Z(i,j)).
%
% USAGE
% [y,S,x] = simulate_obs(pars,T)
%
% INPUTS
% pars: struct with fields 'A', 'C', 'Q', 'R', 'mu', 'Sigma', 'Pi', 'Z' 
%       of respective dimensions (r,r,p,M), (N,r), (r,r,M), (N,N), (r,M), 
%       (r,r,M), (M,1), (M,M)
% T:    time series length
%
% OUTPUTS
% x:    state vectors, dim = (r,T)
% y:    observation vectors, dim = (N,T)
% S:    regimes, dim = (1,T)
%
%--------------------------------------------------------------------------


% Model parameters and dimensions
A = pars.A;
C = pars.C;
Q = pars.Q;
R = pars.R;
mu = pars.mu;
Sigma = pars.Sigma;
Pi = pars.Pi;
Z = pars.Z;
[r,~,p,M] = size(A);
N = size(C,1);

% Generate regime sequence S(t)
if M == 1
    S = ones(1,T);
else
    S = zeros(1,T);
    cumPi = [0;cumsum(Pi(1:M-1))];
    cumZ = [zeros(M,1), cumsum(Z(:,1:M-1),2)];
    cp = [];
    % Make sure that there is at least one change point 
    % (This part can be commented out) 
    while isempty(cp) 
        u = rand(1,T);
        S(1) = sum(u(1) > cumPi);
        for t = 2:T
            S(t) = sum(u(t) > cumZ(S(t-1),:));
        end
        cp = find(diff(S) ~= 0) + 1;
    end
end

% Generate state vectors x(t) and observations y(t)
y = zeros(N,T);
for j = 1:M
    x = zeros(r,T+p-1); % x(t), t=2-p:T
    cholSigma = chol(Sigma(:,:,j),'lower');
    x(:,1:p) = mu(:,j) + cholSigma * randn(r,p);  
    %    x(:,1:p) = mvnrnd(mu(:,j)',Sigma(:,:,j),p)';  
    A_j = reshape(A(:,:,:,j),r,p*r);
    cholQ = chol(Q(:,:,j),'lower');
    v = cholQ * rand(r,T);
    % mvnrnd(zeros(1,r),Q(:,:,j),T)'; 
    for t = 1:T-1
        x(:,t+p) = A_j * reshape(x(:,t+p-1:-1:t),[],1) + v(:,t+1);
    end          
    x = x(:,p:end); % discard (p-1) first values
    idx = (S == j);
    y(:,idx) = C(:,:,j) * x(:,idx);
end
cholR = chol(R,'lower');
y = y + cholR * randn(N,T);
% y = y + mvnrnd(zeros(1,N),R,T)';
        
