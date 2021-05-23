function [y,S] = simulate_var(pars,T)

%--------------------------------------------------------------------------
%
%     SIMULATE REALIZATION OF SWITCHING VECTOR AUTOREGRESSIVE MODEL
%
%
% PURPOSE
% Simulate processes y(t) and S(t) (t=1:T) in model
%           y(t) = A(1,S(t)) y(t-1) + ... + A(p,S(t)) y(t-p) + w(t,S(t))
% where w(1),...,w(T) iid ~ N(0,Q(j)), v(1),...,v(T) are independent
% conditionally on S(1),...,S(T) and S(t),t=1:T, is a Markov chain with M
% states (initial probabilities Pi(j) and transition probability matrix
% Z(i,j)).
%
% USAGE
% [y,S] = simulate_dyn(pars,T)
%
% INPUTS
% pars: structure containing model parameters 'A', 'Q', 'mu', 'Sigma', 
%       'Pi', 'Z' with respective dimensions (r,r,p,M), (r,r,M), (r,M), 
%        (r,r,M), (M,1), (M,M)
% T:    time series length
%
% OUTPUTS
% y:    observation vectors, dim = (N,T)
% S:    regimes, dim = (1,T)
%
%--------------------------------------------------------------------------


% Model parameters and dimensions
A = pars.A;
Q = pars.Q;
mu = pars.mu;
Sigma = pars.Sigma;
Pi = pars.Pi;
Z = pars.Z;
[N,~,p,M] = size(A);
A = reshape(A,N,p*N,M);

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
        S(1:p) = sum(u(1) > cumPi);
        for t = p+1:T
            S(t) = sum(u(t) > cumZ(S(t-1),:));
        end
        cp = find(diff(S) ~= 0) + 1;
    end
end

% Generate time series y(t)
w = zeros(N,T); 
for j = 1:M
    Sj = (S == j);
    w(:,Sj) = mvnrnd(zeros(1,N),Q(:,:,j),sum(Sj))'; 
end        
y = zeros(N,T); 
for t = 1:p
    y(:,t) = mvnrnd(mu(:,S(t))',Sigma(:,:,S(t)))';
end
for t = p+1:T
    y(:,t) = A(:,:,S(t)) * reshape(y(:,t-1:-1:t-p),[],1) + w(:,t);
end    


        