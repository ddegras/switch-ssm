function [y,S,x] = simulate_dyn(pars,T)

%--------------------------------------------------------------------------
%
%           SIMULATE DATA ACCORDING TO STATE-SPACE MODEL 
%                   WITH MARKOV-SWITCHING DYNAMICS
%
%
% PURPOSE
% Simulate processes x(t), y(t), and S(t) (t=1:T) in model
%           y(t) = C x(t) + w(t)
%           x(t) = A(1,S(t)) x(t-1) + ... + A(p,S(t)) x(t-p) + v(t,S(t)) 
% where w(1),...,w(T) iid ~ N(0,R), v(1),...,v(T) are independent
% conditionally on S(1),...,S(T), v(t)|S(t)=j ~ N(0,Q(j)) for j=1:M, 
% and S(t),t=1:T, is a Markov chain with M states (initial probabilities 
% Pi(j) and transition probability matrix Z(i,j)).
%
% USAGE
% [y,S,x] = simulate_dyn(pars,T)
%
% INPUTS
% pars: structure containing model parameters in fields 'A', 'C', 'Q', 'R',
%       'mu', 'Sigma', 'Pi', 'Z' with respective dimensions (r,r,p,M),
%       (N,r), (r,r,M), (N,N), (r,M), (r,r,M), (M,1), (M,M)
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
A = reshape(A,r,p*r,M);

% Generate regime sequence S(t)
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
        
% Generate state vectors x(t)
v = zeros(r,T); 
for j = 1:M
    Sj = (S == j);
    v(:,Sj) = mvnrnd(zeros(1,r),Q(:,:,j),sum(Sj))'; 
end        
x = zeros(r,T+p-1); % x(t), t=2-p:T
x(:,1:p) = mvnrnd(mu(:,S(1))',Sigma(:,:,S(1)),p)';            
for t = 1:T-1
    x(:,t+p) = A(:,:,S(t+1)) * ...
        reshape(x(:,t+p-1:-1:t),[],1) + v(:,t+1);
end      
x = x(:,p:end); % discard (p-1) first values

% Generate observations y(t)
w = mvnrnd(zeros(1,N),R,T)';
y = C * x + w; 

        