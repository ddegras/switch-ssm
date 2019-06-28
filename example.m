
%=========================================================================%
%           TOY EXAMPLE FOR MATLAB TOOLBOX 'switch-ssm'                   %
%=========================================================================%


clc; clearvars; close all;



 

%-------------------------------------------------------------------------%
%               1) Simulate SSM with switching dynamics                   %
%-------------------------------------------------------------------------%


% Model: 
% y(t) = C(S(t)) x(t) + w(t)                        (observation equation)
% x(t) = A(1,S(t)) x(t-1) + ... + A(p,S(t)) x(t-p) + v(t) (state equation)
% P(S(t)=j | S(t-1)=i) = Z(i,j)                          (regime equation)


%@@@@@ Model dimensions
N = 5; % number of variables
T = 100; % number of time points
M = 2; % number of regimes
p = 2; % VAR order
r = 2; % factor dimension

%@@@@@ Model parameters
% VAR transition matrices A(l,j), l=1:p, j=1:M 
A = zeros(r,r,p,M); 
A(:,:,1,1) = 0.6 * eye(r);
A(:,:,1,2) = 0.9 * eye(r);
A(:,:,2,1) = 0.1 * eye(r);
A(:,:,2,2) = -0.2 * eye(r);
% Observation matrix 
[C,~] = qr(randn(N,r),0); 
% Innovation variance/covariance V(v(t)|S(t)=j), j=1:M 
sigQ = .01;
Q = repmat(sigQ * eye(r),[1,1,M]); 
% Noise variance/covariance matrix V(w(t))
sigR = .005;
R = sigR * eye(N);  
% Initial mean of state vector E(x(1)|S(1)=j), j=1:M 
mu = zeros(r,M); 
% Initial variance/covariance V(x(1)|S(1)=j), j=1:M
Sigma = repmat(.02 * eye(r),[1,1,M]);
% Initial probabilities  P(S(1)=j), j=1:M
% Pi = repelem(1/M,M,1);
% Transition probabilities P(S(t)=j | S(t-1)=i), i,j=1:M
% Z = [.99,.01;.01,.99];


%@@@@@ Simulate data
S = repelem([1,2],[T/2,T/2]); % (for simplicity, take S nonrandom) 
x = zeros(r,T+p-1);
y = zeros(N,T);
x(:,1:p) = mvnrnd(mu(:,S(1))',Sigma(:,:,S(1)),p)';
for t = 1:T
    if t > 1
        vt = mvnrnd(zeros(1,r),Q(:,:,S(t)))';
        idx = t+p-2:-1:t-1;
        Xtm1 = reshape(x(:,idx),p*r,1); % X(t-1)=(x(t-1),...,x(t-p))
        x(:,t+p-1) = reshape(A(:,:,:,S(t)),r,p*r) *  Xtm1 + vt;
    end
    wt = mvnrnd(zeros(1,N),R)';
    y(:,t) = C * x(:,t) + wt;
end
% y = y - mean(y,2); % center each time series on zero


clear vt wt Xtm1


%% 



%-------------------------------------------------------------------------%
%               2) Fit switching SSM to simulated data                    %
%-------------------------------------------------------------------------%



%@@@@@ Minimal example (no algorithm tuning, no estimation constraints)


% Run EM
[Mf,Ms,Sf,Ss,xf,xs,Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,LL] = ... 
    switch_dyn(y,M,p,r);
  



%@@@@@ More advanced example

% Specify EM initialization method: use fixed intervals of length 50 to
% estimate VAR parameters A and Q (one pair of estimate for each interval),
% then cluster the parameter estimates by k-means and re-estimate A(j) and
% Q(j) (j=1:M) over associated segmentation (i.e. clustering) of time
% series range
opts = struct('segmentation','fixed','len',50); 

% Other initialization method: dichotomic segmentation (binary search for
% change points in time series). Set minimal distance between change points 
% to 20
% opts = struct('segmentation','binary','delta',20);

% Set up equality constraints: force mu(j) and Sigma(j) to be equal
% across regimes (j=1:M)
equal = struct('mu',true,'Sigma',true); 

% Set up fixed coefficient constraints: make variance/covariance matrices Q
% and R diagonal (set off-diagonal terms to zero and leave other terms
% unspecified - NaN)
fixed = struct('Q',repmat(diag(NaN(r,1)),[1,1,M]),'R',diag(NaN(N,1)));

% Force the columns of the observation matrix C to have Euclidean norm 1
% and force the eigenvalues associated with VAR processes x(t) to be less
% than .98 (for stationarity)
scale = struct('A',.98,'C',1);

% Find starting parameters for EM
[A0,C0,Q0,R0,mu0,Sigma0,Pi0,Z0,S0] = ...
    init_dyn(y,M,p,r,opts,[],equal,fixed,scale); 

% Fix mu and Sigma to their pilot estimates (these parameters are usually
% not relevant and fixing them can help the numerical stability of the EM
fixed.mu = mu0; 
fixed.Sigma = Sigma0;

% Set number of EM iterations to 10^4, turn off display of EM progress 
control = struct('ItrNo',1e4,'verbose',0);

% Run EM (first pass)
[Mf,Ms,Sf,Ss,xf,xs,Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,LL] = ... 
    switch_dyn(y,M,p,r,A0,C0,Q0,R0,mu0,Sigma0,Pi0,Z0,control,equal,fixed,scale); %#ok<*ASGLU>

% Run deterministic annealing EM (DAEM): useful to check if EM got stuck in
% local optimum of likelihood function. During EM iterations, smoothing
% probabilities P(S(t)=j|y(1:T)), t=1:T, j=1:M are raised to power beta and
% rescaled with beta = beta0^(betarate^(k-1)) on k-th EM iteration. This
% prevents the EM to converge too fast. See help page of switch_dyn for
% other arguments in 'control'
control2 = struct('eps',1e-8,'ItrNo',1000,'beta0',.9,'betarate',.9,...
    'abstol',1e-8,'reltol',1e-8,'safe',false,'verbose',0);

[Mf2,Ms2,Sf2,Ss2,xf2,xs2,Ahat2,Chat2,Qhat2,Rhat2,muhat2,Sigmahat2,...
    Pihat2,Zhat2,LL2] = switch_dyn(y,M,p,r,Ahat,Chat,Qhat,Rhat,...
        muhat,Sigmahat,Pihat,Zhat,control2,equal,fixed,scale);

% Check if DAEM has improved upon first EM pass
disp(LL2(end));
disp(LL(end));



%%    


%-------------------------------------------------------------------------%
%                               3) Misc                                   %
%-------------------------------------------------------------------------%



% A useful diagnostic: check the eigenvalues of the fitted VAR processes
% (the complex eigenvalues should not be to close to 1 in modulus)
for j = 1:M
    if  p == 1
        e = eig(Ahat2(:,:,:,j));
    else
        Abig = zeros(p*r,p*r);
        Abig(1:r,:) = reshape(Ahat2(:,:,:,j),r,p*r);
        Abig(r+1:p*r,1:(p-1)*r) = eye((p-1)*r);
        e = eig(Abig);
    end
    fprintf('Eigenvalues of VAR process for regime %d\n',j);
    disp(abs(e));
end


% Plot estimated regimes (based on smoothed occupancy probabilities
% P(S(t)=j|y(1:T))
plot(Ss2);


% Estimate steady-state variance-covariance matrix V(x(t)|S(t)=j) for
% regime j, j=1:M
VX = zeros(p*r,p*r,M); % V(X(t)|S(t)=j) with X(t)=(x(t),...,x(t-p+1))
for j = 1:M
    if p == 1
        Aj = Ahat2(:,:,1,j);
        B = Aj/(Aj'*Aj);
        VX(:,:,j) = sylvester(A,-B,-Qhat2(:,:,j)*B);
    else
        Abig = zeros(p*r,p*r);
        Abig(1:r,:) = reshape(Ahat(:,:,:,j),r,p*r);
        Abig(r+1:p*r,1:(p-1)*r) = eye((p-1)*r);
        Qbig = zeros(p*r,p*r);
        Qbig(1:r,1:r) = Qhat2(:,:,j);
        B = Abig/(Abig'*Abig);
        VX(:,:,j) = sylvester(Abig,-B,-Qbig*B);
    end
end
Vx = VX(1:r,1:r,:); % V(x(t)|S(t)=j)




    
    



