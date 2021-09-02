
%=========================================================================%
%           TOY EXAMPLE FOR MATLAB TOOLBOX 'switch-ssm'                   %
%=========================================================================%


clc; clearvars; close all;



% This example is not calibrated to produce good statistical results,
% it is simply meant to illustrate the toolbox functionalities.

% The example demonstrates the toolbox functions for switching dynamics 
% models. The functions for the switching VAR and switching observations
% models work exactly in the same way.




%-------------------------------------------------------------------------%
%               1) Simulate SSM with switching dynamics                   %
%-------------------------------------------------------------------------%


fprintf(['\n\n---------------\n',...
    'DATA SIMULATION',...
    '\n---------------\n\n']);

% Model: 
% y(t) = C(S(t)) x(t) + w(t)                        (observation equation)
% x(t) = A(1,S(t)) x(t-1) + ... + A(p,S(t)) x(t-p) + v(t) (state equation)
% P(S(t)=j | S(t-1)=i) = Z(i,j)                          (regime equation)


%@@@@@ Model dimensions
N = 5; % number of variables
T = 200; % number of time points
M = 2; % number of regimes
p = 2; % VAR order
r = 2; % factor dimension

fprintf('Model: switching dynamics\nM = %d\np = %d\nr = %d\n',...
    M,p,r);

%@@@@@@ Model parameters
% VAR transition matrices A(l,j), l=1:p, j=1:M 
A = zeros(r,r,p,M); 
A(:,:,1,1) = 0.6 * eye(r);
A(:,:,1,2) = 0.9 * eye(r);
A(:,:,2,1) = 0.1 * eye(r);
A(:,:,2,2) = -0.2 * eye(r);
% Observation matrix 
[C,~,~] = svd(randn(N,r),'econ'); 
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
Pi = repelem(1/M,M,1);
% Transition probabilities P(S(t)=j | S(t-1)=i), i,j=1:M
Z = [.98,.02;.02,.98];

% Collect all model parameters in single structure
theta = struct('A',A, 'C',C, 'Q',Q, 'R',R, 'mu',mu, 'Sigma',Sigma, ...
    'Pi',Pi, 'Z',Z);


%@@@@@ Simulate data
fprintf('Generating model... ')
[y,S,x] = simulate_dyn(theta,T);

% Visualize the data
tiledlayout(3,1);
nexttile
plot(x');
title("Hidden state vector")
nexttile
plot(y');
title("Observed time series")
nexttile
plot(S,'*');
title("Regimes")
xlabel("Time")

fprintf('Done')
 


%%



%-------------------------------------------------------------------------%
%               2) Fit switching SSM to simulated data                    %
%-------------------------------------------------------------------------%


fprintf(['\n\n----------------\n',...
    'MODEL ESTIMATION',...
    '\n----------------\n\n']);


% Minimal example (no algorithm tuning, no estimation constraints)
% [Mf,Ms,Sf,Ss,xf,xs,theta,LL] = switch_dyn(y,M,p,r);
  

%@@@@@ More advanced example

% Set the relative tolerance for convergence to 1e-6 (The EM will stop if
% for 5 sucessive iterations, the relative change in log-likelihood is less
% than 1e-6) and the maximum number of EM iterations to 500. Disable
% progress display.
control = struct('eps',1e-6,'ItrNo',500,'verbose',false);

% Specify EM initialization method: use fixed intervals of length 50 to
% estimate VAR parameters A and Q (one pair of estimate for each interval),
% then cluster the parameter estimates by k-means and re-estimate A(j) and
% Q(j) (j=1:M) over associated time segmentation 
opts = struct('segmentation','fixed','len',25); 

% Other initialization method: dichotomic segmentation (binary search for
% change points in time series). Set minimal distance between change points 
% to 20
% opts = struct('segmentation','binary','delta',20);

% Set up equality constraints: force mu(j) and Sigma(j) to be equal
% across regimes (j=1:M)
equal = struct('mu',true,'Sigma',true); 

% Set up fixed coefficient constraints: make covariance matrices Q
% and R diagonal (set off-diagonal terms to zero and leave other terms
% unspecified as NaN)
fixed = struct('Q',repmat(diag(NaN(r,1)),[1,1,M]),'R',diag(NaN(N,1)));

% Force the columns of the observation matrix C to have Euclidean norm 1
% and force the eigenvalues associated with VAR processes x(t) to be less
% than .98 (for stationarity)
scale = struct('A',.98,'C',1);

% Initialize  EM
fprintf('Calculating MLE... ')
[thetahat0,S0] = init_dyn(y,M,p,r,opts,[],equal,fixed,scale); 

% Fix mu and Sigma to their pilot estimates (these parameters are usually
% not relevant and fixing them can help the numerical stability of the EM
fixed.mu = thetahat0.mu; 
fixed.Sigma = thetahat0.Sigma;

% Run EM (first pass)
[Mf,Ms,Sf,Ss,xf,xs,thetahat1,LL] = ...
    switch_dyn(y,M,p,r,thetahat0,control,equal,fixed,scale); %#ok<*ASGLU>


%@@@@@ Run deterministic annealing EM (DAEM): useful to check if EM got
% stuck in local optimum of likelihood function. During EM iterations,
% smoothing probabilities P(S(t)=j|y(1:T)), t=1:T, j=1:M are raised to
% power beta and rescaled with beta = beta0 * betarate^(k-1) on k-th EM
% iteration. This prevents the EM to converge too fast. See help page of
% switch_dyn for other arguments in 'control'
control2 = struct('eps',1e-8,'ItrNo',500,'beta0',.85,'betarate',1.02,...
    'abstol',1e-8,'reltol',1e-8,'safe',false,'verbose',0);

[Mf2,Ms2,Sf2,Ss2,xf2,xs2,thetahat2,LL2] = switch_dyn(y,M,p,r,thetahat1,...
    control2,equal,fixed,scale);

fprintf('Done\n\n')

% Check if DAEM has improved upon first EM pass
fprintf('Log-likelihood for EM: %f\n',max(LL));
fprintf('Log-likelihood for DAEM: %f\n',max(LL2));

% Select the estimate with highest log-likelihood
if max(LL) > max(LL2)
    thetahat = thetahat1; % estimated model parameters
    Shat = Ss;            % estimated regimes
    xhat = xs;            % estimated state vectors
else
    thetahat = thetahat2;
    Shat = Ss2;
    xhat = xs2;
end

% Graph log-likelihood values across iteration
figure
plot(LL)
title('Maximum likelihood estimation under correct model') 
hold on 
x = numel(LL):numel(LL) + numel(LL2) - 1;
plot(x,LL2,'--')
xlabel('Iteration'); ylabel('Log-likelihood')
legend('EM','DAEM','Location','South')
hold off


    
%%




%-------------------------------------------------------------------------%
%                        3) Estimation performance                        %
%-------------------------------------------------------------------------%



% The switching dynamics model has an infinity of observationally
% equivalent parameterizations. For this reason, it is not advisable to
% directly compare parameter estimates to true parameters. On the other
% hand, quantities like the projection matrix P(C) on the linear space
% spanned by the observation matrix C are uniquely defined and can be
% directly compared. The same is true for the regime-specific stationary
% covariances & autocorrelations of the time series y(t) and state vectors
% x(t)

fprintf(['\n\n----------------------\n',...
    'ESTIMATION PERFORMANCE',...
    '\n----------------------\n\n']);



% Visualize estimated and true regimes
figure
plot([S(:),Shat(:)+.02],'*')
ylim([1,2.02])
title("Regime estimation under correct model")
legend('Truth','Estimate','Location','East')


%@@@@@ Match parameter estimates to true parameters by regime
% Because parameters are defined up to label permutations for regimes, this
% matching step is essential for meaningful comparisons

% Permute regime labels in estimates if agreement is less than 0.5
if mean(Shat == S) < 0.5
    sigma = [2,1]; % permutation vector
    thetahat.A = thetahat.A(:,:,:,sigma);
    thetahat.Q = thetahat.Q(:,:,sigma);
    thetahat.mu = thetahat.mu(:,sigma);
    thetahat.Sigma = thetahat.Sigma(:,:,sigma);
    thetahat.Pi = thetahat.Pi(sigma);
    thetahat.Z = thetahat.Z(sigma,sigma);
    Shat = sigma(Shat);
end

%@@@@@ Calculate projection matrices associated with observation matrices C 
[QC,~] = qr(theta.C,0);
PC = QC * QC';
[QC,~] = qr(thetahat.C,0);
PChat = QC * QC';
% Compare the two matrices and calculate relative estimation error  
fprintf('True observation matrix (associated projection):\n\n');
disp(PC); 
fprintf('Estimated observation matrix (associated projection):\n\n');
disp(PChat);
fprintf('Relative error: %f\n',norm(PChat-PC,'fro')/norm(PC,'fro'));

%@@@@@ Calculate true and estimated regime-specific stationary 
% autocorrelation and covariance structures of the state vector x(t) and 
% observation vector y(t)  
% lim V(x(t)|S(1:t)=j), lim V(y(t)|S(1:t)=j), j=1:M, t -> infinity
lagmax = 10; nfreq = 0;
[ACFx,~,Vx] = get_covariance(rmfield(theta,{'C','R'}),lagmax,nfreq);
[ACFy,~,Vy] = get_covariance(theta,lagmax,nfreq);
[ACFxhat,~,Vxhat] = get_covariance(rmfield(thetahat,{'C','R'}),lagmax,nfreq);
[ACFyhat,~,Vyhat] = get_covariance(thetahat,lagmax,nfreq);

% Plot the autocorrelation functions for x
figure
tiledlayout(2,2)
nexttile
plot(ACFx(:,:,1)')
title('True ACF for x, regime 1')
nexttile
plot(ACFx(:,:,2)')
title('True ACF for x, regime 2')
nexttile
plot(ACFxhat(:,:,1)')
title('Estimated ACF for x, regime 1')
nexttile
plot(ACFxhat(:,:,2)')
title('Estimated ACF for x, regime 2')


    


%%

   


%-------------------------------------------------------------------------%
%                          4) Boostrap inference                          %
%-------------------------------------------------------------------------%



% Here we show how to build boostrap confidence intervals (CIs) for an
% identifiable model parameter or function thereof. Let's look for example
% at the regime-specific stationary covariance matrices for y(t)

fprintf(['\n\n-------------------\n',...
    'BOOTSTRAP INFERENCE',...
    '\n-------------------\n\n']);

% Number of bootstraps 
B = 100; 
% This number is selected for calculation speed. It is too small for actual
% data analyses

%@@@@@ Bootstrap the MLE for the switching dynamics SSM
fprintf('Running %d bootstraps... ',B);
thetaboot = bootstrap_dyn(thetahat,T,B,opts,control);
fprintf('Done\n\n')
% Note: by default, calculations are run in parallel and bootstrap
% estimates are automatically matched to the MLE by regimes. The latter
% feature is so that the package user does not have to do the matching 
% explicitly (or forget to do it).

%@@@@@ Calculate 95% bootstrap CIs. The CIs are calculated for all
% identifiable model parameters and for the regime-specific stationary
% covariance, correlation, and autocorrelation. Basic, percentile, and
% normal boostrap CIs are calculated (see e.g., Efron and Tibshirani, 1993)
level = 0.95;
bootci = bootstrap_ci(thetaboot,thetahat,level,lagmax);

% Obtain the lower and upper bounds of the normal CIs for V(y(t)|S(1:t)=j)
% (t large, j=1,2)
L = bootci.COV.normal.lo;
U = bootci.COV.normal.up;

% Assess the average observed coverage
coverage = mean(L <= Vy & Vy <= U, 'all'); 
fprintf(['Bootstrap confidence intervals for regime-specific\n',...
    'stationary covariance matrices of time series\nCI method: normal\n']);
fprintf('Target coverage level: %f\n',level); 
fprintf('Average coverage of %d CIs: %f\n',numel(L),coverage); 





%%

    


%-------------------------------------------------------------------------%
%                            5) Model selection                           %
%-------------------------------------------------------------------------%



% Here we show how to fit switching SSMs over a range of hyperparameters M
% (# regimes), p (autoregressive order), and r (dimension of state vector),
% calculating each time the Akaike and Bayesian information criteria
% (AIC/BIC). For simplicity, we use cold start

fprintf(['\n\n---------------\n',...
    'MODEL SELECTION',...
    '\n---------------\n\n']);


Mgrid = 1:4;
pgrid = 1:3;
rgrid = 1:4;

nM = numel(Mgrid);
np = numel(pgrid);
nr = numel(rgrid); 

thetahat = cell(nM,np,nr);
LL = NaN(nM,np,nr);
AIC = NaN(size(LL));
BIC = NaN(size(LL));
Marr = NaN(size(LL));
parr = NaN(size(LL));
rarr = NaN(size(LL));

fprintf('Fitting models...')

% Model fitting
for i = 1:nM
    for k = 1:nr
        for j = 1:np
            
            % Model dimensions
            M_ = Mgrid(i);
            p_ = pgrid(j);
            r_ = rgrid(k);
            
            % For later use
            Marr(i,j,k) = M_; 
            parr(i,j,k) = p_;
            rarr(i,j,k) = r_;
            
            % Initialize EM
            if j == 1 % cold start
                thetahat0 = init_dyn(y,M_,p_,r_,opts);
            else % warm start
                thetahat0 = thetahat{i,j-1,k};
                Ahat0 = zeros(r_,r_,p_,M_);
                Ahat0(:,:,1:p_-1,:) = thetahat0.A;
                thetahat0.A = Ahat0;
            end
            
            % Run EM
            [~,~,~,~,~,~,thetahatMpr,LLMpr] = ...
                switch_dyn(y,M_,p_,r_,thetahat0,control);
            
            % Store MLE
            thetahat{i,j,k} = thetahatMpr;
            
            % Calculate AIC and BIC
            % Number of free coefficients in A,C,Q,R,mu,Sigma,Pi,Z
            % Note: in a covariance matrix, only the upper half is free
            % Also: mu(1)=...=mu(M) and Sigma(1)=...=Sigma(M) by assumption
            npars = M_*p_*r_^2 + (N*r_) + (M_*r_*(r_+1)/2) + ...
                + (N*(N+1)/2) + r_ + (r_*(r_+1)/2) + M_-1 + (M_ * (M_-1)); 
            LL(i,j,k) =  max(LLMpr);
            AIC(i,j,k) = - 2 * max(LLMpr) + 2 * npars;
            BIC(i,j,k) = - 2 * max(LLMpr) + log(T) * npars;
            
        end
    end
end

fprintf(' Done\n\n')

% Organize all results in a table for easier visualization
result = table(Marr(:),parr(:),rarr(:),LL(:),AIC(:),BIC(:),...
    'VariableNames',{'M','p','r','LL','AIC','BIC'});

fprintf('Best 5 models for AIC:\n\n');
[~,o1] = sort(result.AIC);
disp(result(o1(1:5),:));

fprintf('\nBest 5 models for BIC:\n\n');
[~,o2] = sort(result.BIC);
disp(result(o2(1:5),:));
            
% Extract best model fit
thetabest = thetahat{o1(1)}; % for AIC
% thetabest = thetahat{o2(1)}; % for BIC



            


