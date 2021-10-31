% Load data
load('eeg_eye_state.mat')

% Visualize EEG signals
visualize_mts(eeg,channel)

% Fit standard (non-switching) VAR model
M = 1; p = 6; S = ones(1,T); scale = []; % struct('A',.98);
pars = fast_var(eeg,M,p,S,[],[],[],scale);

% Visualize transition and noise variance matrices A and Q
tiledlayout(1,2)
nexttile
imagesc(reshape(pars.A,N,p*N))
colorbar; title("A"); yticks(1:N); yticklabels(channel)
nexttile
imagesc(pars.Q)
colorbar; title("Q"); yticks(1:N); yticklabels(channel)

% Subtract common stationary component from data
e = eeg;
e(:,1:p) = e(:,1:p) - pars.mu;
for l = 1:p
    e(:,p+1:T) = e(:,p+1:T) - pars.A(:,:,l) * eeg(:,p+1-l:T-l);
end

% visualize_mts(e,channel,2)
%%
% Fit switching VAR to residuals
M = 6; % number of regimes
% Initialization parameters: length of segments, number of random starts 
% for K-means clustering
opts = struct('len',500,'Replicates',50); 
% EM parameters: maximum number of iterations, initial inverse temperature 
% and change rate for DAEM 
control = struct('ItrNo',300,'beta0',.7,'betarate',1.02); 
pars2 = init_var(e,M,p,opts,control,[],[],scale); % EM initialization 
[~,Ms,~,Shat,pars2,LL] = switch_var(e,M,p,pars2,control,[],[],scale); % EM

%% 

figure(1)
tiledlayout(4,4);
for j = 1:M
    nexttile
    imagesc(reshape(pars2.A(:,:,:,j),N,p*N))
    yticklabels(channel)
    title(sprintf("A (Regime %d)",j))
    colorbar
end
for j = 1:M
    nexttile
    imagesc(pars2.Q(:,:,j))
    title(sprintf("Q (Regime %d)",j))
    colorbar
end


tabulate(Shat) % brain functional connectivity regimes 
tabulate(state) % eye status: 0 = open 1 = closed
twoway = crosstab(Shat,state);
disp(twoway); 
disp(twoway ./ sum(twoway,2)); % eye status and FC regime are largely independent

figure(2)
plot(Shat,'*')

