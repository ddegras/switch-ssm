function ci = bootstrap_ci(parsboot,pars,level,lagmax)

%-------------------------------------------------------------------------%
%
% Title:    Bootstrap confidence intervals for switching state-space models
% 
% Purpose:  BOOTSTRAP_CI takes as input bootstrap replicates of the 
%           maximum likelihood estimator (MLE) and produces pointwise 
%           bootstrap confidence intervals (CI) for all model parameters in  
%           each regime as well as for the stationary autocorrelation, 
%           covariance, and correlation of the observed time series in each 
%           regime
%
% Usage:    ci = bootstrap_ci(parsboot,pars,level,lagmax)
%
% Inputs:   parboots - structure containing bootstrap replicates of the  
%               MLE, typically, the result of a call to one of the  
%               functions bootstrap_dyn, bootstrap_obs, or bootstrap_var
%           par - structure containing the MLE, typically, the result of a
%               call to switch_dyn, switch_obs, or switch_var 
%           level - pointwise confidence level of CIs. Default = 0.95
%           lagmax - maximum lag for the autocorrelation 
%
% Outputs:  ci - structure with fields 'A', 'C', ... (all model parameters)
%               and 'ACF', 'COV', 'COR', 'PCOR' (stationary autocorrelation, 
%               covariance, correlation, and partial correlation). Each
%               field has 3 subfields 'percentile', 'basic', and 'normal' 
%               indicating the CI method. Each of these has in turn 2 
%               subfields containing the lower and upper confidence bounds 
%               for the target parameter. 
% 
% Author:   David Degras, david.degras@umb.edu
%           University of Massachusetts Boston
%
%-------------------------------------------------------------------------%



% Number of bootstraps
B = size(parsboot.A,5);

% Confidence level 
if ~exist('level','var')
    level = 0.95;
else
    assert(level > 0 && level < 1)
end
q = norminv((1+level)/2);

% Maximum lag for autocorrelation 
if ~exist('lagmax','var')
    lagmax = 50;
end

% Model dimensions and type
[r,~,~,M] = size(pars.A);
if ~isfield(pars,'C') || isempty(pars.C)
    model = 'var';
elseif ismatrix(pars.C)
    model = 'dyn';
else
    model = 'obs';
end
if strcmp(model,'var') 
    N = r;
else
    N = size(pars.C,1);
end

% Output structure 
ci = struct('A',[], 'C',[], 'Q',[], 'R',[], 'mu',[], 'Sigma',[], ... 
    'Pi',[], 'Z',[], 'ACF',[], 'COV',[], 'COR',[], 'PCOR',[]);
fname = fieldnames(ci);

% Bootstrap CIs for model parameters
for i = 1:8
    f = fname{i};
    if ~isfield(pars,f) || isempty(pars.(f))
        continue
    end
    ndim = ndims(parsboot.(f));
    mean_boot = mean(parsboot.(f),ndim,'omitNaN');
    sd_boot = std(parsboot.(f),1,ndim,'omitNaN');
    loqt_boot = quantile(parsboot.(f),(1-level)/2,ndim);
    upqt_boot = quantile(parsboot.(f),(1+level)/2,ndim);
    ci.(f).percentile.lo = loqt_boot;
    ci.(f).percentile.up = upqt_boot;
    ci.(f).basic.lo = 2 * mean_boot - upqt_boot;
    ci.(f).basic.up = 2 * mean_boot - loqt_boot;
    ci.(f).normal.lo = 2 * pars.(f) - mean_boot - q * sd_boot;
    ci.(f).normal.up = 2 * pars.(f) - mean_boot + q * sd_boot;
end

% MLEs of stationary autocorrelation, covariance, and correlation
[ACF,~,COV,VAR] = get_covariance(pars,lagmax,0);
COR = NaN(N,N,M);
PCOR = NaN(N,N,M);
mask = logical(eye(N));
for j = 1:M
    try 
        COR(:,:,j) = corrcov(COV(:,:,j)+COV(:,:,j)');
    catch
        SDj = sqrt(VAR(:,j));
        SDj(SDj == 0) = 1;
        COR(:,:,j) = diag(1./SDj) * COV * diag(1./SDj);
    end
    iCORj = myinv(COR(:,:,j));
    try 
        PCORj = - corrcov(iCORj + iCORj');
    catch
       SDj = sqrt(diag(PCORj));
        SDj(SDj == 0) = 1;
        PCORj = diag(1./SDj) * PCORj * diag(1./SDj);
    end
    PCORj(mask) = 1;
    PCOR(:,:,j) = PCORj;
end

% Bootstrap distribution of stationary autocorrelation, covariance, and 
% correlation
ACFboot = NaN(N,lagmax+1,M,B);
COVboot = NaN(N,N,M,B);
CORboot = NaN(N,N,M,B);
PCORboot = NaN(N,N,M,B);
parsb = struct('A',[], 'C',[], 'Q',[], 'R',[]);
for b = 1:B
    parsb.A = parsboot.A(:,:,:,:,b);
    parsb.Q = parsboot.Q(:,:,:,b);
    if strcmp(model,'dyn')
        parsb.C = parsboot.C(:,:,b);
        parsb.R = parsboot.R(:,:,b);
    elseif strcmp(model,'obs')
        parsb.C = parsboot.C(:,:,:,b);      
        parsb.R = parsboot.R(:,:,b);
    end        
    [ACFb,~,COVb,CORb,PCORb] = get_covariance(parsb,lagmax,0);
    ACFboot(:,:,:,b) = ACFb;
    COVboot(:,:,:,b) = COVb;
    CORboot(:,:,:,b) = CORb;
    PCORboot(:,:,:,b) = PCORb;
end

% Bootstrap CIs for stationary autocorrelation, covariance, and correlation
for i = 9:12
    f = fname{i};
    switch i
        case 9
            mle = ACF; boot = ACFboot;
        case 10
            mle = COV; boot = COVboot;
        case 11
            mle = COR; boot = CORboot;
        case 12
            mle = PCOR; boot = PCORboot;
    end
    mean_boot = mean(boot,4,'omitNaN');
    sd_boot = std(boot,1,4,'omitNaN');
    loqt_boot = quantile(boot,(1-level)/2,4);
    upqt_boot = quantile(boot,(1+level)/2,4);
    ci.(f).percentile.lo = loqt_boot;
    ci.(f).percentile.up = upqt_boot;
    ci.(f).basic.lo = 2 * mean_boot - upqt_boot;
    ci.(f).basic.up = 2 * mean_boot - loqt_boot;
    ci.(f).normal.lo = 2 * mle - mean_boot - q * sd_boot;
    ci.(f).normal.up = 2 * mle - mean_boot + q * sd_boot;
end
    
    
            