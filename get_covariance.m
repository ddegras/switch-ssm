function [ACF,COH,COV,VAR] = get_covariance(pars,lagmax,nfreq)

%-------------------------------------------------------------------------%
%
% Purpose:  calculate stationary autocorrelation function, coherence function,
%           and instant covariance matrix in switching state-space model 
%           y(t) = C(S(t)) x(t) + w(t)
%           x(t) = A(1,S(t)) x(t-1) + ... + A(p,S(t)) x(t-p) + v(t)
%
% Usage:    [ACF,COH,COV,VAR] = get_covariance(A,C,Q,R,lagmax,nfreq)
%
% Inputs:   pars - struct with fields  
%               A - transition matrices ([r,r,p,M])
%               C - observation matrices ([N,r], [N,r,M], or [])
%               Q - state noise covariance matrices ([N,N,M]) 
%               R - observation noise covariance matrix ([N,N] or [])
%           lagmax - maximum lag for autocorrelation function (optional, 
%               default = 2000) 
%           nfreq - number of frequencies at which to calculate coherence
%               (optional, default = 100)
%
% Outputs:  ACF - autocorrelation functions ([N,lagmax])
%           COH - coherence functions ([N,N,nfreq])
%           COV - instantaneous covariance matrices ([N,N,M])
%           VAR - variances ([N,M])
%
%-------------------------------------------------------------------------%

narginchk(1,3);

assert(isstruct(pars))

A = pars.A;
Q = pars.Q;
if isfield(pars,'C')
    C = pars.C;
else
    C = [];
end
if isfield(pars,'R')
    R = pars.R;
else
    R = [];
end
if ~exist('lagmax','var') || isempty(lagmax)
    lagmax = 2000;
end
if ~exist('nfreq','var') || isempty(nfreq)
    nfreq = 100;
end

% Model dimensions
[r,~,p,M] = size(A);

% Check arguments C and R
if ~isempty(C) || ~isempty(R)
    assert(~isempty(C) && ~isempty(R));
    N = size(C,1);
    if ismatrix(C)
        C = repmat(C,1,1,M);
    end
else
    N = r;
end

    

% Initialize various quantities
ACF = NaN(N,lagmax+1,M); % auto-correlation diag(cor(x(t),x(t-l)|S(t)=j)
COH = NaN(N,N,nfreq,M); % coherence
COV = NaN(N,N,M); % instantaneous covariance Cov(x(t)|S(t)=j)
Abig = zeros(p*r); % container for A 
invAbig = zeros(p*r);
if p > 1
    Abig(r+1:end,1:end-r) = eye((p-1)*r);
    invAbig(1:end-r,r+1:end) = eye((p-1)*r);
end
Qbig = zeros(p*r); % container for Q
idx_acf = (repmat(eye(N),1,1,lagmax+1) == 1);
idx_coh = (repmat(eye(N),1,1,nfreq) == 1);

% Calculations
for j = 1:M
    A_j = A(:,:,:,j);
    Abig(1:r,:) = reshape(A_j,r,p*r); 
    Qbig(1:r,1:r) = Q(:,:,j);
    if all(A_j(:) == 0)
        Vbig = Qbig;
    else
        eigA = abs(eig(Abig));
        if any(eigA >= 1) 
            continue
        elseif min(eigA) <= 1e-8 * max(1,max(eigA)) % case: Abig numerically singular
            Vbig = get_covariance_aux(A_j,Q(:,:,j)); % Cov(X(t)|S(t)=j)
        else % case: Abig full rank
            invAbig((p-1)*r+1:p*r,1:r) = inv(A_j(:,:,p));
            if p > 1
                invAbig((p-1)*r+1:p*r,r+1:p*r) = ...
                    -A_j(:,:,p)\Abig(1:r,1:(p-1)*r);
            end
            Vbig = sylvester(invAbig,-Abig',invAbig*Qbig); 
        end
    end        
    Vbig = 0.5 * (Vbig + Vbig.');
    if isempty(C)
        COV(:,:,j) = Vbig(1:r,1:r);
    else
        COV(:,:,j) = (C(:,:,j) * Vbig(1:r,1:r) * C(:,:,j).') + R;
    end
    COV(:,:,j) = 0.5 * (COV(:,:,j) + COV(:,:,j).');
    
    if lagmax > 0    
        CCV_tmp = Vbig; % Cov(X(t),X(t-l)|S(t)=j)
        CCV = zeros(r,r,lagmax+1); % Cov(x(t),x(t-l)|S(t)=j)
        CCV(:,:,1) = Vbig(1:r,1:r); 
        A_up = Abig(1:r,:);
        for l = 1:lagmax
            B = A_up * CCV_tmp;
            CCV_tmp(r+1:end,:) = CCV_tmp(1:end-r,:);
            CCV_tmp(1:r,:) = B;
            CCV(:,:,l+1) = CCV_tmp(1:r,1:r);
        end
        if ~isempty(C) % Cov(y(t),y(t-l)|S(t)=j)
            CCV = C(:,:,j) * reshape(CCV,r,r*(lagmax+1));
            CCV = reshape(CCV,N,r,lagmax+1);
            CCV = permute(CCV,[1,3,2]);
            CCV = reshape(CCV,N*(lagmax+1),r) * C(:,:,j).';
            CCV = reshape(CCV,N,lagmax+1,N);
            CCV = permute(CCV,[1,3,2]);
            CCV(:,:,1) = COV(:,:,j);
        end
        % Autocorrelation 
        ACF(:,:,j) = reshape(CCV(idx_acf),N,lagmax+1);
        ACF(:,:,j) = ACF(:,:,j) ./ diag(COV(:,:,j));
    end
    
    if nfreq == 0
        continue
    end
    % Contribution from non-negative lags to cross-spectral density (CSD) 
    % Angular frequency (stop at normalized Nyquist frequency = pi rad/sample)
 	w = linspace(0, pi, nfreq+1); 
    w = w(1:end-1);
    lags = 0:lagmax;
    cplxsin = exp(-1j * lags' * w); % complex sinusoidal wave
    CSD = reshape(CCV,N^2,lagmax+1) * cplxsin;
    % Contribution from negative lags 
    % R_xy(-h) = cov(x(t),y(t+h)) = cov(y(t),x(t-h)) = R_yx(h) (h>0)
    CCV = permute(CCV(:,:,2:end),[2,1,3]);  
    lags = -(1:lagmax);
    cplxsin = exp(-1j * lags' * w);
    CSD = CSD + reshape(CCV,N^2,lagmax) * cplxsin;
    CSD = reshape(CSD,N,N,nfreq);
      
    % Coherence 
    % Extract diagonal terms for normalization
    SD = reshape(real(CSD(idx_coh)),N,nfreq); 
    SD(SD < 0) = NaN;
    if any(isnan(SD(:)))
        warning(['Some of the calculated power spectral densities have ',...
            'negative values. Consider increasing ''lagmax'' for ',...
            'more accurate estimation.'])  
    end
    CSD = abs(CSD).^2;
    for f = 1:nfreq
        nrm = 1./SD(:,f);
        CSD(:,:,f) = nrm .* CSD(:,:,f) .* (nrm.');
    end
    COH(:,:,:,j) = CSD;
    
end

VAR = zeros(N,M);
for j = 1:M
    VAR(:,j) = diag(COV(:,:,j));
end
    
if M == 1
    ACF = squeeze(ACF);
    COV = squeeze(COV);
    COH = squeeze(COH);
end

