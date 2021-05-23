function [ACF,CCV] = get_crosscovariance(pars,lagmax)

%-------------------------------------------------------------------------%
%
% Purpose:  calculate stationary cross-covariance function in switching state-space model 
%           y(t) = C(S(t)) x(t) + w(t)
%           x(t) = A(1,S(t)) x(t-1) + ... + A(p,S(t)) x(t-p) + v(t)
%
% Usage:    [ACF,CCV] = get_crosscovariance(pars,lagmax)
%
% Inputs:   pars - struct with fields 
%               A - state transition matrices ([r,r,p,M])
%               C - observation matrices ([N,r], [N,r,M], or [])
%               Q - state noise covariance matrices ([N,N,M]) 
%               R - observation noise covariance matrix ([N,N] or [])
%           lagmax - maximum lag for autocorrelation function (optional, 
%               default = 5*p) 
%
% Outputs:  ACF - autocorrelation functions ([N,lagmax+1])
%           CCV - cross-covariance functions ([N,N,lagmax+1,M])
%
%-------------------------------------------------------------------------%

narginchk(1,2);

% Model parameters
A = pars.A;
if isfield(pars,'C')
    C = pars.C;
else
    C = [];
end
Q = pars.Q;
if isfield(pars,'R')
    R = pars.R;
else
    R = [];
end

% Model dimensions
[r,~,p,M] = size(A);

if ~exist('lagmax','var') || isempty(lagmax)
    lagmax = 5*p;
end

lagmax = round(lagmax);
assert(lagmax >= 0)

% Check arguments C and R
if ~isempty(C) || ~isempty(R)
    assert(~isempty(C) && ~isempty(R))
    assert(ismatrix(R) && size(R,1) == size(R,2))
    assert(size(C,1) == size(R,1))
    N = size(C,1);
    if ismatrix(C)
        C = repmat(C,1,1,M);
    end
else
    N = r;
end

    

% Initialize various quantities
ACF = NaN(N,lagmax+1,M);   % auto-correlation diag(cor(x(t),x(t-l)|S(t)=j)
CCV = NaN(N,N,lagmax+1,M); % cross-correlation (cov(x(t),x(t-l)|S(t)=j)
Abig = zeros(p*r); % container for A 
if p > 1
    Abig(r+1:end,1:end-r) = eye((p-1)*r);
end
Qbig = zeros(p*r); % container for Q
idx_acf = (repmat(eye(N),1,1,lagmax+1) == 1);

% Calculations
for j = 1:M
    Aj = A(:,:,:,j); 
    Abig(1:r,:) = reshape(Aj,r,p*r);
    Qbig(1:r,1:r) = Q(:,:,j);
    if all(Aj(:) == 0)
        Vbig = Qbig;
    else
        eigA = abs(eig(Abig));
        if any(eigA >= 1) 
            continue
        elseif min(eigA) <= 1e-10 * max(eigA) % case: Abig numerically singular
            Vbig = get_covariance_aux(Aj,Q(:,:,j)); % Cov(X(t)|S(t)=j)
        else % case: Abig full rank
            B = (Abig.' * Abig)\(Abig.');
            Vbig = sylvester(B,-Abig.',B*Qbig); 
        end
    end
    Vbig = 0.5 * (Vbig + Vbig.');
    
    if isempty(C)
        COV = Vbig(1:r,1:r);
    else
        COV = (C(:,:,j) * Vbig(1:r,1:r) * C(:,:,j).') + R;
    end
    COV = 0.5 * (COV + COV');
    
    if lagmax == 0
        CCV(:,:,:,j) = COV;
        ACF(:,:,j) = ones(N,1);
        continue
    end
    
    CCV_tmp = Vbig; % Cov(X(t),X(t-l)|S(t)=j)
    CCVj = zeros(r,r,lagmax+1); % Cov(x(t),x(t-l)|S(t)=j)
    CCVj(:,:,1) = Vbig(1:r,1:r); 
    A_up = Abig(1:r,:);
    for l = 1:lagmax
        B = A_up * CCV_tmp;
        CCV_tmp(r+1:end,:) = CCV_tmp(1:end-r,:);
        CCV_tmp(1:r,:) = B;
        CCVj(:,:,l+1) = CCV_tmp(1:r,1:r);
    end
    if ~isempty(C) % Cov(y(t),y(t-l)|S(t)=j)
        CCVj = C(:,:,j) * reshape(CCVj,r,r*(lagmax+1));
        CCVj = reshape(CCVj,N,r,lagmax+1);
        CCVj = permute(CCVj,[1,3,2]);
        CCVj = reshape(CCVj,N*(lagmax+1),r) * C(:,:,j).';
        CCVj = reshape(CCVj,N,lagmax+1,N);
        CCVj = permute(CCVj,[1,3,2]);
        CCVj(:,:,1) = COV;
    end
    CCV(:,:,:,j) = CCVj;
    
    % Autocorrelation 
    ACF(:,:,j) = reshape(CCVj(idx_acf),N,lagmax+1);
    ACF(:,:,j) = ACF(:,:,j) ./ diag(COV);
end
 



