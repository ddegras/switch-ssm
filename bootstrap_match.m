function outboot = bootstrap_match(parsboot,pars,target)

%-------------------------------------------------------------------------%
%
% Title:    Match bootstrap replicates to maximum likelihood estimate (MLE)
%
% Purpose:  Permute the regime-specific components of each bootstrap  
%           estimate so as to best match the components of the MLE 
%           according to a certain target 
%  
% Usage:    outboot = match_bootstrap(parsboot,pars,target)
%
% Inputs:   parsboot - structure of bootstrapped parameter estimates,    
%               typically the result of a call to bootstrap_dyn,
%               bootstrap_var, or bootstrap_obs
%           pars - structure of MLE, typically the result of a call to 
%               switch_dyn, switch_var, or switch_obs
%           target - parameter(s) or functions thereof to base the matching 
%               on: 'A', 'C', 'AQ' (both A and Q), 'COV', and 'COR'. The 
%               last two are the stationary covariance and correlation of 
%               the observed time series in each regime.    
%                   
%                   
% Outputs:  outboot - structure of matched bootstrap replicates with the 
%               same fields as 'parsboot':  
%               'A': Bootstrap distribution of A (size rxrxpxMxB)
%               'C': Bootstrap distribution of C if available (size NxrxB) 
%               'Q': Bootstrap distribution of Q (size rxrxMxB)
%               'R': Bootstrap distribution of R (size NxNxB)
%               'mu': Bootstrap distribution of mu (size rxMxB) 
%               'Sigma': Bootstrap distribution of Sigma (size rxrxMxB)
%               'Pi': Bootstrap distribution of Pi (size MxB)
%               'Z': Bootstrap distribution of Z (size MxMxB)
%                    
% Author:   David Degras, david.degras@umb.edu
%           University of Massachusetts Boston
%
%-------------------------------------------------------------------------%

narginchk(2,3)

% Initialize target if needed
if nargin == 2
    target = 'COV';
end

% Check validity of argument 'target'
trgt_list = {'A','C','AQ','COV','COR'};
assert(ismember(target,trgt_list));

% Argument dimensions
[~,~,~,M,B] = size(parsboot.A);
N = [];
if isfield(pars,'C')
    N = size(pars.C,1);
end

% Initialize output
outboot = parsboot;

% Trivial case
if M == 1
    return
end

% Select and reshape target
trgt_val = [];
switch target
    case 'A'
        trgt_val = pars.A;
    case 'C'
        if ~isfield(pars,'C')
            error('Field ''C'' missing in argument ''pars''')
        end
        N = size(pars.C,1);
        if size(pars.C,3) ~= M
            error(['Number of matrices ''C'' (%d) incompatible ', ...
                'with number of regimes (%d)'],size(pars.C,3),M)
        end
        trgt_val = zeros(N,N,M);
        for j = 1:M
            trgt_val(:,:,j) = pars.C(:,:,j) * pars.C(:,:,j)';
        end
    case 'AQ'
        trgt_val = [reshape(pars.A,[],M) ; reshape(pars.Q,[],M)];
    case 'COV'
        [~,~,trgt_val] = get_covariance(pars,0,0);
    case 'COR'
        [~,~,COV,VAR] = get_covariance(pars,0,0);
        for j = 1:M
            try 
                trgt_val(:,:,j) = corrcov(COV(:,:,j) + COV(:,:,j)');
            catch 
                SDj = sqrt(VAR(:,j));
                SDj(SDj == 0) = 1;
                trgt_val(:,:,j) = diag(1 ./ SDj) * COV(:,:,j) * diag(1 ./ SDj);
            end
        end
end            
trgt_val = reshape(trgt_val,[],M);

% Determine type of model
model = [];
if ~isfield(pars,'C') || isempty(pars.C)
    model = 'var';
elseif ismatrix(pars.C)
    model = 'dyn';
else
    model = 'obs';
end

% Match bootstrapped estimates to target
P = perms(1:M);
nperm = size(P,1);
parb = struct('A',[], 'C',[], 'Q',[], 'R',[]);
dist = zeros(1,nperm);
for b = 1:B
    % Reshape bootstrap estimate
    boot_val = [];
    switch target
        case 'A'
            boot_val = parsboot.A(:,:,:,:,b);
        case 'C'            
            boot_val = zeros(N,N,M);
            for j = 1:M
                boot_val(:,:,j) = parsboot.C(:,:,j,b) * parsboot.C(:,:,j,b)';
            end
        case 'AQ'
            boot_val = [reshape(parsboot.A(:,:,:,:,b),[],M) ; ...
                reshape(parsboot.Q(:,:,:,b),[],M)];
        case 'COV'
            parb.A = parsboot.A(:,:,:,:,b);
            parb.Q = parsboot.Q(:,:,:,b);
            parb.R = parsboot.R(:,:,b);
            if strcmp(model,'dyn')
                parb.C = parsboot.C(:,:,b);
            end            
            if strcmp(model,'obs')
                pars.C = parsboot.C(:,:,:,b);
            end
            [~,~,boot_val] = get_covariance(parb,0,0);
        case 'COR'
            parb.A = parsboot.A(:,:,:,:,b);
            parb.Q = parsboot.Q(:,:,:,b);
            parb.R = parsboot.R(:,:,b);
            if strcmp(model,'dyn')
                parb.C = parsboot.C(:,:,b);
            end            
            if strcmp(model,'obs')
                pars.C = parsboot.C(:,:,:,b);
            end
            [~,~,COV] = get_covariance(parb,0,0);
            for j = 1:M
                try 
                    boot_val(:,:,j) = corrcov(COV(:,:,j) + COV(:,:,j)');
                catch 
                    SDj = sqrt(VAR(:,j));
                    SDj(SDj == 0) = 1;
                    boot_val(:,:,j) = diag(1 ./ SDj) * COV(:,:,j) * diag(1 ./ SDj);
                end
            end
    end            
    boot_val = reshape(boot_val,[],M);  
    
    % Calculate distances between bootstrap estimates and targets
    for i = 1:nperm
        dist(i) = norm(trgt_val-boot_val(:,P(i,:)),1);
    end
    [~,i] = min(dist);
    if ~isequal(P(i,:),1:M)
        outboot.A(:,:,:,:,b) = parsboot.A(:,:,:,P(i,:),b);
        outboot.Q(:,:,:,b) = parsboot.Q(:,:,P(i,:),b);
        if strcmp(model,'obs')
            outboot.C(:,:,:,b) = parsboot.C(:,:,P(i,:),b);
        end
        outboot.mu(:,:,b) = parsboot.mu(:,P(i,:),b);
        outboot.Sigma(:,:,:,b) = parsboot.Sigma(:,:,P(i,:),b);
        outboot.Pi(:,b) = parsboot.Pi(P(i,:),b);
        outboot.Z(:,:,b) = parsboot.Z(P(i,:),P(i,:),b);
    end
end


        
        
        