function [outpars,control,equal,fixed,scale,skip] = ... 
    preproc_obs(M,N,p,r,pars,control,equal,fixed,scale)



% Dimension of original state vector x(t,j): r
% Dimension of expanded state vector X(t,j) = (x(t,j),...,x(t-p+1,j)): p*r

parname = {'A','C','Q','R','mu','Sigma','Pi','Z'};    
A = pars.A; C = pars.C; Q = pars.Q; R = pars.R; mu = pars.mu; 
Sigma = pars.Sigma; Pi = pars.Pi; Z = pars.Z;


% Subfunction to compare dimension attributes
function out = identical(x,y,drop1)
if ~(isvector(x) && isvector(y))
    out = false; return
end
if drop1 
    x = x(x>1); y = y(y>1);
end
out = numel(x) == numel(y) && all(x == y);
end

    
%-------------------------------------------------------------------------%    
%         Check dimensions of parameters and replicate as needed          %
%-------------------------------------------------------------------------%    


% Check A 
if ~identical(size(A),[r,r,p,M],false) % case where dimensions do not match
    % Check that non-singleton dimensions match     
    if identical(size(A),[r,r,p,M],true)   
        A = reshape(A,[r,r,p,M]); % if yes, reshape parameter properly
    else
        % If not, check whether (i) M > 1, (ii) equal.A is true, (iii) the
        % non-singleton dimensions of A match those of [r,r,p]. If all
        % these conditions are true, this means that the user has specified
        % only one set of matrices A (say, for regime 1) with the
        % understanding that this set of values should replicated M times
        % (across regimes). In this case, replicate as expected
        cond1 = M > 1;
        cond2 = exist('equal','var') && isstruct(equal) && ... 
            isfield(equal,'A') && equal.A; 
        cond3 = identical(size(A),[r,r,p],true);
        if cond1 && cond2 && cond3 
            A = repmat(reshape(A,[r,r,p]),[1,1,1,M]);
        else % other cases: wrong dimension specification
            error('''A'' must be an array of dimensions [r,r,p,M]');
        end
    end
end

% Check C
if ~identical(size(C),[N,r,M],false)
    if identical(size(C),[N,r,M],true)
        C = reshape(C,[N,r,M]);
    else       
        error(['''C'' must be a matrix of dimensions [N,r,M] where ', ...
            'N = size(y,1)']);
    end
end

% Check Q
if ~identical(size(Q),[r,r,M],false)
    if identical(size(Q),[r,r,M],true)
        Q = reshape(Q,[r,r,M]);
    else 
        cond1 = M > 1;
        cond2 = exist('equal','var') && isstruct(equal) && ... 
            isfield(equal,'Q') && equal.Q; 
        cond3 = identical(size(Q),[r,r],true);
        if cond1 && cond2 && cond3 
            Q = repmat(reshape(Q,[r,r]),[1,1,M]);
        else % other cases: wrong dimension specification
            error('''Q'' must be an array of dimensions [r,r,M]');            
        end
    end
end    

% Check R
if ~identical(size(R),[N,N],false)
    error(['''R'' must be a matrix of dimensions [N,N]',...
        'where N = size(y,1).']);
end

% Check mu
if ~identical(size(mu),[r,M],false)
    if identical(size(mu),[r,M],true)
        mu = reshape(mu,[r,M]);
    else
        cond1 = M > 1;
        cond2 = exist('equal','var') && isstruct(equal) && ... 
            isfield(equal,'mu') && equal.mu; 
        cond3 = identical(size(mu),r,true);
        if cond1 && cond2 && cond3
            mu = repmat(mu,[r,M]);
        else
            error('''mu'' must be a matrix of dimensions [r,M]')
        end
    end
end
        
% Check Sigma
if ~identical(size(Sigma),[r,r,M],false)
    if identical(size(Sigma),[r,r,M],true) 
        Sigma = reshape(Sigma,[r,r,M]);
    else 
        cond1 = M > 1;
        cond2 = exist('equal','var') && isstruct(equal) && ... 
            isfield(equal,'Sigma') && equal.Sigma; 
        cond3 = identical(size(Sigma),[r,r],true); 
        if cond1 && cond2 && cond3 
            Sigma = repmat(reshape(Sigma,[r,r]),[1,1,M]);
        else 
            error('''Sigma'' must be an array of dimensions [r,r,M]');            
        end
    end
end    

% Check Pi
if ~identical(size(Pi),[M,1],false)
    if identical(size(Pi),M,true)
        Pi = reshape(Pi,[M,1]);
    else
        error('''Pi'' must be a vector of length M.');
    end
end    

% Check Z
if ~identical(size(Z),[M,M],false)
    error('''Z'' must be a matrix of dimensions [M,M].');
end

% Group initial estimates in a list
par = {A,C,Q,R,mu,Sigma,Pi,Z};



%-------------------------------------------------------------------------%    
%                     Optional control parameters                         %
%-------------------------------------------------------------------------%    

% Default values
control0 = struct('eps',1e-7,'ItrNo',1000,'beta0',1,'betarate',1,...
    'safe',false,'abstol',1e-8,'reltol',1e-8,'verbose',true);

% Override with user-specified control parameters if any 
if isstruct(control)
    fnames = fieldnames(control0);
    for i = 1:numel(fnames)
        name = fnames{i};
        if isfield(control,name)
            control0.(name) = control.(name);
        end
    end
end
control = control0;
vals = struct2cell(control);
[eps,ItrNo,beta0,betarate,safe,abstol,reltol,verbose] = vals{:};
if (abstol <= 0 || reltol <= 0) 
    error('Arguments ''abstol'' and ''retol'' must be stricly positive.')
end
if (beta0 <= 0 || beta0 > 1)
    error('Argument ''beta0'' must be in (0,1].')
end
if (betarate < 1)
    error('Argument ''betarate'' must be in [1,Inf).')
end




%-------------------------------------------------------------------------%    
%                Equality constraints on model parameters                 %
%-------------------------------------------------------------------------%    


% Default constraints
equal0 = struct('A',false,'C',false,'Q',false,'mu',true,'Sigma',true);

% Override with user-specified constraints if any
if isstruct(equal)
    fnames = fieldnames(equal0);
    for i = 1:numel(fnames)
        name = fnames{i};
        if isfield(equal,name)
            equal0.(name) = equal.(name);
        end
    end
end
equal = equal0;



%-------------------------------------------------------------------------%    
%                 Check symmetry of Q, R, and Sigma                       %
%-------------------------------------------------------------------------%    


for j=1:M
    if ~issymmetric(Q(:,:,j))
        error('Q(:,:,%d) is not symmetric.',j)
    end    
    if ~issymmetric(Sigma(:,:,j))
        error('Sigma(:,:,%d) is not symmetric.',j)
    end
end

if ~issymmetric(R)
    error('''R'' is not symmetric.')
end



%-------------------------------------------------------------------------%    
%         Check box and unit sum constraints on Pi and Z                  %
%-------------------------------------------------------------------------%    

if ~all(Pi >= 0 & Pi <= 1)
    error('Values of ''Pi'' not all in [0,1]')
end

if ~all(Z(:) >= 0 & Z(:) <= 1)
    error('Values of ''Z'' not all in [0,1]')
end

if ~(abs(sum(Pi)-1) <= eps)
    error('Values of ''Pi'' do not sum up to 1')
end

if ~all(abs(sum(Z,2)-1) <= eps)
    error('Rows of ''Z'' do not all sum up to 1')
end



%-------------------------------------------------------------------------%    
%                  Check scaling arguments for A and C                    %
%-------------------------------------------------------------------------%    


scale0 = struct('A',.999,'C',[]);
if isstruct(scale)
    fnames = fieldnames(scale0);
    for i = 1:numel(fnames)
        name = fnames{i};
        if isfield(scale,name)
            scale0.(name) = scale.(name);
        end
    end
end
scale = scale0;

if scale.A <= 0 || scale.A >= 1
    error('''scale.A'' must be in (0,1).')
end

if scale.C <= 0
    error('''scale.C'' must be positive.')
end



%-------------------------------------------------------------------------%    
%                  Check fixed coefficients constraints                   %
%-------------------------------------------------------------------------%    



% * Check formatting of fixed coefficients constraints 
% * Check compatibility between fixed coefficient constraints and equality
% constraints 
% * Expand fixed coefficients specification  w.r.t. equality constraints as
% needed
% * Check that for covariance matrices, either all coefficients are
% fixed, all are free, or a diagonal structure is specified
% * Check that for covariance matrices, fixed coefficients constraints are
% compatible with symmetry

% Logical flags for skipping parameter update if all its coefficients are fixed
skip = struct('A',false,'C',false,'Q',false,'R',false,'mu',false,...
    'Sigma',false,'Pi',false,'Z',false);
z = zeros(0,2);
fixed0 = struct('A',z,'C',z,'Q',z,'R',z,'mu',z,'Sigma',z,'Pi',z,'Z',z);

for i = 1:8
    name = parname{i};
    skip.(name) = false;
    if ~isfield(fixed,name) || isempty(fixed.(name))       
        continue
    end
    fixed_i = fixed.(name); 

    % Check field dimensions. If active equality constraint and fixed
    % coefficients specified only for one regime, replicate accordingly
    size_i = size(par{i});
    if ~identical(size(fixed_i),size_i,false)
        if identical(size(fixed_i),size_i,true)
            fixed_i = reshape(fixed_i,size_i);
        elseif M > 1 && equal.(name) && ...
                identical(size(fixed_i),size_i(1:end-1),true)
            fixed_i = repmat(reshape(fixed_i,size_i(1:end-1)),...
                [repelem(1,numel(size_i)-1),M]);
        else
            error(['''fixed.%s'' must be an array of same dimension ', ...
                'as ''%s'''],name,name)
        end
    end
             
    % If active equality constraint and fixed coefficients specified beyond
    % 1st regime, check compatibility
    if  M>1 && isfield(equal,name) && equal.(name)
        fixed_i = reshape(fixed_i,[],M);
        rowmin = min(fixed_i,[],2,'omitnan');
        rowmax = max(fixed_i,[],2,'omitnan');
        testnan = all(isnan(rowmin) == isnan(rowmax));
        testnum = all(rmmissing(rowmin) == rmmissing(rowmax));
        if ~(testnan && testnum)
            error(['Equality contraints and fixed coefficient', ...
                'constraints not compatible for ''%s'''],name)
        end
    end
   
    % Represent fixed coefficients as matrix with 2 columns, 1st with
    % indices of fixed coefficients, 2nd with their values
    idx = find(~isnan(fixed_i));
    fixed0.(name) = [idx,fixed_i(idx)];
    
    % If all coefficients are fixed, set 'skip' to true (else false)
    skip.(name) = all(~isnan(fixed_i(:)));
        
end
fixed = fixed0;



% Check fixed coefficients in A %@@@@@@@@
if ~skip.A && ~isempty(fixed.A)
    Atmp = NaN(N*N*p,M);
    Atmp(fixed.A(:,1)) = fixed.A(:,2);
    test1 = all(~isnan(Atmp));
    test2 = all(isnan(Atmp) | Atmp == 0);
    assert(all(test1 | test2), ['For each regime j, if A(:,:,:,j) is ',... 
        'not entirely fixed, then any fixed coefficient must be zero.'])
end


% For covariance matrices Q, R, and Sigma, check (i) compatibility of fixed
% coefficient constraints and symmetry and (ii) that either all
% coefficients are fixed, all are free, or a diagonal structure is
% specified
msg1 = ['Fixed coefficients for ''%s'' incompatible with ',...
                'symmetry constraint'];
msg2 = ['Coefficients of ''%s'' must either be all fixed, all free, ', ...
            'or have a diagonal structure (free on diagonal, zero outside)'];
if ~isempty(fixed.R)
    temp = zeros(N);
    temp(fixed.R(:,1)) = fixed.R(:,2);
    if ~issymmetric(temp)
        error(msg1,'R')
    end
    temp = NaN(N);
    temp(fixed.R(:,1)) = fixed.R(:,2);
    test1 = all(isnan(diag(temp)));  % are all diagonal coefficients free?
    offdiag = (eye(N) == 0);
    test2 = all(temp(offdiag) == 0); % are all off-diagonal coefficients zero?   
    if ~(skip.R || (test1 && test2)) 
        error(msg2,'R')
    end
end
fnames = {'Q','Sigma'};
for i = 1:2
    name = fnames{i};
    if isempty(fixed.(name))
        continue
    end
    % Test symmetry 
    temp = zeros(r,r,M);
    temp(fixed.(name)(:,1)) = fixed.(name)(:,2);
    test = arrayfun(@(j) issymmetric(temp(:,:,j)),1:M);
    if ~all(test)
        error(msg1,name)
    end
    if skip.(name) 
        continue
    end
    % Test for diagonal structure
    temp = NaN(r,r,M);
    temp(fixed.(name)(:,1)) = fixed.(name)(:,2);
    offdiag = find(repmat(eye(r),1,1,M) == 0);
    test = all(isnan(temp(~offdiag)) & temp(offdiag) == 0); 
    if ~test
        error(msg2,name)
    end  
end      

% Check compatibility of fixed coefficient constraints with box constraints
% and unit sum constraints for Pi and Z
if ~isempty(fixed.Pi)
    if ~skip.Pi
        error('Coefficients of ''Pi'' must be either all fixed or all free')
    end
    vals = fixed.Pi(:,2);
    if any(vals < 0) || any(vals > 1)
        error('Some fixed coefficients of ''Pi'' are outside of [0,1]')
    end
    if abs(sum(vals)-1) > eps
        error('Fixed coefficients of ''Pi'' do not sum up to 1.')
    end
end

if ~isempty(fixed.Z)
    if ~all(fixed.Z(:,2) >= 0 & fixed.Z(:,2) <= 1)
        error('Some fixed coefficients of ''Z'' are outside of [0,1]')
    end
    Ztmp = NaN(M);
    Ztmp(fixed.Z(:,1)) = fixed.Z(:,2);
    test = sum(isnan(Ztmp),2);
    if ~all(test == 0 | test == M)
        error(['For each row of ''Z'', coefficients  must either be ', ...
            'all fixed or all free'])
    end
    Ztmp = rmmissing(Ztmp);
    test = (abs(sum(Ztmp,2) - 1) <= eps);
    if ~all(test)
        error('Rows of fixed coefficients in ''Z'' do not sum up to 1')
    end
end

% Check compatibility between arguments 'fixed' and 'scale' for C
% If 'fixed.C' and 'scale.C' are both specified, the fixed coefficient
% values must be zero for the two types of constraints to be compatible
if ~isempty(fixed.C) && ~isempty(scale.C) && ~all(fixed.C(:,2) == 0)
        error(['If both ''fixed.C'' and ''scale.C'' are specified,'...
            'all fixed coefficients in C must be zero.']);
end

% Check compatibility between arguments 'fixed' and 'equal' for A 
% Fixed coefficient constraints and equality constraints cannot be
% specified at the same time
if equal.A && ~isempty(fixed.A) 
        error(['Simultaneous specification of fixed coefficient constraints', ...
            ' and equality constraints for ''A'' currently not supported']);
end



%-------------------------------------------------------------------------%    
%       Check fixed coefficients constraints, equality constraints        % 
%             and scaling constraints on initial estimates                %
%-------------------------------------------------------------------------%    


for i=1:8
    name = parname{i};
    param = par{i}(:);   
    
    if ~isempty(fixed.(name))
        fixed_i = fixed.(name);
        if any(param(fixed_i(:,1)) ~= fixed_i(:,2))
            error(['Initial estimate ''%s'' incompatible with fixed ',...
                'coefficient constraints.'],name)
        end 
    end
    
    if isfield(equal,name) && equal.(name)
        param = reshape(param,[],M);
        rowmin = min(param,[],2);
        rowmax = max(param,[],2);
        if any(rowmin ~= rowmax)
            error(['Initial estimate ''%s'' incompatible with equality ', ...
                'contraints.'],name)
        end
    end
end

% Check eigenvalues of A
for j = 1:M
    if p == 1
        A_j = squeeze(A(:,:,:,j));
    else
        A_j = diag(ones((p-1)*r,1),-r);
        A_j(1:r,:) = reshape(A(:,:,:,j),[r,p*r]);
    end
    check = all(abs(eig(A_j)) <= scale.A);
    if ~check
        error(['Initial estimate of A(:,:,:,%d) incompatible with ',...
            'eigenvalue constraint (<= %f).'],j,scale.A)
    end   
end

% Check scaling of C
if ~isempty(scale.C)
    for j = 1:M
        nrm_C = sqrt(sum(C(:,:,j).^2));
        if any(nrm_C > (1+reltol) * scale.C)
            msg = ['Initial estimate of ''C''(%d) incompatible with ',...
                'scaling constraint (column norms = %f).'];
            error(msg,j,scale.C)
        end
    end
end

clear param name rowmin rowmax



%-------------------------------------------------------------------------%    
%                        Reshape initial estimates                        %
%-------------------------------------------------------------------------%    


A = reshape(A,[r,p*r,M]);

    

%-------------------------------------------------------------------------%    
%                       Regularize Q, R, and Sigma                        %
%-------------------------------------------------------------------------%    


for j=1:M
    Q(:,:,j) = regfun(Q(:,:,j),abstol,reltol);
end
if ~isempty(fixed.Q)
    Q(fixed.Q(:,1)) = fixed.Q(:,2);
end

R = regfun(R,abstol,reltol);
if ~isempty(fixed.R)
    R(fixed.R(:,1)) = fixed.R(:,2);
end

for j=1:M
    Sigma(:,:,j) = regfun(Sigma(:,:,j),abstol,reltol);
end
if ~isempty(fixed.Sigma)
    Sigma(fixed.Sigma(:,1)) = fixed.Sigma(:,2);
end



outpars = struct('A', A, 'C', C, 'Q', Q, 'R', R, 'mu', mu, ...
    'Sigma', Sigma, 'Pi', Pi, 'Z', Z);





end