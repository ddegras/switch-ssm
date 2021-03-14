function pars = reestimate_dyn(y,M,p,r,S,control,equal,fixed,scale)


%-------------------------------------------------------------------------%
%                        Preprocessing                                    %
%-------------------------------------------------------------------------%

% Check number of inputs
narginchk(5,9);

% Data dimensions
[N,T] = size(y);

% Data centering
y = y - mean(y,2);

% Check dimensions of input arguments
assert(r<=N, 'Dimension of state vector ''r'' exceeds dimension of time series')
assert(M<=T, 'Number of regimes ''M'' exceeds length of time series')
assert(r<=T, 'Dimension of state vector ''r'' exceeds length of time series')
assert(p<=T, 'VAR order ''p'' exceeds length of time series')

% Disable warnings for (nearly) singular matrices
warning('off','MATLAB:singularMatrix'); 
warning('off','MATLAB:nearlySingularMatrix');


%-------------------------------------------------------------------------%
%     Set optional arguments to default values if not specified           %
%-------------------------------------------------------------------------%


control0 = struct('abstol',1e-8,'reltol',1e-4); 
if exist('control','var') && isstruct(control) 
    fname = fieldnames(control0);
    for i = 1:numel(fname)
        if isfield(control,fname{i})
            control0.(fname{i}) = control.(fname{i});
        end
    end
end
control = control0;
abstol = control.abstol;
reltol = control.reltol;

equal0 = struct('A',false,'C',false,'Q',false);
if exist('equal','var') && isstruct(equal)
    fname = fieldnames(equal0);
    for i = 1:3
        if isfield(equal,fname{i})
            equal0.(fname{i}) = equal.(fname{i});
        end
    end
end
equal = equal0;   

fixed0 = struct('A',[],'C',[],'Q',[],'R',[],'mu',[],'Sigma',[],...
    'Pi',[],'Z',[]);
if exist('fixed','var') &&isstruct(fixed)    
    fname = fieldnames(fixed0);
    for i = 1:8
        if isfield(fixed,fname{i}) 
            fixed0.(fname{i}) = fixed.(fname{i});
        end
    end
end
fixed = fixed0;   

scale0 = struct('A',.999,'C',[]);
if exist('scale','var') &&isstruct(scale)
    fname = fieldnames(scale0);
    for i = 1:2 
        if isfield(scale,fname{i})
            scale0.(fname{i}) = scale.(fname{i});
        end
    end
end
scale = scale0;



%-------------------------------------------------------------------------%
%         Check whether some parameters are entirely fixed                %
%-------------------------------------------------------------------------%

skip = struct();
fname = fieldnames(fixed);
for i = 1:8
    name = fname{i};
    skip.(name) = ~isempty(fixed.(name)) && all(~isnan(fixed.(name)(:)));
end



%-------------------------------------------------------------------------%
%                    Estimate observation matrix C                        %
%-------------------------------------------------------------------------%


if skip.C
    Chat = fixed.C;
    xhat = (Chat'*Chat)\(Chat'*y);
else
    % SVD
    [U,D,V] = svd(y,'econ');

    % Unconstrained estimate
    Chat = U(:,1:r); 

    % Estimates of state vectors x(t)
    xhat = D(1:r,1:r) * V(:,1:r)'; 
end

% Rescale C and x if required
if ~isempty(scale.C)
    Chat = Chat * scale.C;
    xhat = xhat / scale.C;
end

% Re-estimate C under fixed coefficient constraints if required
if ~isempty(fixed.C) && ~skip.C
    idx = find(~isnan(fixed.C));
    fixed_C = [idx,fixed.C(idx)];
    [Chat,err] = PG_C(Chat,xhat*xhat',xhat*y',eye(N),scale.C,fixed_C);
    if err
        error(['Cannot find estimate of C satisfying both fixed',...
            ' coefficient constraints (''fixed.C'') and scale constraints',...
            ' (''scale.C''). Please check that the two constraints',...
            '  are mutually compatible.'])
    end
end



%-------------------------------------------------------------------------%
%                 Estimate observation error covariance R                 %
%-------------------------------------------------------------------------%


if skip.R 
    Rhat = fixed.R;
else
    % Estimate R from the residuals y - Cx
    % Since the empirical variance matrix of the residuals is rank-deficient 
    % (rank = N-r) only use its diagonal to make Rhat (likely) positive definite
    Rhat = diag(var(y-Chat*xhat,0,2));

    % Further regularize (to improve conditioning) if needed
    Rhat = regfun(Rhat,abstol,reltol);

    % Apply fixed coefficient constraints 
    if ~isempty(fixed.R) 
        idx = ~isnan(fixed.R);
        Rhat(idx) = fixed.R(idx);
    end
end


%-------------------------------------------------------------------------%
%            Estimate initial mean mu and covariance Sigma                %
%-------------------------------------------------------------------------%



t0 = min([10,5*p,T]); % number of time points used in estimation  

if skip.mu
    muhat = fixed.mu;
else
    muhat = repmat(mean(xhat(:,1:t0),2),[1,M]); 
    if ~isempty(fixed.mu) 
        % If fixed constraints are specified only on mu(1) and equality
        % constraints are active, replicate fixed constraints as required
        if isvector(fixed.mu) && isfield(equal,'mu') && equal.mu
            fixed.mu = repmat(fixed.mu(:),1,M);
        end
        idx = ~isnan(fixed.mu);
        muhat(idx) = fixed.mu(idx);
    end
end

if skip.Sigma
    Sigmahat = fixed.Sigma;
else
    Sigmahat = diag(var(xhat(:,1:t0),0,2));
    Sigmahat = regfun(Sigmahat,abstol,reltol);
    Sigmahat = repmat(Sigmahat,[1,1,M]);
    if ~isempty(fixed.Sigma)
        if ismatrix(fixed.Sigma) && isfield(equal,'Sigma') && equal.Sigma
            fixed.Sigma = repmat(fixed.Sigma(:),[1,1,M]);
        end
        idx = ~isnan(fixed.Sigma);
        Sigmahat(idx) = fixed.Sigma(idx);
    end
end



%-------------------------------------------------------------------------%
%                         Estimate VAR matrix A                           %
%-------------------------------------------------------------------------%


% The estimator of matrix A(j) is obtained by minimizing
% trace(A(j) sum(t:S(t)=j) Xhat(t) Xhat(t)' A(j)' - 2 A(j)' sum(t:S(t)=j)
% Xhat(t) Xhat(t)') where Xhat(t) = (xhat(t),...,xhat(t-p+1))

if skip.A
    Ahat = fixed.A;
else
    Ahat = zeros(r,p*r,M);
    Y = xhat(:,p+1:T);  % response matrix
    X = zeros(p*r,T-p); % predictor matrix
    for lag = 1:p
        indx = (lag-1)*r+1:lag*r;
        indt = p-lag+1:T-lag;
        X(indx,:) = xhat(:,indt);
    end
    XX = zeros(p*r,p*r,M);  % sum(t:S(t)=j) Xhat(t) Xhat(t)'
    YX = zeros(r,p*r,M);    % sum(t:S(t)=j) xhat(t) Xhat(t)'
    YY = zeros(r,r,M);      % sum(t:S(t)=j) xhat(t) xhat(t)'
    fixed_A = cell(M,1);


    % Estimate A
    for j = 1:M

        % Fixed coefficients in A(j), two-column format
        if ~isempty(fixed.A)
            Atmp = fixed.A(:,:,:,j);
            idx = find(~isnan(Atmp));
            fixed_A{j} = [idx,Atmp(idx)];
        end

        % Set up autoregression
        if equal.A
            idx = 1:T-p;
        else
            idx = (S(p+1:T) == j);
        end
        XXj = X(:,idx) * X(:,idx)';
        XX(:,:,j) = XXj;
        YXj = Y(:,idx) * X(:,idx)';
        YX(:,:,j) = YXj;
        YY(:,:,j) = Y(:,idx) * Y(:,idx)';

        % Unconstrained estimate of A(j)
        if isempty(fixed_A{j})
            A_j = YXj/XXj; 
            if any(isnan(A_j(:)) | isinf(A_j(:)))
                A_j = YXj * pinv(XXj);
            end

        % Constrained estimate of A(j)
        else
            isfixed = fixed_A{j}(:,1);
            isfree = setdiff(1:p*r^2,isfixed);
            % Vectorize the problem and remove rows associated with fixed
            % coefficients of A(j)
            mat = kron(XXj,eye(r));
            vec = reshape(YXj,p*r^2,1);
            A_j = zeros(p*r^2,1);
            A_j(isfree) = mat(isfree,isfree)\vec(isfree);
            if any(isnan(A_j)|isinf(A_j))
                A_j(isfree) = pinv(mat(isfree,isfree)) * vec(isfree);
            end
            A_j(isfixed) = fixed_A{j}(:,2);        
            A_j = reshape(A_j,r,p*r);
        end    

        if equal.A
            Ahat = repmat(A_j,1,1,M);
            break
        end
        Ahat(:,:,j) = A_j;

    end

    % Check that A define a stationary process and regularize if needed
    Abig = diag(ones((p-1)*r,1),-r);
    for j = 1:M
        Abig(1:r,:) = Ahat(:,:,j);
        eigval = abs(eig(Abig));
        if any(eigval > scale.A)
            % Easy regularization: no fixed coefficients or all fixed
            % coefficients are zero. Use algebraic properties of eigenvalues
            % and eigenvectors of Abig
            if isempty(fixed_A{j}) || all(fixed_A{j}(:,2) == 0)
                A_j = reshape(Ahat(:,:,j),r,r,p);
                c = .999 * scale.A / max(eigval);
                for l = 1:p
                    A_j(:,:,l) = c^l * A_j(:,:,l);
                end
                Ahat(:,:,j) = reshape(A_j,r,p*r);

            % Standard regularization: nonzero fixed coefficients. 
            % Use projected gradient method
            else
                A_j = Ahat(:,:,j);
                if equal.A
                    YXj = sum(YX,3);
                    XXj = sum(XX,3);
                else
                    YXj = YX(:,:,j);
                    XXj = XX(:,:,j);
                end
                [A_j,err] = PG_A(A_j,YXj,XXj,eye(r),scale.A,fixed_A{j},100);
                Ahat(:,:,j) = A_j;
                if err
                    error(['Cannot find estimate of A%d satisfying both the',...
                    ' fixed coefficient constraints (''fixed.A'') and',...
                    ' eigenvalue constraints (''scale.A'').\nPlease check',...
                    ' that the constraints are mutually compatible and',...
                    ' consider modifying/removing some constraints.'],j);
                end
            end
        end
        if equal.A 
            Ahat = repmat(Ahat(:,:,1),[1,1,M]);
            break
        end
    end
end


%-------------------------------------------------------------------------%
%                 Estimate innovation variance matrix Q                   %
%-------------------------------------------------------------------------%


if skip.Q
    Qhat = fixed.Q;
else
    Qhat = zeros(r,r,M);
    groupsize = arrayfun(@(j)sum(S(p+1:T)==j),1:M)';

    % Unconstrained estimate  
    for j = 1:M
        if groupsize(j) > 0
            % Residual covariance
            XXj = XX(:,:,j);
            YXj = YX(:,:,j);
            YYj = YY(:,:,j);
            A_j = Ahat(:,:,j);
            Q_j =  YYj - YXj * A_j' - A_j * YXj' + A_j * XXj * A_j'; 
            Qhat(:,:,j) = diag(diag(Q_j)/groupsize(j));
        elseif any(S == j)
            Qhat(:,:,j) = diag(var(xhat(:,S == j),1,2));
        end
    end
    if equal.Q
        Qhat = reshape(Qhat,r^2,M) * (groupsize / sum(groupsize));
        Qhat = repmat(reshape(Qhat,r,r),1,1,M);
    end

    % Apply fixed coefficient constraints and regularize Q 
    if ~isempty(fixed.Q)
        idx = ~isnan(fixed.Q);
        Qhat(idx) = fixed.Q(idx);
    end
    for j = 1:M
        Qhat(:,:,j) = regfun(Qhat(:,:,j),abstol,reltol);
    end
    if ~isempty(fixed.Q)
        Qhat(idx) = fixed.Q(idx);
    end
end

% Reshape A
Ahat = reshape(Ahat,[r,r,p,M]);



%-------------------------------------------------------------------------%
%    Estimate initial probabilities Pi and transition probabilities Z     %
%            of Markov process {S(t):t=1:T} (regime switching)            %
%-------------------------------------------------------------------------%


if skip.Pi
    Pihat = fixed.Pi;
else
    Pihat = zeros(M,1);
    Pihat(S(1)) = 1;
    if ~isempty(fixed.Pi)
        idx = ~isnan(fixed.Pi);
        Pihat(idx) = fixed.Pi(idx);
    end
end

if skip.Z
    Zhat = fixed.Z;
else
    Zhat = zeros(M);
    for i=1:M
        for j=1:M
            Zhat(i,j) = sum(S(1:T-1) == i & S(2:T) == j);
        end
        if any(Zhat(i,:) > 0)
            Zhat(i,:) = Zhat(i,:) / sum(Zhat(i,:));
        else
            Zhat(i,i) = 1;
        end
    end
    if ~isempty(fixed.Z)
        idx = ~isnan(fixed.Z);
        Zhat(idx) = fixed.Z(idx);
    end
end



%-------------------------------------------------------------------------%
%     Check the specification of 'control, 'equal', 'fixed', 'scale'      %
%     and the agreement of parameter estimates with these arguments       %
%-------------------------------------------------------------------------%


pars = struct('A',Ahat, 'C', Chat, 'Q', Qhat, 'R', Rhat, 'mu', muhat, ...
    'Sigma', Sigmahat, 'Pi', Pihat, 'Z', Zhat);

test = preproc_dyn(M,N,p,r,pars,control,equal,fixed,scale); %#ok<NASGU>

% Turn warnings back on
% warning('on','MATLAB:singularMatrix'); 
% warning('on','MATLAB:nearlySingularMatrix');


