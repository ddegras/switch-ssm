function pars = reestimate_obs(y,M,p,r,S,control,equal,fixed,scale)



%-------------------------------------------------------------------------%
%                           Preprocessing                                 % 
%-------------------------------------------------------------------------%

% Data dimensions
[N,T] = size(y);

%@@@@@ Optional arguments
control0 = struct('abstol',1e-8,'reltol',1e-4); 
if exist('control','var') && isstruct(control) 
    fnames = fieldnames(control0);
    for i = 1:numel(fnames)
        if isfield(control,fnames{i})
            control0.(fnames{i}) = control.(fnames{i});
        end
    end
end
control = control0;
abstol = control.abstol;
reltol = control.reltol;

equal0 = struct('A',false,'C',false,'Q',false);
if exist('equal','var') && isstruct(equal)
    name = {'A','C','Q'};
    for i=1:3        
        if isfield(equal,name{i}) && ~isempty(equal.(name{i}))
        equal0.(name{i}) = equal.(name{i});
        end
    end
end
equal = equal0;

z = zeros(0,2);
fixed0 = struct('A',z,'C',z,'Q',z,'R',z,'mu',z,'Sigma',z,'Pi',z,'Z',z);
if exist('fixed','var') && isstruct(fixed)
    dims = struct('A',[r,r,p,M],'C',[N,r,M],'Q',[r,r,M],'mu',[r,M],'Sigma',[r,r,M]);
    name = fieldnames(fixed0);
    for i = 1:numel(name)
        if isfield(fixed,name{i})
            fixed_i = fixed.(name{i});
            % If argument 'equal' is true and argument 'fixed' is only
            % specified for one regime, replicate 'fixed' M times
            if isfield(equal,name{i}) && equal.(name{i}) && ...
                    isfield(dims,name{i}) 
                dim_i = dims.(name{i});
                if numel(fixed_i) == prod(dims_i)/M
                    if isrow(fixed_i)
                        fixed_i = fixed_i(:);
                    end                
                    fixed_i = reshape(fixed_i,dim_i(1:numel(dim_i)-1));
                    fixed_i = repmat(fixed_i,[repelem(1,numel(dim_i)-1),M]);
                end
            end
            fixed0.(name{i}) = fixed_i;
        end
    end
end
fixed = fixed0;            

scale0 = struct('A',.999,'C',[]);
if exist('scale','var') &&isstruct(scale)
        fnames = fieldnames(scale0);
    for i = 1:numel(fnames)
        if isfield(scale,fnames{i})
            scale0.(fnames{i}) = scale.(fnames{i});
        end
    end
end
scale = scale0;

% Center the data
y = y - mean(y,2);

% Remove warnings when inverting singular matrices
warning('off','MATLAB:singularMatrix');
warning('off','MATLAB:nearlySingularMatrix');
warning('off','MATLAB:illConditionedMatrix');


%-------------------------------------------------------------------------%
%         Check whether some parameters are entirely fixed                %
%-------------------------------------------------------------------------%

skip = struct();
fnames = fieldnames(fixed);
for i = 1:numel(fnames)
    name = fnames{i};
    skip.(name) = ~isempty(fixed.(name)) && all(~isnan(fixed.(name)(:)));
end


%-------------------------------------------------------------------------%
%                   Estimate observation matrices C                       % 
%-------------------------------------------------------------------------%


freq = tabulate(S);  % frequency table of regime
Meff = size(freq,1); % effective number of regimes 
unq = freq(:,1)';    % effective regime labels

% Unconstrained estimate
if skip.C
    Chat = fixed.C;
elseif equal.C
    [U,~,~] = svd(y,'econ');
    Chat = repmat(U(:,1:r),1,1,M);
else    
    Chat = zeros(N,r,M);
    if Meff < M 
        [Uall,~,~] = svd(y,'econ');
    end
    for j = 1:M
        if ismember(j,unq)
            [U,~,~] = svd(y(:,S == j),'econ');
            Chat(:,:,j) = U(:,1:r);
        else
            Chat(:,:,j) = Uall(:,1:r);
        end
    end
end

% Apply eventual scale constraints
if ~skip.C && ~isempty(scale.C)
    Chat = Chat * scale.C;
end
        
% Estimate state vectors x(t,j) for t: S(t)=j
xhat = NaN(r,T);
for j = 1:M     
    if equal.C
        Sj = 1:T;
    else
        Sj = (S == j); 
    end
    C_j = Chat(:,:,j);
    xhat(:,Sj) = (C_j'*C_j)\(C_j'*y(:,Sj));
end

% Re-estimate C(j) under fixed coefficient constraints if required
if ~skip.C && ~isempty(fixed.C) 
    for j = 1:M
        Ctmp = fixed.C(:,:,j);
        idx = find(~isnan(Ctmp));
        if isempty(idx)
            continue
        end
        fixed_C = [idx,Ctmp(idx)];
        C_j = Chat(:,:,j);
        XX = xhat(:,Sj) * xhat(:,Sj)';
        XY = xhat(:,Sj) * y(:,Sj)';
        [C_j,err] = PG_C(C_j,XY,XX,eye(N),scale.C,fixed_C,100);
        Chat(:,:,j) = C_j;        
        if err
             error(['Cannot find estimate of C%d satisfying both',...
                ' fixed coefficient constraints (''fixed.C'') and scale',...
                ' constraints (''scale.C'').\nPlease check that the two',...
                ' constraints are mutually compatible and consider',...
                ' modifying/removing some constraints.'],j)
        end
    end
end



%-------------------------------------------------------------------------%
%                  Estimate observation noise covariance R                % 
%-------------------------------------------------------------------------%


    
% Unconstrained estimates over each subset Sj = {t:S(t)=j)
Rhat = zeros(N,N,Meff);
for j = 1:Meff
    Sj = (S == unq(j));
    Ej = y(:,Sj) - Chat(:,:,unq(j)) * xhat(:,Sj);
    Rhat(:,:,j) = cov(Ej');
end

% Weighted average of previous estimates, with weights proportional to
% occupancy time
occup = freq(:,3) / sum(freq(:,3));
Rhat = reshape(Rhat,N*N,Meff) * occup;
Rhat = reshape(Rhat,N,N);
Rhat = 0.5 * (Rhat + Rhat');

% Apply eventual fixed coefficient constraints
if ~isempty(fixed.R)
    idx = ~isnan(fixed.R);
    Rhat(idx) = fixed.R(idx);
end

% Check positive-definiteness and conditioning. 
% Diagonalize and regularize if needed. 
if ~skip.R
    eigval = eig(Rhat);
    if min(eigval) < max(abstol,reltol*max(eigval))
        Rhat = regfun(diag(diag(Rhat)),abstol,reltol);
        if ~isempty(fixed.R)
            Rhat(idx) = fixed.R(idx);
        end
    end
end


%-------------------------------------------------------------------------%
%     Estimate initial mean mu and covariance Sigma of state process      % 
%-------------------------------------------------------------------------%


% Assume mu(1)=...=mu(M) and Sigma(1)=...=Sigma(M)


% Number of time points used in estimation of mu and Sigma
t0 = min([10,5*p,T]);  

muhat = repmat(mean(xhat(:,1:t0),2),[1,M]); 
if ~isempty(fixed.mu)
    idx = ~isnan(fixed.mu);
    muhat(idx) = fixed.mu(idx);
end

Sigmahat = diag(var(xhat(:,1:t0),0,2));
Sigmahat = repmat(Sigmahat,[1,1,M]); 
if ~isempty(fixed.Sigma)
    idx = ~isnan(fixed.Sigma);
    Sigmahat(idx) = fixed.Sigma(idx);       
end
Sigmahat = regfun(Sigmahat(:,:,1),abstol,reltol);
Sigmahat = repmat(Sigmahat,[1,1,M]); 
if ~isempty(fixed.Sigma)
    Sigmahat(idx) = fixed.Sigma(idx);       
end



%-------------------------------------------------------------------------%
%                   Estimate VAR transition matrices A                    %
%-------------------------------------------------------------------------%


if skip.A
    Ahat = fixed.A;
end
if ~skip.A || ~skip.Q
    Ahat = zeros(r,p*r,M);
    group = cell(M,1);      % G(j) = {t:S(t)=...=S(t+p)=j}
    groupsize = zeros(M,1); % #G(j)
    XX = zeros(p*r,p*r,M);  % sum(t in G(j)) Xhat(t,j) Xhat(t,j)'
    YX = zeros(r,p*r,M);    % sum(t in G(j)) xhat(t,j) Xhat(t,j)'
    YY = zeros(r,r,M);      % sum(t in G(j)) xhat(t,j) xhat(t,j)'
    fixed_A = cell(M,1);

    % Prepare estimation
    for j = 1:M

        % Fixed coefficients in A(j), two-column format
        if ~isempty(fixed.A)
            Atmp = fixed.A(:,:,:,j);
            idx = find(~isnan(Atmp));
            fixed_A{j} = [idx,Atmp(idx)];
        end

        Sj = (S == j);          % Sj = {t:S(t)=j)
        test = zeros(p+1,T-p); 
        for l = 0:p
            test(l+1,:) = Sj(p-l+1:T-l);
        end
        group{j} = find(all(test));   
        groupsize(j) = numel(group{j});
        if groupsize(j) == 0
            continue
        end

        % Set up autoregression for A(j) and Q(j)
        Yj = xhat(:,p+group{j});
        Xj = zeros(p*r,groupsize(j));
        for l = 1:p
            Xj((l-1)*r+1:l*r,:) = xhat(:,(p-l)+group{j});
        end 
        XX(:,:,j) = Xj * Xj';
        YX(:,:,j) = Yj * Xj';
        YY(:,:,j) = Yj * Yj';
    end

    % Estimate A(j) 
    for j = 1:M

        % Set up autoregression
        if equal.A
            XXj = sum(XX,3);
            YXj = sum(YX,3);
        else
            XXj = XX(:,:,j);
            YXj = YX(:,:,j);
        end

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
                [A_j,err] = PG_A(A_j,YXj,XXj,eye(r),scale.A,fixed_A,100);
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
%                   Estimate state noise covariance Q                     %
%-------------------------------------------------------------------------%


if skip.Q
    Qhat = fixed.Q;
else
    Qhat = zeros(r,r,M);

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
        if sum(groupsize) > 0
            Qhat = reshape(Qhat,r^2,M) * (groupsize / sum(groupsize));
            Qhat = reshape(Qhat,r,r,M);
        else
            Qhat = repmat(diag(var(xhat,1,2)),1,1,M);
        end
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


pars = struct('A',Ahat, 'C', Chat, 'Q', Qhat, 'R', Rhat, 'mu', muhat, ...
    'Sigma', Sigmahat, 'Pi', Pihat, 'Z', Zhat);


