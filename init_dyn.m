function [pars,Shat] = init_dyn(y,M,p,r,opts,control,equal,fixed,scale)

%--------------------------------------------------------------------------
%
%       INITIALIZATION OF EM ALGORITHM FOR STATE-SPACE MODEL 
%                   WITH MARKOV-SWITCHING DYNAMICS
%
% PURPOSE 
% This function calculates initial parameter estimates for the main
% EM fitting function 'switch_dyn'
%
% USAGE 
% [Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,Shat] ...
%           = init_dyn(y,M,p,r,opts,control,equal,fixed,scale)
%
% INPUTS
% y:    data (time = cols, variables = rows)
% M:    number of regimes for Markov chain
% p:    order of VAR state process
% r:    dimension of state vector
% opts:  optional structure with fields:
%       'segmentation':  with possible values 'fixed' (fixed segments) and
%           'binary' (binary segmentation). Default = 'fixed'
%       'len':  segment length for fixed segmentation or minimal distance
%           between two consecutive change points for binary segmentation.
%       'tol':  minimum relative decrease in loss function for a point to be
%           acceptable as change point. Only for binary segmentation. See
%           function find_single_cp for more details.
%       'Replicates': number of replicates in k-means (default=10)
%       'UseParallel': use parallel computing for k-means? (default=false)
% control:  optional structure with fields
%       'abstol':  absolute tolerance for eigenvalues when regularizing 
%           estimates of covariance matrices Q, R, and Sigma. Eigenvalues 
%           less than the lower bound abstol are replaced by it
%       'reltol':  relative tolerance for eigenvalues when regularizing 
%           estimates of covariance matrices Q, R, and Sigma. Eigenvalues 
%           less than the lower bound (max eigenvalue * reltol) are replaced 
%           by it
% 
% OUTPUTS
% pars: struct with fields
%       A:  estimated transition matrices (rxrxpxM)
%       C:  estimated observation matrix (Nxr with N = #rows in y) 
%       Q:  estimated state noise covariance matrices (rxrxM)
%       R:  estimated observation noise covariance matrix (NxN)
%       mu: estimated initial mean of state vector (rx1)
%       Sigma:  estimated initial covariance of state vector (rxr)
%       Pi: estimated probabilities of initial state of Markov chain (Mx1)
%       Z:  estimated transition probabilities (MxM)
% Shat: estimated Markov chain states S(t) (Tx1)
%
%--------------------------------------------------------------------------




%-------------------------------------------------------------------------%
%                        Preprocessing                                    %
%-------------------------------------------------------------------------%

% Check number of inputs
narginchk(4,9);

% Data dimensions
[N,T] = size(y);

% Data centering
y = y - mean(y,2);

% Check dimensions of input arguments
assert(r<=N, 'Dimension of state vector ''r'' exceeds dimension of time series')
assert(M<=T, 'Number of regimes ''M'' exceeds length of time series')
assert(r<=T, 'Dimension of state vector ''r'' exceeds length of time series')
assert(p<=T, 'VAR order ''p'' exceeds length of time series')
assert(p>=1)

% Disable warnings for (nearly) singular matrices
warning('off','MATLAB:singularMatrix'); 
warning('off','MATLAB:nearlySingularMatrix');



%-------------------------------------------------------------------------%
%     Set optional arguments to default values if not specified           %
%-------------------------------------------------------------------------%

opts0 = struct('segmentation','fixed','len',min(5*p*r,floor(T/(2*M))),...
    'tol',.05,'UseParallel',false,'Distance','cityblock', 'Replicates',10);
if exist('opts','var') && isstruct(opts)
    fname = fieldnames(opts0);
    for i = 1:numel(fname)
        if isfield(opts,fname{i})
            opts0.(fname{i}) = opts.(fname{i});
        end
    end
end
opts = opts0;
if opts.UseParallel
    opts.UseParallel = statset('UseParallel',1);
else
    opts.UseParallel = [];
end

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

equal0 = struct('A',false,'Q',false);
if exist('equal','var') && isstruct(equal)
    fname = fieldnames(equal0);
    for i = 1:numel(fname)
        if isfield(equal,fname{i})
            equal0.(fname{i}) = equal.(fname{i});
        end
    end
end
equal = equal0;   

assert(~(equal.A && equal.Q && M > 1),...
    ['If ''equal.A'' and ''equal.Q'' are both true, the model has ',...
        'effectively only one regime. Please modify these argument values ',...
        'or set ''M=1'' explicitly.'])
    
fixed0 = struct('A',[],'C',[],'Q',[],'R',[],'mu',[],'Sigma',[],...
    'Pi',[],'Z',[]);
if exist('fixed','var') &&isstruct(fixed)    
    fname = fieldnames(fixed0);
    for i = 1:numel(fname)
        if isfield(fixed,fname{i})
            fixed0.(fname{i}) = fixed.(fname{i});
        end
    end
end
fixed = fixed0;   

scale0 = struct('A',.999,'C',[]);
if exist('scale','var') &&isstruct(scale)
        fname = fieldnames(scale0);
    for i = 1:numel(fname)
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
for i = 1:numel(fname)
    name = fname{i};
    skip.(name) = ~isempty(fixed.(name)) && all(~isnan(fixed.(name)(:)));
end



%-------------------------------------------------------------------------%
%                      Estimate state vectors x(t)                        %
%-------------------------------------------------------------------------%

if skip.C
    Chat = fixed.C;
    xhat = (Chat'*Chat)\(Chat'*y);
else
    % SVD
    [~,D,V] = svd(y,'econ');
    % Estimates of state vectors x(t)
    xhat = D(1:r,1:r) * V(:,1:r)'; 
end

% Rescale x if required
if ~isempty(scale.C)
    xhat = xhat / scale.C;
end



%-------------------------------------------------------------------------%
%            Segment time series and estimate VAR matrix A                %
%             and state noise covariance matrix Q on segments             %
%-------------------------------------------------------------------------%



% In this part, the estimates of A(j) and Q(j) are pilot estimates and
% eventual fixed coefficient constraints are not taken into account (except
% if an entire parameter is fixed)

Y = xhat(:,p+1:T);  % response matrix
X = zeros(p*r,T-p); % predictor matrix
for lag = 1:p
    indx = (lag-1)*r+1:lag*r;
    indt = p-lag+1:T-lag;
    X(indx,:) = xhat(:,indt);
end

%@@@@@ Trivial cases for A
% Case: A entirely fixed
if skip.A
    Ahat = reshape(fixed.A,r,p*r,M);
% Case: M = 1 or active equality constraints on A
elseif (equal.A || M == 1)
    % Check 'one in ten' rule
    if (T-p) >= 10*p*r
        Ahat = (Y*X')/(X*X');
        if any(isnan(Ahat(:))|isinf(Ahat(:)))
            Ahat = (Y*X')*pinv(X*X');
        end
    else
        Ahat = zeros(r,p*r);
        for k = 1:r
            idx = k:r:k+(p-1)*r;
            Ahat(r,idx) = (Y(k,:)*X(idx,:)')/(X(idx,:)*X(idx,:)');
        end
        Ahat(isnan(Ahat(:)) | isinf(Ahat(:))) = 0;
    end
end

%@@@@@ Trivial cases for Q 
if skip.Q
    Qhat = fixed.Q;
elseif (equal.A && equal.Q) || M == 1 
%     Qhat = var(Y-Ahat*X,1,2);
%     lb = min(control.abstol,max(Qhat)*control.reltol);
%     Qhat(Qhat < lb) = lb;
%     Qhat = diag(Qhat);
    % full Qhat (non diagonal) @@@@@@
    e = Y-Ahat*X;
    Qhat = (e * e.') / T;
    Qhat = regfun(Qhat,control.abstol,control.reltol);
end

% Adjust the type of segmentation to be performed depending on equality
% constraints on A and Q. If there are equality constraints on both A and Q
% or if M = 1, no need for segmentation & clustering. 
if (equal.A && equal.Q) || M == 1 || (skip.A && skip.Q)
    opts.segmentation = '';
end
Adiag = 0;

switch opts.segmentation
    % Case: fixed segmentation
    case 'fixed'
    % Partition time range 1:T into shorter segments for VAR estimation &
    % classification. Heuristic: segments must be long enough so that
    % parameters (A,Q) can be reasonably well estimated, yet short enough
    % so that most segments do not contain change points and can be used
    % for subsequent clustering. 
    % Starting points of segments (accounting for shift by p)
    len = opts.len;
    start = [1:len:T-p,T-p+1];  
    % If last segment too short, collapse it with previous one
    if (T-p+1 - start(end-1)) < 0.9 * len 
        start = [start(1:end-2),T-p+1];
    end

    % Number of segments
    I = length(start)-1; 
    % VAR estimation on each segment
    % Rewrite VAR equation as Yi = Ai Xi + Ei i=1:I
    % with Yi: rx(ni-p), Ai: rx(p*r), Xi: (p*r)x(ni-p)
    % and ni = length of i-th segment
    if equal.A
        Ahat = repmat(Ahat(:,:,1),[1,1,I]);
    else
        Ahat = zeros(r,p*r,I);
    end
    Qhat = zeros(r,r,I);
    % Flag short segments: if segments too short to accurately estimate 
    % full A, make estimate of A diagonal
    Adiag = opts.len < 5*p*r; 
    for i = 1:I
        idx = start(i):start(i+1)-1;
        Xi = X(:,idx);
        Yi = Y(:,idx);
        % Estimated transition matrix A=[A1...Ap]
        if ~equal.A
            if ~Adiag
                Ai = (Yi*Xi')/(Xi*Xi'); 
                if any(isnan(Ai(:)) | isinf(Ai(:)))
                    Ai = (Yi*Xi') * pinv(Xi*Xi');
                end
            else
                Ai = zeros(r,p*r);
                for k = 1:r
                    idx = k:r:k+(p-1)*r;
                    Ai(k,idx) = (Yi(k,:)*Xi(idx,:)')/(Xi(idx,:)*Xi(idx,:)');
                end
                Ai(isnan(Ai(:)) | isinf(Ai(:))) = 0;
            end
            Ahat(:,:,i) = Ai;
        end
        % Estimated innovation covariance
        Qi = var(Yi-Ai*Xi,1,2);
        lb = min(control.abstol,max(Qi)*control.reltol);
        Qi(Qi < lb) = lb;
        Qhat(:,:,i) = diag(Qi); 
    end   
    if equal.Q
        Qhat = repmat(mean(Qhat,3),1,1,I);
    end
    % Shift back segment starts
    start = start+p;
    start(1) = 1;

    case 'binary'
    % Case: binary segmentation
    [Atmp,Qtmp,start] = find_all_cp(X,Y,opts.len,opts.tol);
    start = start+p; 
    start(1) = 1;
    I = length(start)-1;
    if equal.A
        Ahat = repmat(Ahat,[1,1,I]);
    else
        Ahat = Atmp;
    end
    if equal.Q
        Qhat = repmat(mean(Qhat,3),[1,1,I]);
    else
        Qhat = Qtmp;
    end
end



%-------------------------------------------------------------------------%
%          K-means clustering of estimates of A & Q on segments           %
%-------------------------------------------------------------------------%

% Typically the number I of segments is small compared to the size of A & Q,
% and not much larger than the number M of clusters/regimes. Therefore, to 
% enhance detection capacity, perform clustering on the diagonal terms of 
% A & Q rather than on the entire parameters
% if ~((equal.A && equal.Q) || M == 1 || (skip.A && skip.Q))
if (equal.A && equal.Q) || M == 1 
    Shat = repelem(1,T);    
elseif skip.A && skip.Q
    sse = zeros(M,T-p);
    for j = 1:M
        E = Y - Ahat(:,:,j) * X;
        Lj = chol(Qhat(:,:,j),'lower');
        sse(j,:) = sum((Lj\E).^2);
    end
    [~,Shat] = min(sse);
    Shat = [repelem(Shat(1),p), Shat];
else        
    % Indices of diagonal terms in [A,Q]
%     idx = find(repmat(eye(r),[1,1,p+1]));
%     d = length(idx); % (p+1)*r
    
    % For accurate clustering, estimated parameters A & Q for each segment
    % should be replicated according to segment lengths. (Segment lengths
    % may be very different in binary segmentation.) To avoid replicating 
    % the estimated parameters to a full set of T vectors (T may be large), 
    % divide all segment lengths by the shortest and replicate accordingly
%     start = start(:);
    segment_len = diff(start);
    switch opts.segmentation
        case 'fixed'
            new_len = ones(1,I);
        case 'binary'     
            new_len = round(10*segment_len/min(segment_len));
    end
    Thetahat = cell(I,1);
    for i = 1:I
%         AQ = [Ahat(:,:,i),Qhat(:,:,i)];
%         Thetahat{i} = repmat(reshape(AQ(idx),[1,d]),[new_len(i),1]);
        % Threshold small values to remove noise
        Ai = reshape(Ahat(:,:,i),1,[]); 
        thres = .1 * max(abs(Ai));
        Ai(abs(Ai) < thres) = 0;
        Qi = reshape(Qhat(:,:,i),1,[]); 
        thres = .1 * max(diag(Qi));
        Qi(abs(Qi) < thres) = 0;
        Thetahat{i} = repmat([Ai,Qi],new_len(i),1);
    end
    Thetahat = vertcat(Thetahat{:});
    
    % K-means clustering
    [Shat,~] = kmeans(Thetahat,M,'Replicates',opts.Replicates,...
        'Distance',opts.Distance, 'Options',opts.UseParallel); 
    clear Thetahat
   
    % If S(1)!=1, say S(1)=j, swap cluster labels 1 and j so that S(1)=1 
    if Shat(1) ~= 1
        j = Shat(1);
        idx = (Shat == 1);
        Shat(Shat == j) = 1;
        Shat(idx) = j;
    end    
    
    % Extract I "true" values in Shat
    idx = cumsum([1,new_len(1:end-1)]);
    Shat = Shat(idx);
    
    % Replicate the elements Shat as required (I --> T)
%     segment_len(1) = segment_len(1) + p; 
    Shat = repelem(Shat,segment_len);
end

 


%-------------------------------------------------------------------------%
%                     Final parameter estimates                           %
%-------------------------------------------------------------------------%


% Case: time series too short to accurately estimate full A
if Adiag && isempty(fixed.A)
    mask = diag(NaN(r,1));
    fixed.A = repmat(mask,[1,1,p,M]);
end

pars = reestimate_dyn(y,M,p,r,Shat,control,equal,fixed,scale);

% Estimate initial probabilities Pi and transition probabilities Z
% with a regularization step
Pihat = ones(M,1) * .01;
Pihat(Shat(1)) = 1;
Pihat = Pihat / sum(Pihat);
% Pihat = round(Pihat,6);
% Pihat(1) = 1 - sum(Pihat(2:end));
if ~isempty(fixed.Pi)
    idx = ~isnan(fixed.Pi);
    Pihat(idx) = fixed.Pi(idx);
end  
pars.Pi = Pihat;

Zhat = zeros(M);
for i=1:M
        for j=1:M
            Zhat(i,j) = sum(Shat(1:T-1) == i & Shat(2:T) == j);
        end
        if all(Zhat(i,:) == 0) 
            Zhat(i,:) = 1/M;
        end
        % Clamp very small values of Z(i,j)
        Zi = Zhat(i,:);
        lb = .01 * max(Zi);
        Zi(Zi < lb) = lb;
        % Rescale so that row sums are 1
        Zi = Zi/sum(Zi);
%         Zi = round(Zi,6);
%         Zi(1) = 1 - sum(Zi(2:M));
        Zhat(i,:) = Zi;
end
if ~isempty(fixed.Z)
    idx = ~isnan(fixed.Z);
    Zhat(idx) = fixed.Z(idx);
end
pars.Z = Zhat;

% Turn warnings back on
% warning('on','MATLAB:singularMatrix'); 
% warning('on','MATLAB:nearlySingularMatrix');



end




%%

%-------------------------------------------------------------------------%
%                       FUNCTION find_single_cp                           %
%-------------------------------------------------------------------------%


% Purpose: find a single change point in time series regression (Xt,Yt)
% Input
% X:    predictor matrix (dimension pxT)
% Y:    response matrix (dimension NxT)
% delta: mininum distance between two consecutive change points
% tol:  minimum relative decrease in loss function for a point to be
% accepted as change point

% Output
% cp:   change point

% Details
% For each time point t in 1+delta:T+1-delta, the time range 1:T is split
% into 1:t-1 and t:T and the regression model Y = AX + E is fit on each of
% these two segments. Denote by A0, A1, and A2 the OLS estimates for 1:T,
% 1:t-1, and t:T, respectively, and denote by SSE0, SSE1(t), and SSE2(t)
% the associated sum of squared errors. The candidate change point t0 is
% the time point that minimizes SSE1(t) + SSE2(t). It is accepted as a
% change point if (SSE1(t0)+SSE2(t0)) <= (1-tol) * SSE0.
% If no change point is found, the function returns the time point 1.
% In particular, there can be no change point if 2*delta > T. 


function cp = find_single_cp(X,Y,delta,tol)

% Data dimensions
T = size(Y,2);

% Check that search interval is long enough (at least twice delta)
% cp = 1;
cp = []; % @@@@@@ dev
if T < 2*delta
    return
end

% Make estimate of A diagonal if not enough observations to estimate A
% accurately
Adiag = delta < 5 * size(X,1);

% Initialization
YX = Y*X'; % sum(i=1:n) Y(i)X(i)'
XX = X*X'; % sum(i=1:n) X(i)X(i)'
A2 = YX/XX;
if any(isnan(A2))
    A2 = YX*pinv(XX);
end
sst = norm(Y,'fro')^2; % sum(i=1:n) Y(i)'Y(i)
sse0 = sst - sum(diag(A2*YX')); % sum of squared errors assuming no change point
sse_best = sse0;


% Search for best candidate change point 
XXt = X(:,1:delta)*X(:,1:delta)'; % running sum Xt*Xt'
YXt = Y(:,1:delta)*X(:,1:delta)'; % running sum Yt*Xt'
for t=1+delta:T+1-delta
    YXt = YXt + Y(:,t)*X(:,t)';
    XXt = XXt + X(:,t)*X(:,t)';
    A1 = YXt/XXt; % estimate for left interval
    A2 = (YX-YXt)/(XX-XXt); % estimate for right interval
    sse = sst - sum(diag(A1*YXt')) - sum(diag(A2*(YX-YXt)'));
    % Case where XXt or XX-XXt is numerically singular
    if isnan(sse)
        A1 = YXt*pinv(XXt);
        A2 = (YX-YXt)*pinv(XX-XXt);
        sse = sst - sum(diag(A1*YXt')) - sum(diag(A2*(YX-YXt)'));
    end
    if sse < sse_best
        sse_best = sse;
        cp = t;
    end
end

% Check that candidate change point achieves sufficient reduction in
% loss function
if sse_best > (1-tol) * sse0 % @@@@@@@@ dev
% if (sse_best > (1-tol)*sse0 || cp <= delta || cp >= T-delta)
    cp = []; % @@@@@@ dev 
%     cp = 1;
end

end



%%


%-------------------------------------------------------------------------%
%                         FUNCTION find_all_cp                            %
%-------------------------------------------------------------------------%


% Purpose: find all change points in time series regression (Xt,Yt) by
% binary segmentation. 

% Input
% X:    predictor matrix (dimension pxT)
% Y:    response matrix (dimension NxT)
% delta: mininum distance between two consecutive change points
% tol:  minimum relative decrease in loss function for a point to be
% accepted as change point

% Output
% Ahat: estimates of A on each segment (dimension NxpxI, with I=#segments)
% Qhat: estimates of Q on each segment (dimension NxNxI)
% cp:   change points (first points of each segment)

% Details
% For convenience, the first change point is always set to 1 and the last
% to T+1.


function [Ahat,Qhat,cp] = find_all_cp(X,Y,delta,tol)


% Data dimensions
[N,T] = size(Y);
% VAR order
% p = size(X,1)/size(Y,1);

% Initial change points
cp = [1,T+1];

% Search for change points
while 1 
    cp_old = cp;
    I = length(cp)-1; % number of segments
%     cp_new = zeros(1,I);
    cp_new = []; % @@@@@@ dev
    for i=1:I
        idx = cp(i):cp(i+1)-1;
        pos = find_single_cp(X(:,idx),Y(:,idx),delta,tol);
        if ~isempty(pos) % @@@@@@ dev
            cp_new = [cp_new,cp(i)-1+pos]; %#ok<AGROW>
        end
%         cp_new(i) = cp(i)-1+pos;
    end
    cp = sort(unique([cp,cp_new]));
    if isequal(cp_old,cp)
        break
    end
end
    
% Parameter estimates
I = length(cp)-1; 
Ahat = zeros(N,size(X,1),I);
Qhat = zeros(N,N,I);
for i=1:I
    idx = cp(i):cp(i+1)-1;
    Xi = X(:,idx);
    Yi = Y(:,idx);
    Ai = (Yi*Xi')/(Xi*Xi');
    if any(isnan(Ai))
        Ai = (Yi*Xi')*pinv(Xi*Xi');
    end
    Ahat(:,:,i) = Ai;
    Qhat(:,:,i) = cov((Yi-Ai*Xi)');
end
end

