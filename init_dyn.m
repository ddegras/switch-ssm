
function [Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,Pihat,Zhat,Shat] ...
    = init_dyn(y,M,p,r,opts,control,equal,fixed,scale)

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
%       'len':  segment length. Only for fixed segmentation. 
%       'delta':  minimal distance between two consecutive change points.
%           Only for binary segmentation.
%       'tol':  minimum relative decrease in loss function for a point to be
%           acceptable as change point. Only for binary segmentation. See
%           function find_single_cp for more details.
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
% Ahat:     estimate of transition matrices (rxrxpxM)
% Chat:     estimate of observation matrix (Nxr with N = #rows in y) 
% Qhat:     estimate of state noise covariance matrices (rxrxM)
% Rhat:     estimate of transition matrices (NxN)
% muhat:    estimate of initial mean of state vector (rx1)
% Sigmahat:  estimate of initial covariance of state vector (rxr)
% Pihat:    estimate of probabilities of initial state of Markov chain (Mx1)
% Zhat:     estimate of transition probabilities (MxM)
% Shat:     estimate of Markov chain states S(t) (Tx1)
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

% Disable warnings for (nearly) singular matrices
warning('off','MATLAB:singularMatrix'); 
warning('off','MATLAB:nearlySingularMatrix');



%-------------------------------------------------------------------------%
%     Set optional arguments to default values if not specified           %
%-------------------------------------------------------------------------%

opts0 = struct('segmentation','fixed',...
    'len',[],'delta',max(2*p,floor(T/10)),'tol',.05);
if exist('opts','var') && isstruct(opts)
    % Check argument 'segmentation'
    if isfield(opts,'segmentation') 
        opts0.segmentation = opts.segmentation;
    end
    % Check arguments 'segments', 'delta' and 'tol'
    if isfield(opts,'len') 
            opts0.len = min(opts.len,floor(T/M));
    end
    if isfield(opts,'delta') 
            opts0.delta = opts.delta;
    end
    if isfield(opts,'tol')
        opts0.tol = opts.tol;
    end
end
opts = opts0;

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

equal0 = struct('A',false,'Q',false);
if exist('equal','var') && isstruct(equal)
    fnames = fieldnames(equal0);
    for i = 1:numel(fnames)
        if isfield(equal,fnames{i})
            equal0.(fnames{i}) = equal.(fnames{i});
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
    fnames = fieldnames(fixed0);
    for i = 1:numel(fnames)
        if isfield(fixed,fnames{i})
            fixed0.(fnames{i}) = fixed.(fnames{i});
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
    Ahat = (Y*X')/(X*X');
    if any(isnan(Ahat(:))|isinf(Ahat(:)))
        Ahat = (Y*X')*pinv(X*X');
    end
end

%@@@@@ Trivial cases for Q 
if skip.Q
    Qhat = fixed.Q;
elseif (equal.A && equal.Q) || M == 1 
    Qhat = diag(var(Y-Ahat*X,0,2));
end

% Adjust the type of segmentation to be performed depending on equality
% constraints on A and Q. If there are equality constraints on both A and Q
% or if M = 1, no need for segmentation & clustering. 
if (equal.A && equal.Q) || M == 1 || (skip.A && skip.Q)
    opts.segmentation = 'other';
    opts.reestimation = false;
end

switch opts.segmentation
    % Case: fixed segmentation
    case 'fixed'
    % Partition time range 1:T into shorter segments for VAR estimation &
    % classification. Heuristic: segments must be long enough so that
    % parameters (A,Q) can be reasonably well estimated, yet short enough
    % so that most segments do not contain change points and can be used
    % for subsequent clustering. Practical rule: take segment length at
    % least 6*p*r for accurate estimation and at most T/(3*M) for accurate
    % clustering (this ensures at least 3 times as many segments as
    % regimes/clusters).
    % Segment length (# time points)
    if ~isempty(opts.len)
        len = opts.len;
    else
        lb = 6*p*r;
        ub = floor(T/(3*M));        
        len = min(max(lb,round(T/10)),ub);
        len = min(max(len,p+1),floor(T/M));
    end
    % Starting points of segments
    start = [1:len:T-len+1,T+1];
    % If last segment too short, collapse it with previous one
    if (start(end) - start(end-1)) < .9 * len
        start = setdiff(start,start(end-1));
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
%     if equal.Q
%         Qhat = repmat(Qhat(:,:,1),[1,1,I]);
%     else
        Qhat = zeros(r,r,I);
%     end
    for i = 1:I
        idx = start(i):start(i+1)-p-1;
        Xi = X(:,idx);
        Yi = Y(:,idx);
        % Estimated transition matrix A=[A1...Ap]
        if ~equal.A
            Ai = (Yi*Xi')/(Xi*Xi'); 
            if any(isnan(Ai(:)) | isinf(Ai(:)))
                Ai = (Yi*Xi') * pinv(Xi*Xi');
            end
            Ahat(:,:,i) = Ai;
        end
        % Estimated innovation covariance
%         if ~equal.Q
            Qhat(:,:,i) = diag(var(Yi-Ahat(:,:,i)*Xi,0,2)); 
%         end
    end   
    if equal.Q
        Qhat = repmat(mean(Qhat,3),1,1,I);
    end
    
    case 'binary'
    % Case: binary segmentation
    [Atmp,Qtmp,start] = find_all_cp(X,Y,opts.delta,opts.tol);
    start = start+p; 
    start(1) = 1;
    I = length(start)-1;
    if equal.A
        Ahat = repmat(Ahat,[1,1,I]);
    else
        Ahat = Atmp;
    end
    if equal.Q
        Qhat = repmat(Qhat,[1,1,I]);
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
if ~((equal.A && equal.Q) || M == 1 || (skip.A && skip.Q))
    % Indices of diagonal terms in [A,Q]
    idx = find(repmat(eye(r),[1,1,p+1]));
    d = length(idx); % (p+1)*r
    
    % For accurate clustering, estimated parameters A & Q for each segment
    % should be replicated according to segment lengths. (Segment lengths
    % may be very different in binary segmentation.) To avoid replicating 
    % the estimated parameters to a full set of T vectors (T may be large), 
    % divide all segment lengths by the shortest and replicate accordingly
    start = start(:);
    segment_len = diff(start);
    new_len = round(segment_len/min(segment_len));
    Thetahat = cell(I,1);
    for i = 1:I
        AQ = [Ahat(:,:,i),Qhat(:,:,i)];
        Thetahat{i} = repmat(reshape(AQ(idx),[1,d]),[new_len(i),1]);
    end
    Thetahat = vertcat(Thetahat{:});
    
    % K-means clustering
    [Shat,~] = kmeans(Thetahat,M,'Replicates',10); 
    
    % If S(1)!=1, say S(1)=j, swap cluster labels 1 and j so that S(1)=1 
    if Shat(1) ~= 1
        j = Shat(1);
        idx = (Shat == 1);
        Shat(Shat == j) = 1;
        Shat(idx) = j;
    end    
    
    % Extract I "true" values in Shat
    idx = cumsum([1;new_len(1:end-1)]);
    Shat = Shat(idx);
    
    % Replicate the elements Shat as required (I --> T)
    Shat = repelem(Shat,segment_len);
end

clear AQ Thetahat
 


%-------------------------------------------------------------------------%
%                     Final parameter estimates                           %
%-------------------------------------------------------------------------%

[Ahat,Chat,Qhat,Rhat,muhat,Sigmahat,~,~] = ...
    reestimate_dyn(y,M,p,r,Shat,control,equal,fixed,scale);



% Estimate initial probabilities Pi and transition probabilities Z
% with a regularization step
Pihat = zeros(M,1);
Pihat(Shat(1)) = 1;
Pihat(Pihat < 1) = .01;
Pihat = Pihat / sum(Pihat);
if ~isempty(fixed.Pi)
    idx = ~isnan(fixed.Pi);
    Pihat(idx) = fixed.Pi(idx);
end  

Zhat = zeros(M);
for i=1:M
        for j=1:M
            Zhat(i,j) = sum(Shat(1:T-1) == i & Shat(2:T) == j);
        end
        % Clamp very small values of Z(i,j)
        Zi = Zhat(i,:);
        lb = .01 * max(Zi);
        Zi(Zi < lb) = lb;
        % Rescale so that row sums are 1
        Zhat(i,:) = Zi/sum(Zi);
end
if ~isempty(fixed.Z)
    idx = ~isnan(fixed.Z);
    Zhat(idx) = fixed.Z(idx);
end

% Turn warnings back on
warning('on','MATLAB:singularMatrix'); 
warning('on','MATLAB:nearlySingularMatrix');


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
% change point if (SSE1(t0)+SSE2(t0)) <= tol * SSE0.
% If no change point is found, the function returns the time point 1.
% In particular, there can be no change point if 2*delta > T. 


function cp = find_single_cp(X,Y,delta,tol)

% Data dimensions
[~,T] = size(Y);

% Check that search interval is long enough (at least twice delta)
cp = 1;
if T < (2*delta)
    return
end

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
if (sse_best > (1-tol)*sse0 || cp <= delta || cp >= T-delta)
    cp = 1;
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
cp = [1 T+1];

% Search for change points
while 1 
    cp_old = cp;
    I = length(cp)-1; % number of segments
    cp_new = zeros(1,I);
    for i=1:I
        ind = cp(i):cp(i+1)-1;
        pos = find_single_cp(X(:,ind),Y(:,ind),delta,tol);
        cp_new(i) = cp(i)-1+pos;
    end
    cp = unique([cp cp_new]);
    if isequal(cp_old,cp)
        break
    end
end
    
% Parameter estimates
I = length(cp)-1; 
Ahat = zeros(N,size(X,1),I);
Qhat = zeros(N,N,I);
for i=1:I
    ind = cp(i):cp(i+1)-1;
    Xi = X(:,ind);
    Yi = Y(:,ind);
    Ai = (Yi*Xi')/(Xi*Xi');
    if any(isnan(Ai))
        Ai = (Yi*Xi')*pinv(Xi*Xi');
    end
    Ahat(:,:,i) = Ai;
    Qhat(:,:,i) = cov((Yi-Ai*Xi)');
end
end

