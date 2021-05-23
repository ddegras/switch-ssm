function V = get_covariance_aux(A,Q)


% Model dimensions
[r,~,p] = size(A);

% Define pr x pr matrix mapping pr x pr block-matrix V = (Gamma(k-l)) 
% (0 <= k,l < p) with Gamma(k) = Cov(x(t),x(t-k)) to corresponding indices
% in gamma = (vech(Gamma(0)), vec(Gamma(1)), ... , vec(Gamma(p))
% The idea is to map the entries of V, which are duplicated multiple times, 
% to their unique values in gamma
count = 0;
for l = 0:p-1
    if l == 0
        mask = logical(tril(ones(r)));
        Gamma_idx = zeros(r);
        Gamma_idx(mask) = 1:r*(r+1)/2;
        Gamma_idx = Gamma_idx + tril(Gamma_idx,-1)';
        Vtmp = kron(eye(p),Gamma_idx);
        count = count + r*(r+1)/2;
    else
        idx = count+1:count+r^2;
        Gamma_idx = reshape(idx(:),r,r);
        Vtmp = Vtmp + kron(diag(ones(p-l,1),l),Gamma_idx) + ...
            kron(diag(ones(p-l,1),-l),Gamma_idx');
        count = count + r^2;
    end
end
Gamma_idx = Vtmp;

% Solve associated linear equations to get gamma
nvars = p*r^2-r*(r-1)/2;
Abig = diag(ones((p-1)*r,1),-r);
Abig(1:r,:) = reshape(A,[r,p*r]);
LHS = eye((p*r)^2) - kron(Abig,Abig);
keep = zeros(1,nvars);
for i = 1:nvars
    idx = find(Gamma_idx == i);
    keep(i) = idx(1);
    LHS(:,idx(1)) = sum(LHS(:,idx),2);
end
LHS = LHS(:,keep);
RHS = zeros(p*r);
RHS(1:r,1:r) = Q; 
RHS = RHS(:);
gamma = LHS\RHS;

% Rearrange results for Cov(x(t),x(t-l))
V = reshape(gamma(Gamma_idx),p*r,p*r);
