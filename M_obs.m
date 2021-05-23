function outpars = M_obs(pars,Ms,P0,sum_CP,sum_MP,sum_Ms2,...
        sum_Mxy,sum_P,sum_Pb,sum_yy,x0,control,equal,fixed,scale,skip)
    

abstol = control.abstol;
reltol = control.reltol;
verbose = control.verbose;
outpars = pars;
[N,r,M] = size(pars.C);
p = size(sum_Pb,1) / size(sum_P,1);
T = size(Ms,2);

% Mask of rxr diagonal blocks in a (p*r)x(p*r) matrix
% (used to update Sigma in M-step)
Sigmamask = reshape(find(kron(eye(p),ones(r))),r,r,p);

% Unconstrained estimates 
% A = ( sum(t=p+1:T) P(t,t-1|T) ) * ( sum(t=p+1:T) P~(t-1|T) )^{-1}
% Cj = (sum(t=1:T) Wj(t) y(t) xj(t)') * (sum(t=1:T) Wj(t) Pj(t|T))^{-1}
% Qj = (sum(t=2:T) Wj(t) Pj(t) - Aj * sum(t=2:T) Wj(t) Pj(t-1,t)') / sum(t=2:T) Wj(t)
% R = sum(t=1:T) y(t) y(t)' / T - sum(t=1:T) x(t|T) y(t)'
%
% where Wj(t) = P(S(t)=j|y(1:T)), 
% xj(t|T) = E(x(t)|S(t)=j,y(1:T)), 
% x(t|T) = E(x(t)|y(1:T)), 
% Pj(t|T) = E(x(t)x(t)'|S(t)=j,y(1:T)), 
% P~(t-1|T) = E(x(t-1)x(t-1)'|S(t)=j,y(1:T))
% and P(t,t-1|T) = E(x(t)x(t-1)'|y(1:T))

% Scale constraints for C are handled with a
% projected gradient technique (maximize Q-function under constraint)



%=========================================================================%
%                               Update A                                  %
%=========================================================================%




% Case: no fixed coefficient constraints
if isempty(fixed.A)
    if equal.A && equal.Q
        sum_Pb_all = sum(sum_Pb,3);
        sum_CP_all = sum(sum_CP,3);
        Ahat = sum_CP_all / sum_Pb_all;
        if any(isnan(Ahat(:))|isinf(Ahat(:)))
            Ahat = sum_CP_all * pinv(sum_Pb_all);
        end
        Ahat = repmat(Ahat,[1,1,M]);
    elseif equal.A
        lhs = zeros(p*r*r);
        rhs = zeros(r,p*r);
        for j = 1:M
            Qinv_j = myinv(pars.Q(:,:,j));
            lhs = lhs + kron(sum_Pb(:,:,j),Qinv_j);
            rhs = rhs + Qinv_j * sum_CP(:,:,j);
        end
        Ahat = lhs\rhs(:);
        if any(isnan(Ahat))|| any(isinf(Ahat))
            Ahat = pinv(lhs) * rhs(:);
        end
        Ahat = repmat(reshape(Ahat,r,p*r),1,1,M);
    else
        Ahat = zeros(r,p*r,M);
        for j = 1:M
            A_j = sum_CP(:,:,j) / sum_Pb(:,:,j);
            if any(isnan(A_j(:)) | isinf(A_j(:)))
                A_j = sum_CP(:,:,j) * pinv(sum_Pb(:,:,j));
            end
            Ahat(:,:,j) = A_j;
        end
    end
end

% Case: fixed coefficient constraints on A --> Vectorize matrices and
% solve associated problem after discarding rows associated with fixed
% coefficients. Recall: there cannot be both fixed coefficient
% constraints *and* equality constraints on A
if ~skip.A && ~isempty(fixed.A)
    for j = 1:M
        % Linear indices of free coefficients in A(j)
        idx = (fixed.A(:,1) > (j-1)*p*r^2) & (fixed.A(:,1) <= j*p*r^2); 
        fixed_Aj = fixed.A(idx,:);
        fixed_Aj(:,1) = fixed_Aj(:,1) - (j-1)*p*r^2;
        free = setdiff(1:p*r^2,fixed_Aj(:,1));
        free = reshape(free,[],1);
        Qinv_j = myinv(pars.Q(:,:,j));
        % Matrix problem min(X) trace(W(-2*B1*X' + X*B2*X')) 
        % (under fixed coefficient constraints) becomes vector problem 
        % min(x) x' kron(B2,W) x - 2 x' vec(W*B1)
        % with X = A(j), x = vec(A(j)), W = Q(j)^(-1), B1 = sum_CP(j),
        % and B2 = sum_Pb(j) (remove fixed entries in x)
        mat = kron(sum_Pb(:,:,j),Qinv_j);
        vec = reshape(Qinv_j * sum_CP(:,:,j),p*r^2,1);            
        A_j = zeros(p*r^2,1);
        A_j(fixed_Aj(:,1)) = fixed_Aj(:,2);
        A_j(free) = mat(free,free)\vec(free);
        if any(isnan(A_j)|isinf(A_j))
            A_j(free) = pinv(mat(free,free)) * vec(free);
        end
        Ahat(:,:,j) = reshape(A_j,r,p*r);
    end
end

% Check eigenvalues of estimate and regularize if less than 'scale.A'.
% Regularization: algebraic method if no fixed coefficients or all
% fixed coefficients are zero, projected gradient otherwise      
if ~skip.A
    Abig = diag(ones((p-1)*r,1),-r);
    for j = 1:M
        % Check eigenvalues
        Abig(1:r,:) = Ahat(:,:,j);
        eigval = eig(Abig);
        if any(abs(eigval) > scale.A)
            if verbose
                warning(['Eigenvalues of A%d greater than %f.',...
                    ' Regularizing.'],j,scale.A)
            end
            c = .999 * scale.A / max(abs(eigval));
            A_j = reshape(Ahat(:,:,j),[r,r,p]);
            for l = 1:p
                A_j(:,:,l) = c^l * A_j(:,:,l);
            end 
            Ahat(:,:,j) = reshape(A_j,[r,p*r]);
        end
        if equal.A
            Ahat = repmat(Ahat(:,:,1),[1,1,M]);
            break
        end 
    end
    
    % Check that parameter update actually increases Q-function 
    % If not, keep previous parameter estimate 
    Qval_old = Q_obs(outpars,Ms,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,...
        sum_P,sum_Pb,sum_yy,x0);
    outpars.A = Ahat;
    Qval = Q_obs(outpars,Ms,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,...
        sum_P,sum_Pb,sum_yy,x0);
    if Qval < Qval_old
        outpars.A = pars.A;
    end
end







if skip.C 
    Chat = pars.C;
elseif isempty(fixed.C) && isempty(scale.C)
    if equal.C
        sum_yx = sum(sum_Mxy,3).';
        Chat = sum_yx / sum_P;
        if any(isnan(Chat(:))|isinf(Chat(:)))
            Chat = sum_yx * pinv(sum_P);
        end
        Chat = repmat(Chat,1,1,M);
    else
        Chat = zeros(N,r,M);
        for j=1:M
            C_j = (sum_Mxy(:,:,j).')/sum_MP(:,:,j);
            if any(isnan(C_j(:)))|| any(isinf(C_j(:)))
                C_j = (sum_Mxy(:,:,j).') * pinv(sum_MP(:,:,j));
           end
            Chat(:,:,j) = C_j;
        end
    end
else
    if equal.C
        sum_xy = sum(sum_Mxy,3);
        if ~isempty(fixed.C)
            idx = fixed.C(:,1) <= N*r;
        	fixed.C = fixed.C(idx,:);
        end
        Chat = PG_C(pars.C(:,:,1),sum_xy,sum_P,pars.R,scale.C,fixed.C);
        Chat = repmat(Chat,1,1,M);
    else
        Chat = zeros(N,r,M);
        for j=1:M
            C_j = pars.C(:,:,j);
            sum_Mxyj = sum_Mxy(:,:,j);
            sum_MPj = sum_MP(:,:,j);
            fixed_Cj = [];
            if ~isempty(fixed.C)
                idx = (fixed.C(:,1) > (j-1)*N*r) & (fixed.C(:,1) <= j*N*r);
                fixed_Cj = fixed.C(idx,:);
            end
            Chat(:,:,j) = PG_C(C_j,R,sum_yy,sum_Mxyj,sum_MPj,scale.C,fixed_Cj);
        end
    end
    
    % Check that parameter update actually increases Q-function 
    % If not, keep previous parameter estimate 
    Qval_old = Q_obs(outpars,Ms,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,...
        sum_P,sum_Pb,sum_yy,x0);
    outpars.C = Chat;
    Qval = Q_obs(outpars,Ms,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,...
        sum_P,sum_Pb,sum_yy,x0);
    if Qval < Qval_old
        outpars.C = pars.C;
    end
end




%=========================================================================%
%                               Update Q                                  %
%=========================================================================%



% Unconstrained solution
if ~skip.Q 
    Qhat = zeros(r,r,M);
    for j=1:M
        A_j = Ahat(1:r,:,j);
        sum_Pj = sum_P(:,:,j);
        sum_CPj = sum_CP(:,:,j);
        sum_Pbj = sum_Pb(:,:,j);                
        Q_j = (sum_Pj - (sum_CPj * A_j.') - ...
            (A_j * sum_CPj.') + A_j * sum_Pbj * A_j.') / (T-1);
        Qhat(:,:,j) = 0.5 * (Q_j + Q_j.');    
    end
    % Apply equality constraints 
    if equal.Q
        Qhat = repmat(mean(Qhat,3),1,1,M);
    end
    % Enforce fixed coefficient constraints
    if ~isempty(fixed.Q)
        Qhat(fixed.Q(:,1)) = fixed.Q(:,2);
    end

    % Regularize estimate if needed
    for j = 1:M
        eigval = eig(Qhat(:,:,j));
        if min(eigval) < max(abstol,max(eigval)*reltol)
            if verbose 
                warning(['Q%d ill-conditioned and/or nearly singular.', ... 
                    ' Regularizing.'],j);
            end
            Qhat(:,:,j) = regfun(Qhat(:,:,j),abstol,reltol);
        end
        if equal.Q
            Qhat = repmat(Qhat(:,:,1),[1,1,M]);
            break
        end
    end

    % Apply fixed coefficient constraints
    if ~isempty(fixed.Q)
        Qhat(fixed.Q(:,1)) = fixed.Q(:,2);
    end  

    % Check that estimate Qhat increases Q-function. If not, keep
    % estimate from previous iteration
    Qval_old = Q_obs(outpars,Ms,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,...
        sum_P,sum_Pb,sum_yy,x0);
    outpars.Q = Qhat;
    Qval = Q_obs(outpars,Ms,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,...
        sum_P,sum_Pb,sum_yy,x0);
    if Qval < Qval_old
        outpars.Q = pars.Q;
    end
end




%=========================================================================%
%                               Update R                                  %
%=========================================================================%



if ~skip.R
    Rhat = sum_yy;
    for j=1:M
        C_j = Chat(:,:,j);
        sum_MPj = sum_MP(:,:,j);
        sum_Mxyj = sum_Mxy(:,:,j);
        Rhat = Rhat -  (C_j*sum_Mxyj) - (C_j*sum_Mxyj)' + (C_j*sum_MPj*C_j');
    end
    Rhat = Rhat / T;
    Rhat = 0.5 * (Rhat+Rhat');
    Rhat(fixed.R(:,1)) = fixed.R(:,2);
    % Regularize R if needed
    eigval = eig(Rhat);
    if min(eigval) < max(abstol,max(eigval)*reltol)
        if verbose
            warning('R ill-conditioned and/or nearly singular. Regularizing.');
        end
        Rhat = regfun(Rhat,abstol,reltol);
        Rhat(fixed.R(:,1)) = fixed.R(:,2);
    end 
    % Make sure that parameter update increases Q-function
    % If not, do not update parameter estimate 
    Qval_old = Q_obs(outpars,Ms,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,...
        sum_P,sum_Pb,sum_yy,x0);
    outpars.R = Rhat;
    Qval = Q_obs(outpars,Ms,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,...
        sum_P,sum_Pb,sum_yy,x0);
    if Qval < Qval_old
        outpars.R = pars.R;
    end
end



%=========================================================================%
%                               Update mu                                 %
%=========================================================================%


if skip.mu
    muhat = mu;
else
    muhat = reshape(x0,r,p,M); % unconstrained solution
    muhat = reshape(mean(muhat,2),r,M); 
    % Assume E(x(1,j))=E(x(0,j))=...=E(x(1-p+1,j) for j=1:M
    if equal.mu
        muhat = repmat(mean(muhat,2),1,M);
    end
end

if ~isempty(fixed.mu)
    muhat(fixed.mu(:,1)) = fixed.mu(:,2);
    Qval_old = Q_obs(outpars,Ms,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,...
        sum_P,sum_Pb,sum_yy,x0);
    outpars.mu = muhat;
    Qval = Q_obs(outpars,Ms,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,...
        sum_P,sum_Pb,sum_yy,x0);
    if Qval < Qval_old
        outpars.mu = pars.mu;
    end
end



%=========================================================================%
%                               Update Sigma                              %
%=========================================================================%



if ~skip.Sigma
    Sigmahat = zeros(r,r,M);
    for j = 1:M
        mu_j = repmat(muhat(:,j),p,1); % replicate mu(j) to size pr x 1
        B_j = P0(:,:,j) - x0(:,j) * mu_j.' - ...
            mu_j * x0(:,j).' + (mu_j * mu_j.');        
        S_j = mean(B_j(Sigmamask),3);
        Sigmahat(:,:,j) = 0.5 * (S_j+S_j.');
    end

    if equal.Sigma
        Sigmahat = repmat(mean(Sigmahat,3),1,1,M);
    end
    Sigmahat(fixed.Sigma(:,1)) = fixed.Sigma(:,2);

    % Enforce semi-positive definiteness if needed 
    for j = 1:M
        S_j = Sigmahat(1:r,1:r,j);
        eigval = eig(S_j);
        if ~all(eigval >= 0) 
            if verbose
                warning('Sigma%d non semi-positive definite. Regularizing.',j);
            end
        Sigmahat(:,:,j) = regfun(S_j,0,0);
        end
    end
    Sigmahat(fixed.Sigma(:,1)) = fixed.Sigma(:,2);

    % Make sure that parameter update increases Q-function
    % If not, do not update parameter estimate 
    Qval_old = Q_obs(outpars,Ms,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,...
        sum_P,sum_Pb,sum_yy,x0);
    outpars.Sigma = Sigmahat;
    Qval = Q_obs(outpars,Ms,P0,sum_CP,sum_MP,sum_Ms2,sum_Mxy,...
        sum_P,sum_Pb,sum_yy,x0);
    if Qval < Qval_old
        outpars.Sigma = pars.Sigma;
    end
end



%=========================================================================%
%                               Update Pi                                 %
%=========================================================================%


if ~skip.Pi
    outpars.Pi = Ms(:,1);
end
    
    


%=========================================================================%
%                               Update Z                                  %
%=========================================================================%



if ~skip.Z
    Zhat = sum_Ms2 ./ repmat(sum(sum_Ms2,2),1,M); 
    if ~isempty(fixed.Z)
        Zhat(fixed.Z(:,1)) = fixed.Z(:,2);
    end
    outpars.Z = Zhat;
end


