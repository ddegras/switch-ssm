function outpars = M_dyn(pars,MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,...
    sum_Ms2,sum_P,sum_xy,sum_yy,control,equal,fixed,scale,skip)

    
abstol = control.abstol;
reltol = control.reltol;
verbose = control.verbose;
outpars = pars;
[~,r,M] = size(pars.Q);
p = size(pars.A,2) / size(pars.A,1);
T = size(Ms,2);


% Mask of rxr diagonal blocks in a (p*r)x(p*r) matrix
% (used to update Sigma in M-step)
Sigmamask = reshape(find(kron(eye(p),ones(r))),r,r,p);


% Unconstrained parameter estimates 
% A = ( sum(t=p+1:T) P(t,t-1|T) ) * ( sum(t=p+1:T) P~(t-1|T) )^{-1}
% C = (sum(t=1:T) y(t) x(t|T)') * (sum(t=1:T) P(t|T))^{-1}
% Qj = (sum(t=2:T) Wj(t) Pj(t) - Aj * sum(t=2:T) Wj(t) Pj(t-1,t)') / sum(t=2:T) Wj(t)
% R = sum(t=1:T) y(t) y(t)' / T - sum(t=1:T) x(t|T) y(t)'
%
% where Wj(t) = P(S(t)=j|y(1:T)), 
% xj(t|T) = E(x(t)|S(t)=j,y(1:T)), 
% x(t|T) = E(x(t)|y(1:T)), 
% Pj(t|T) = E(x(t)x(t)'|S(t)=j,y(1:T)), 
% P~(t-1|T) = E(x(t-1)x(t-1)'|S(t)=j,y(1:T))
% and P(t,t-1|T) = E(x(t)x(t-1)'|y(1:T))


%=========================================================================%
%                               Update A                                  %
%=========================================================================%



% Case: no fixed coefficient constraints
if skip.A 
    Ahat = reshape(pars.A,r,p*r,M);
end
if isempty(fixed.A)
    if equal.A && equal.Q
        sum_Pb = sum(sum_MPb,3);
        sum_CP = sum(sum_MCP,3);
        Ahat = sum_CP / sum_Pb;
        if any(isnan(Ahat(:))|isinf(Ahat(:)))
            Ahat = sum_CP * pinv(sum_Pb);
        end
        Ahat = repmat(Ahat,[1,1,M]);
    elseif equal.A
        % If the A's are all equal but the Q's are not, there is no closed
        % form expression for the A and Q's that maximize the Q function.
        % In this case, fix the Q's and find the best associated A (ECM) 
        lhs = zeros(p*r,p*r);
        rhs = zeros(r,p*r);
        for j=1:M
            Qinv_j = myinv(pars.Q(:,:,j));         
            lhs = lhs + kron(sum_MPb(:,:,j),Qinv_j);
            rhs = rhs + Qinv_j * sum_MCP(:,:,j);
        end
        rhs = rhs(:);
        Ahat = reshape(lhs\rhs,r,p*r);
        if any(isnan(Ahat(:))|isinf(Ahat(:)))
            Ahat = reshape(pinv(lhs)*rhs,r,p*r);
        end
        Ahat = repmat(Ahat,[1,1,M]);
    else
        Ahat = zeros(r,p*r,M);
        for j=1:M
            A_j = sum_MCP(:,:,j) / sum_MPb(:,:,j);
            if any(isnan(A_j(:)) | isinf(A_j(:)))
                 A_j = sum_MCP(:,:,j) * pinv(sum_MPb(:,:,j));
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
    pr2 = p*r^2;
    for j = 1:M
        % Linear indices of free coefficients in A(j)
        idx = (fixed.A(:,1) > (j-1)*pr2) & (fixed.A(:,1) <= j*pr2); 
        fixed_Aj = fixed.A(idx,:);
        fixed_Aj(:,1) = fixed_Aj(:,1) - (j-1)*pr2;
        free = setdiff(1:pr2,fixed_Aj(:,1));
        free = free(:);
        Qinv_j = myinv(pars.Q(:,:,j));
        % Matrix problem min(X) trace(W(-2*B1*X' + X*B2*X')) 
        % (under fixed coefficient constraints) becomes vector problem 
        % min(x) x' kron(B2,W) x - 2 x' vec(W*B1)
        % with X = A(j), x = vec(A(j)), W = Q(j)^(-1), B1 = sum_MCP(j),
        % and B2 = sum_MPb(j) (remove fixed entries in x)
        mat = kron(sum_MPb(:,:,j),Qinv_j);
        vec = reshape(Qinv_j * sum_MCP(:,:,j),pr2,1);            
        A_j = zeros(pr2,1);
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
            % Case: regularize with no fixed coefficients or all fixed
            % coefficients equal to zero
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
    Qvalold = Q_dyn(outpars,MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,...
        sum_P,sum_xy,sum_yy);
    outpars.A = Ahat;
    Qval = Q_dyn(outpars,MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,...
        sum_P,sum_xy,sum_yy);
    if Qval < Qvalold
        outpars.A = pars.A;
    end
end




%=========================================================================%
%                               Update C                                  %
%=========================================================================%



if ~skip.C
    % Case: no fixed coefficient and/or scale constraints on C
    % Calculate estimate in closed form
    if isempty(fixed.C) && isempty(scale.C)
        Chat = (sum_xy.') / sum_P;
        if any(isnan(Chat(:)) | isinf(Chat(:)))
            Chat = sum_xy.' * pinv(sum_P);
        end
    else
    % Otherwise: perform constrained estimation by projected gradient
        Chat = PG_C(pars.C,sum_xy,sum_P,pars.R,scale.C,fixed.C);
    end

    % Check that parameter update actually increases Q-function. If
    % not, keep previous parameter estimate. (This is a redundancy
    % check: by design, the above constrained and unconstrained
    % estimates cannot decrease the Q-function)
    Qvalold = Q_dyn(outpars,MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,...
        sum_P,sum_xy,sum_yy);
    outpars.C = Chat;
    Qval = Q_dyn(outpars,MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,...
        sum_P,sum_xy,sum_yy);
    if Qval < Qvalold
        outpars.C = pars.C;
    end
end




%=========================================================================%
%                               Update Q                                  %
%=========================================================================%




% Unconstrained solution
if ~skip.Q 
    if equal.Q
        Qtmp = zeros(r,r,M);
        for j=1:M
            A_j = outpars.A(1:r,:,j);
            sum_MPj = sum_MP(:,:,j);
            sum_MCPj = sum_MCP(:,:,j);
            sum_MPbj = sum_MPb(:,:,j);                
            Qtmp(:,:,j) = sum_MPj - (sum_MCPj * A_j.') - ...
                (A_j * sum_MCPj.') + A_j * sum_MPbj * A_j.';
        end
        Qtmp = sum(Qtmp,3) / (T-1);
        Qtmp = 0.5 * (Qtmp + Qtmp.');
        Qhat = repmat(Qtmp,1,1,M);
    else
        Qhat = zeros(r,r,M); 
        for j=1:M
            sum_Mj = sum(Ms(j,2:T));
            sum_MPj = sum_MP(:,:,j);
            sum_MCPj = sum_MCP(:,:,j);
            sum_MPbj = sum_MPb(:,:,j);
            A_j = outpars.A(:,:,j);
            if sum_Mj > 0
                Q_j = (sum_MPj - (A_j * sum_MCPj') - (sum_MCPj * A_j') ...
                    + (A_j * sum_MPbj * A_j')) / sum_Mj;
            else
                Q_j = eye(r);
            end
            Q_j = 0.5 * (Q_j + Q_j');
            Qhat(:,:,j) = Q_j;
        end
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
    Qvalold = Q_dyn(outpars,MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,...
        sum_P,sum_xy,sum_yy);
    outpars.Q = Qhat;
    Qval = Q_dyn(outpars,MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,...
        sum_P,sum_xy,sum_yy);
    if Qval < Qvalold
        outpars.Q = pars.Q;
    end        
end




%=========================================================================%
%                               Update R                                  %
%=========================================================================%




if ~skip.R
    % Unconstrained solution
    Rhat = (sum_yy - outpars.C * sum_xy - (outpars.C * sum_xy)' + ...
        outpars.C * sum_P * outpars.C') / T;
    Rhat = 0.5 * (Rhat + Rhat');
    % Apply fixed coefficient constraints
    if ~isempty(fixed.R)
        Rhat(fixed.R(:,1)) = fixed.R(:,2);
    end
    % Check positive definiteness and conditioning of Rhat. Regularize
    % if needed
    eigval = eig(Rhat);
    if min(eigval) < max(abstol,max(eigval)*reltol)
        if verbose
            warning('R ill-conditioned and/or nearly singular. Regularizing.');
        end
        Rhat = regfun(Rhat,abstol,reltol); 
        if ~isempty(fixed.R)
            Rhat(fixed.R(:,1)) = fixed.R(:,2);
        end
    end 

    % Check that Rhat increases Q-function. If not, keep estimate from
    % previous iteration
    Qvalold = Q_dyn(outpars,MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,...
        sum_P,sum_xy,sum_yy);
    outpars.R = Rhat;
    Qval = Q_dyn(outpars,MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,...
        sum_P,sum_xy,sum_yy);
    if Qval < Qvalold
        outpars.R = pars.R;
    end
end




%=========================================================================%
%                               Update mu                                 %
%=========================================================================%




if ~skip.mu
    sum_Mx0 = reshape(sum(reshape(Mx0,[r,p,M]),2),[r,M]); 
    if equal.mu && equal.Sigma
        muhat = sum(sum_Mx0,2)/p;
        muhat = repmat(muhat,1,M);
    elseif equal.mu
        lhs = zeros(r,r);
        rhs = zeros(r,1);
        for j=1:M
            Sinv_j = myinv(pars.Sigma(:,:,j));         
            lhs = lhs + (p * Ms(j,1)) * Sinv_j;
            rhs = rhs + Sinv_j * sum_Mx0(:,j);
        end
        muhat = lhs\rhs;
        if any(isnan(muhat) | isinf(muhat))
            muhat = myinv(lhs)*rhs;
        end
        muhat = repmat(muhat,1,M);        
    else
        muhat = zeros(r,M);
        for j = 1:M
            if Ms(j,1) > 0
                muhat(:,j) = sum_Mx0(:,j) / Ms(j,1);
            end
        end            
    end

    % Apply fixed coefficient constraints
    if ~isempty(fixed.mu)
        muhat(fixed.mu(:,1)) = fixed.mu(:,2);
    end

    % Check that muhat increases Q-function. If not, keep estimate from
    % previous iteration
    Qvalold = Q_dyn(outpars,MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,...
        sum_P,sum_xy,sum_yy);
    outpars.mu = muhat;
    Qval = Q_dyn(outpars,MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,...
        sum_P,sum_xy,sum_yy);
    if Qval < Qvalold
        outpars.mu = pars.mu;
    end
end





%=========================================================================%
%                            Update Sigma                                 %
%=========================================================================%




if ~skip.Sigma
    mubig = repmat(outpars.mu,[p,1]);
    % Unconstrained solution
    if equal.Sigma
        Stmp = sum(MP0,3) - (mubig * Mx0') - (Mx0 * mubig') + ...
            (mubig * diag(Ms(:,1)) * mubig'); % dimension (p*r)x(p*r)
        Sigmahat = mean(Stmp(Sigmamask),3); % dimension rxr 
        Sigmahat = 0.5 * (Sigmahat + Sigmahat'); % symmetrize
        Sigmahat = repmat(Sigmahat(:,:,1),[1,1,M]); % replicate
    else
        Sigmahat = zeros(r,r,M);
        for j = 1:M
            if Ms(j,1) > 0
                S_j = MP0(:,:,j) - (mubig(:,j) * Mx0(:,j)') - ...
                    (Mx0(:,j) * mubig(:,j)') + Ms(j,1) * (mubig(:,j) * mubig(:,j)');             
            S_j = mean(S_j(Sigmamask),3) / Ms(j,1); 
            S_j = 0.5 * (S_j + S_j'); 
            else
                S_j = eye(r);
            end
            Sigmahat(:,:,j) = S_j; 
        end
    end

    % Enforce any fixed coefficient constraints
    if ~isempty(fixed.Sigma)
        Sigmahat(fixed.Sigma(:,1)) = fixed.Sigma(:,2);
    end

    % Regularize estimate if needed
    for j = 1:M
        eigval = eig(Sigmahat(:,:,j));
        if min(eigval) < max(abstol,max(eigval)*reltol)
            if verbose
                warning(['Sigma%d ill-conditioned and/or nearly singular.', ...
                    ' Regularizing.'],j);
            end
            Sigmahat(:,:,j) = regfun(Sigmahat(:,:,j),abstol,reltol); 
        end 
        if equal.Sigma
            Sigmahat = repmat(Sigmahat(:,:,1),[1,1,M]);
            break
        end
    end
    % Enforce fixed coefficient constraints 
    if ~isempty(fixed.Sigma)
        Sigmahat(fixed.Sigma(:,1)) = fixed.Sigma(:,2);
    end

    % Check that Sigmahat increases Q-function. If not, keep 
    % parameter estimate from previous iteration
    Qvalold = Q_dyn(outpars,MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,...
        sum_P,sum_xy,sum_yy);
    outpars.Sigma = Sigmahat;
    Qval = Q_dyn(outpars,MP0,Ms,Mx0,sum_MCP,sum_MP,sum_MPb,sum_Ms2,...
        sum_P,sum_xy,sum_yy);
    if Qval < Qvalold
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
    outpars.Z = sum_Ms2 ./ repmat(sum(sum_Ms2,2),1,M);
    if ~isempty(fixed.Z)
        outpars.Z(fixed.Z(:,1)) = fixed.Z(:,2);
    end
end

    