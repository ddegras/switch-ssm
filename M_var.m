function outpars = M_var(pars,Ms,sum_MCP,sum_MP,sum_MPb,...
    sum_Ms2,y,control,equal,fixed,scale,skip)

    
abstol = control.abstol;
reltol = control.reltol;
verbose = control.verbose;
outpars = pars;
[M,T] = size(Ms);
r = size(pars.A,1);
p = size(pars.A,2) / r;




%=========================================================================%
%                               Update A                                  %
%=========================================================================%


% Case: no fixed coefficient constraints
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
    pr2 = p*r*r;
    Ahat = zeros(r,p*r,M);
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
        vec = Qinv_j * sum_MCP(:,:,j);            
        A_j = zeros(pr2,1);
        A_j(fixed_Aj(:,1)) = fixed_Aj(:,2);
        A_j(free) = mat(free,free)\vec(free);
        if any(isnan(A_j)|isinf(A_j))
            A_j(free) = pinv(mat(free,free)) * vec(free);
        end
        Ahat(:,:,j) = reshape(A_j,r,p*r);
    end
end

% Check eigenvalues of estimate and regularize if needed    
if ~skip.A
    for j = 1:M
        % Check eigenvalues
        Abig(1:r,:,:) = Ahat(:,:,j);
        eigval = eig(Abig);
        if any(abs(eigval) > scale.A)
            if verbose
                warning(['Eigenvalues of A%d greater than %f.',...
                    ' Regularizing.'],j,scale.A)
            end
            % Case: regularize with no fixed coefficients constraints
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
    Qvalold = Q_var(outpars,Ms,sum_MCP,sum_MP,sum_MPb,sum_Ms2,y);
    outpars.A = Ahat; 
    Qval = Q_var(outpars,Ms,sum_MCP,sum_MP,sum_MPb,sum_Ms2,y);
    if Qval < Qvalold
        outpars.A = pars.A;
    end
end



%=========================================================================%
%                               Update Q                                  %
%=========================================================================%


% Unconstrained solution
if ~skip.Q        
    Qhat = zeros(r,r,M);
    sum_M = sum(Ms(:,p+1:T),2);
    for j=1:M
        if sum_M(j) == 0
            Qhat(:,:,j) = eye(r);
            continue
        end                
        A_j = outpars.A(:,:,j);
        sum_MPj = sum_MP(:,:,j);
        sum_MCPj = sum_MCP(:,:,j);
        sum_MPbj = sum_MPb(:,:,j);                
        Q_j = (sum_MPj - (sum_MCPj * A_j.') - ...
            (A_j * sum_MCPj.') + A_j * sum_MPbj * A_j.') / sum_M(j);
        Qhat(:,:,j) = 0.5 * (Q_j + Q_j');
    end
    if equal.Q
        Qtmp = zeros(r);
        for j = 1:M
            Qtmp = Qtmp + (sum_M(j)/(T-p)) * Qhat(:,:,j);
        end
        Qhat = repmat(Qtmp,1,1,M);
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
    Qvalold = Q_var(outpars,Ms,sum_MCP,sum_MP,sum_MPb,sum_Ms2,y);
    outpars.Q = Qhat;
    Qval =  Q_var(outpars,Ms,sum_MCP,sum_MP,sum_MPb,sum_Ms2,y);
    if Qval < Qvalold
        outpars.Q = pars.Q;
    end        
end



%=========================================================================%
%                               Update mu                                 %
%=========================================================================%


if ~skip.mu    
    if equal.mu && equal.Sigma
        muhat = repmat(mean(y(:,1:p),2),1,M);
    elseif equal.mu
        lhs = zeros(r,r);
        rhs = zeros(r,1);
        for j=1:M
            Sinv_j = myinv(pars.Sigma(:,:,j));         
            lhs = lhs + Sinv_j;
            rhs = rhs + Sinv_j * y(:,1:p) * Ms(j,1:p)';
        end
        muhat = lhs\rhs;
        if any(isnan(muhat) | isinf(muhat))
            muhat = pinv(lhs)*rhs;
        end
        muhat = repmat(muhat,1,M);        
    else
        muhat = zeros(r,M);
        for j = 1:M
            sum_Mj = sum(Ms(j,1:p));                
            if sum_Mj > 0
                muhat(:,j) = (y(:,1:p) * Ms(j,1:p)') / sum_Mj;
            end
        end            
    end

    % Apply fixed coefficient constraints
    if ~isempty(fixed.mu)
        muhat(fixed.mu(:,1)) = fixed.mu(:,2);
    end

    % Check that muhat increases Q-function. If not, keep estimate from
    % previous iteration
    Qvalold = Q_var(outpars,Ms,sum_MCP,sum_MP,sum_MPb,sum_Ms2,y);
    outpars.mu = muhat;
    Qval = Q_var(outpars,Ms,sum_MCP,sum_MP,sum_MPb,sum_Ms2,y);
    if Qval < Qvalold
        outpars.mu = pars.mu;
    end
end



%=========================================================================%
%                               Update Sigma                              %
%=========================================================================%


if ~skip.Sigma
    Sigmahat = zeros(r,r,M);
    % sum(t=1:p) P(S(t)=j|y(1:T))
    sum_M = sum(Ms(:,1:p),2);
    for j = 1:M
        if sum_M(j) == 0
            Sigmahat(:,:,j) = eye(r);
            continue
        end
        B_j = y(:,1:p) - repmat(outpars.mu(:,j),1,p);
        S_j = (B_j * diag(Ms(j,1:p)) * B_j') / sum_M(j);
        Sigmahat(:,:,j) = 0.5 * (S_j + S_j'); 
    end
    if equal.Sigma
        Stmp = zeros(r);
        for j = 1:M
            Stmp = Stmp + (sum_M(j)/p) * Sigmahat(:,:,j);
        end
        Sigmahat = repmat(Stmp,1,1,M);
    end

    % The above estimates of Sigma(j) have rank p at most (VAR order). 
    % If p < r (time series dimension), they are not invertible --> 
    % set off-diagonal terms to zero
    if p < r
        for j = 1:M
            Sigmahat(:,:,j) = diag(diag(Sigmahat(:,:,j)));
        end
    end

    % Enforce fixed coefficient constraints
    if ~isempty(fixed.Sigma)
        Sigmahat(fixed.Sigma(:,1)) = fixed.Sigma(:,2);
    end

    % Regularize estimate if needed
    for j = 1:M
        eigval = eig(Sigmahat(:,:,j));
        if min(eigval) < max(abstol,max(eigval)*reltol)
%                 if verbose
%                     warning(['Sigma%d ill-conditioned and/or nearly singular.', ...
%                         ' Regularizing.'],j);
%                 end
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
    Qvalold = Q_var(outpars,Ms,sum_MCP,sum_MP,sum_MPb,sum_Ms2,y);
    outpars.Sigma = Sigmahat;
    Qval = Q_var(outpars,Ms,sum_MCP,sum_MP,sum_MPb,sum_Ms2,y);
    if Qval < Qvalold
        outpars.Sigma = pars.Sigma;
    end
end



%=========================================================================%
%                               Update Pi                                 %
%=========================================================================%

if ~skip.Pi
    outpars.Pi =  Ms(:,1);
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


