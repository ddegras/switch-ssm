function [A,err] = PG_A(A0,sum_CP,sum_Pb,Q,scale,fixed,maxit,tol)
%--------------------------------------------------------------------------
%     ESTIMATION OF TRANSITION MATRICES IN VECTOR AUTOREGRESSIVE MODEL 
%           EIGENVALUE AND UNDER FIXED COEFFICIENT CONSTRAINTS
% 
% PURPOSE 
% Consider a vector autoregressive (VAR) model of order p: 
% x(t) = A(1) x(t-1) + ... + A(p) x(t-p) + v(t) with x(t) of dimension r 
% and v(t)~ N(0,Q). Write A = (A(1),...,A(p)) size rx(p*r)) and define 
% X(t) = (x(t),...,x(t-p)) so that x(t) = A X(t-1) + v(t). Assume Q known. 
% To estimate A, consider minimizing the objective function 
% F(A) = E(sum(t=2:T) (x(t) - A X(t-1))' Q^(-1) (x(t) - A X(t-1))
% It holds that F(A) = trace{Q^(-1) (sum_P - 2 sum_CP A' + A sum_Pb A')} where 
% sum_P = sum(t=2:T) E(x(t)x(t'), sum_CP = sum(t=2:T) E(x(t)X(t-1)'), and sum_Pb =
% sum(t=2:T) E(X(t-1)X(t-1)'). In practice the matrices sum_P, sum_CP, sum_Pb
% can be replaced by empirical estimates. The function PG_A minimizes the 
% objective function F(A) under fixed coefficient constraints on A and the
% requirement that the VAR process x(t) is invertible, which amounts to an 
% eigenvalue condition on a matrix related to A. Projected gradient is used
% to carry out the minimization.  
%
% USAGE
% A = PG_A(A0,sum_CP,sum_Pb,Q,scale,fixed,maxit,tol)
%
% INPUTS
% A0:       Starting value for projected gradient method
% sum_CP,sum_Pb,Q:  Matrices defining objective function (A,W should be positive definite)
% scale:    Upper bound on the (norms of the) eigenvalues associated with A
%           (should be less than 1 to guarantee an invertible VAR process) 
% fixed:    Pptional matrix with two columns. If specified, 1st column must
%           contain indices of fixed coefficients in X, 2nd column must contain
%           corresponding fixed values
% maxit:    Maximum number of iterations in algorithm, default = 1e4
% tol:      Tolerance for convergence (algorithm terminates if relative change 
%           in objective function between two successive iterations is less 
%           than tol). Default = 1e-6
%
% OUTPUT
% A:        Best solution satisfying the problem constraints
%--------------------------------------------------------------------------


%@@@@@ Initialization @@@@@%

% Check number of input arguments
narginchk(6,8);

% Set control parameters to default values if not specified
if nargin == 6
    maxit = 1e4;
end
if nargin <= 7
    tol = 1e-6;
end

% Logical flag for fixed constraint
fixed_cnstr = ~isempty(fixed);

% Working scale: original scale minus very small number to allow for
% numerical error in eigenvalue calculations
w_scale = scale - eps;

% Starting point
A = A0;

% Embedding matrix used to evaluate eigenvalue condition (size (p*r)x(p*r)) 
r = size(sum_CP,1);
p = size(sum_CP,2)/r;
Abig = diag(ones((p-1)*r,1),-r);
Abig(1:r,:) = A;

% Objective function value
objective = zeros(maxit,1);
Qinv = myinv(Q);
c = trace(sum_Pb\(sum_CP'*Qinv*sum_CP)); 
% add the constant c to objective function to make it non-negative

% Check whether problem constraints are met. If yes, calculate objective. 
% Otherwise, set objective to Inf.
fixed_test = ~fixed_cnstr || all(A(fixed(:,1)) == fixed(:,2));
if ~fixed_test 
    objective_best = Inf;
else
    eig_test = all(abs(eig(Abig)) <= scale);
    if ~eig_test 
        objective_best = Inf;
    else
        objective_best = trace(Qinv*(A*sum_Pb-2*sum_CP)*A') + c;
    end
end
% Best solution to date
Abest = A;

% Lipschitz constant of gradient of objective function (up to factor 2)
beta = norm(A,'fro') * norm(Qinv,'fro');



%@@@@@ MAIN LOOP @@@@@%

for i = 1:maxit
    
    % Gradient step
    grad = Qinv * (A * sum_Pb - sum_CP);
    stepsize = min(1/beta,norm(diag(grad),Inf)/100);
    A = A - stepsize * grad;
    
    % (Alternating) projection step
    for j = 1:maxit
        % Apply fixed coefficient constraint 
        if fixed_cnstr
            A(fixed(:,1)) = fixed(:,2);
        end
        % Check eigenvalue constraint, stop if true, apply it otherwise
        Abig(1:r,:) = A;
        [P,D] = eig(Abig,'vector');
        valid = (abs(D) <= scale);
        if all(valid)
            break
        else
            D(~valid) = D(~valid) ./ abs(D(~valid)) * (w_scale);
            A = real((P(1:r,:) * diag(D)) / P);
        end
    end
    
    % Calculate objective
    % Note: since projection step guarantees eigenvalue constraint is met,
    % no need to check it again here
    fixed_test = ~fixed_cnstr || all(A(fixed(:,1)) == fixed(:,2));
    if ~fixed_test 
        objective(i) = Inf;
    else
        objective(i) = trace(Qinv*(A*sum_Pb-2*sum_CP)*A') + c;
    end
    
    % Update best solution 
    if objective(i) < objective_best
        Abest = A;
        objective_best = objective(i);
    end
    
    % Monitor convergence
    if i >= 10 
        test1 = (abs(objective(i) - objective(i-1)) <= tol * objective(i-1)); % convergence attained
        test2 = (objective(i) == 0); % global (=unconstrained) minimizer attained 
        if test1 || test2
            break
        end
    end
end

A = Abest;
err = isinf(objective_best);

    