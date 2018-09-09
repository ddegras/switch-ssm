function [C,err] = PG_C(C0,sum_xy,sum_P,R,scale,fixed,maxit,tol)
%--------------------------------------------------------------------------
%
%    ESTIMATION OF OBSERVATION MATRIX IN SWITCHING STATE-SPACE MODEL
%           UNDER FIXED COEFFICIENT AND/OR SCALE CONSTRAINTS 
%
% GOAL
% Minimize trace( R^(-1) (C sum_P C' - 2 sum_xy' C') ) under scale constraints 
% and fixed coefficient constraints on C by projected gradient method
%
% USAGE
% [C,err] = PG_quad_scale(C0,sum_P,sum_xy,R,scale,fixed,maxit,tol)
% 
% INPUTS
% C0:       Starting value for projected gradient method
% sum_P:    Matrix defining objective function (must be positive definite)
% sum_xy:   Matrix defining objective function 
% R:        Matrix defining objective function (must be positive definite)
% scale:    the columns of C must have Euclidean norm equal to 'scale' 
% fixed:    optional matrix with two columns. If specified, 1st column must
%           contain indices of fixed coefficients in C, 2nd column must contain
%           corresponding fixed values
% maxit:    maximum number of iterations in algorithm, default = 1e4
% tol:      tolerance for convergence (algorithm terminates if relative change 
%           in objective function between two successive iterations is less 
%           than tol). Default = 1e-6
%
% OUTPUT
% C:        best solution satisfying the problem constraints if any found, otherwise starting value 
% err:      true if a solution satisfying the problem constraints has been found, false otherwise 
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

% Logical flags for constraints
fixed_cnstr = ~isempty(fixed);
scale_cnstr = ~isempty(scale);

% Tolerance for testing scale equality 
if scale_cnstr
    eps = 1e-8 * min(scale,1);
end

% Starting point
C = C0;

% Objective function value
objective = zeros(maxit,1);
Rinv = myinv(R);
sum_yx = sum_xy';
c = trace(sum_P\(sum_yx'*R*sum_yx)); 
% Add constant c to objective function to make it always non-negative

% Check whether problem constraints are met. If yes, calculate objective. 
% Otherwise, set objective to Inf.
fixed_test = ~fixed_cnstr || all(C(fixed(:,1)) == fixed(:,2));
scale_test = ~scale_cnstr || all(abs(sqrt(sum(C.^2))-scale) <= eps);
if ~(fixed_test && scale_test)
    objective_best = Inf;
else
    objective_best = trace(R*(C*sum_P-2*sum_yx)*C') + c;
end
% Best solution to date
Cbest = C;

% Lipschitz constant of gradient of objective function (up to factor 2)
beta = norm(sum_P,'fro') * norm(Rinv,'fro');



%@@@@@ MAIN LOOP @@@@@%

for i = 1:maxit
    
    % Gradient step
    grad = Rinv * (C * sum_P - sum_yx);
    C = C - (1/beta) * grad;
    
    % (Alternating) projection step
    for j = 1:maxit
        % Apply fixed coefficient constraints 
        if fixed_cnstr
            C(fixed(:,1)) = fixed(:,2);
        end
        % Check scale constraint, terminate projection step if true 
        % and apply it otherwise
        if scale_cnstr
            nrmC = sqrt(sum(C.^2));
            scale_test = all(abs(nrmC-scale) <= eps);
            if scale_test 
                break
            else
                C = C ./ nrmC * scale;
            end
        end
    end
    
    % Calculate objective
    % Note: since projection step guarantees that scale constraint is met, 
    % no need to check it again here 
    fixed_test = ~fixed_cnstr || all(C(fixed(:,1)) == fixed(:,2));
    if ~fixed_test 
        objective(i) = Inf;
    else
        objective(i) = trace(R*(C*sum_P-2*sum_yx)*C') + c;
    end
    
    % Update best solution 
    if objective(i) < objective_best
        Cbest = C;
        objective_best = objective(i);
    end
    
    % Monitor convergence
    if i >= 10 
        test1 = abs(objective(i) - objective(i-1)) <= tol * objective(i-1); % convergence attained
        test2 = objective(i) == 0; % best solution (=unconstrained) attained 
        if test1 || test2
            break
        end
    end
end

C = Cbest;
err = isinf(objective_best);

    