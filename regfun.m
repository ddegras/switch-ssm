% OLD VERSION: 
function out = regfun(x,abstol,reltol)
[V,d] = eig(x,'vector');
lb = max(abstol,reltol*max(d));
good = (d >= lb);
if all(good)
    out = x;
else
    d(~good) = lb;
    out = V * diag(d) * V.';
    if ~ issymmetric(out)
        out = (out+out.')/2;
    end
end

% NEW VERSION: add a multiple of the identity matrix to improve
% conditioning

% Input: 
% x: symmetric square matrix 
% abstol: smallest eigenvalue of regularized matrix
% inverse of maximum condition number of regularized matrix

% function out = regfun(x,abstol,reltol)
% [v,d] = eig(x,'vector');
% out = x;
% if any(d < abstol)
%     d(d < abstol) = abstol;
%     out = v * diag(d) * v.';
%     if ~issymmetric(out)
%         out = 0.5 * (out+out.');
%     end
% end
% dmax = max(d);
% dmin = min(d);
% lam = max(abstol-dmin,(reltol*dmax-dmin)/(1-reltol));
% if lam > 0
%     out = out + diag(repelem(lam,numel(d)));
% end
% end
