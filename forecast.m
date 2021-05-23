function yhat = forecast(pars,S,x,model)

model = string(model);
assert(ismember(model,["dyn","obs"]))
[M,T] =size(S);
N = size(pars.C,1);
p = size(pars.A,3);
r = size(x,1); 

A = pars.A;
C = pars.C;
Z = pars.Z;
T = numel(S);

yhat = zeros(N,T);

switch model 
    case "obs"
        for t = p+1:T
            for j = 1:M
                xx = zeros(r,1);
                for k = 1:p
                    xx = xx + A(:,:,k,j) * x(:,j,t-k);
                end
                yhat(:,t) = yhat(:,t) + Z(S(t-1),j) * C(:,:,j) * xx;
            end
        end
    case "dyn"
        for t = p+1:T
            for j = 1:M
                xx = zeros(r,1);
                for k = 1:p
                    xx = xx + Z(S(t-1),j) * A(:,:,k,j) * x(:,t-k);
                end
            end
            yhat(:,t) = C * xx;
        end
end

yhat(:,1:p) = NaN(N,p);

