function xinv = myinv(x)
try
    xinv = inv(x);
catch
    xinv = pinv(x);
end

if any(isnan(xinv(:)))
    xinv = pinv(x);
end
