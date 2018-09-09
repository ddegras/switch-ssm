function xinv = myinv(x)
try
    xinv = inv(x);
catch
    xinv = pinv(x);
end
end