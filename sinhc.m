function y = sinhc(x)
%SINHC sinh(x)/x, equal to one at zero, for numbers and power series (tps); at a series with zero
%   constant term the series sum_m x^(2m)/(2m+1)! is used, so that the removable singularity
%   causes no error.  The cgf of the uniform distribution on [-1, 1] is log(sinhc(t)).
if isa(x, 'tps')
   if all(double(x) ~= 0), y = sinh(x)./x; return; end
   x2 = x.*x; y = tps.const(ones(size(x)), x.k, x.N); term = y;
   for m = 1:floor(x.N/2), term = term.*x2/((2*m)*(2*m + 1)); y = y + term; end
else
   y = ones(size(x)); nz = x ~= 0; y(nz) = sinh(x(nz))./x(nz);
end
end
