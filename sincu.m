function y = sincu(x)
%SINCU sin(x)/x (unnormalized), equal to one at zero, for numbers and power series (tps), with
%   the removable singularity handled as in sinhc.
if isa(x, 'tps')
   if all(double(x) ~= 0), y = sin(x)./x; return; end
   x2 = x.*x; y = tps.const(ones(size(x)), x.k, x.N); term = y;
   for m = 1:floor(x.N/2), term = -term.*x2/((2*m)*(2*m + 1)); y = y + term; end
else
   y = ones(size(x)); nz = x ~= 0; y(nz) = sin(x(nz))./x(nz);
end
end
