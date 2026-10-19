function K = uniform_cgf(a, b)
%UNIFORM_CGF Uniform distribution on [a, b] (default [-1, 1]): K(t) = log sinhc{(b-a)t/2} + (a+b)t/2,
%   finite for all t (sinhc handles t = 0).
if nargin < 1, a = -1; end
if nargin < 2, b = 1; end
if ~(b > a), error('uniform_cgf:ab', 'uniform_cgf: b must exceed a'); end
h = (b - a)/2; c = (a + b)/2;
K = CGF(@(t) log(sinhc(h*t(1))) + c*t(1), 1);
end
