function K = gamma_cgf(a, b)
%GAMMA_CGF Gamma distribution with shape a and scale b (default 1; mean a b):
%   K(t) = -a log(1 - b t), t < 1/b.  gamma_cgf(1, b) is the exponential distribution with mean b.
if nargin < 2, b = 1; end
K = CGF(@(t) -a*log(1 - b*t(1)), 1, 'domain', @(t) b*t(1) < 1);
end
