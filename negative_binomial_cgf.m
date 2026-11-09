function K = negative_binomial_cgf(r, p)
%NEGATIVE_BINOMIAL_CGF Negative binomial distribution, the number of failures before the r-th
%   success in Bernoulli trials with success probability p (mean r(1-p)/p):
%   K(t) = r log p - r log(1 - (1-p) e^t), t < -log(1-p); r need not be an integer.  A lattice
%   variable ('lattice', true in tailprob and density).
if ~(isscalar(r) && isscalar(p) && r > 0 && p > 0 && p < 1), error('negative_binomial_cgf:arg', 'negative_binomial_cgf: r must be positive and 0 < p < 1'); end
q = 1 - p;
K = CGF(@(t) r*log(p) - r*log(1 - q*exp(t(1))), 1, 'domain', @(t) q*exp(t(1)) < 1);
end
