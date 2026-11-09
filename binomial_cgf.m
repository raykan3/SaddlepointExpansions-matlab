function K = binomial_cgf(m, p)
%BINOMIAL_CGF Binomial distribution with m trials and success probability p:
%   K(t) = m log(1 - p + p e^t).  A lattice variable ('lattice', true in tailprob and density).
if ~(isscalar(m) && isscalar(p) && m > 0 && p > 0 && p < 1), error('binomial_cgf:arg', 'binomial_cgf: m must be positive and 0 < p < 1'); end
K = CGF(@(t) m*log(1 - p + p*exp(t(1))), 1);
end
