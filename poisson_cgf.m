function K = poisson_cgf(lam)
%POISSON_CGF Poisson distribution with mean lam: K(t) = lam (e^t - 1).  A lattice variable: use
%   'lattice', true in tailprob and density for P(S >= k) and P(S = k).
if ~(isscalar(lam) && lam > 0), error('poisson_cgf:lam', 'poisson_cgf: lam must be positive'); end
K = CGF(@(t) lam*(exp(t(1)) - 1), 1);
end
