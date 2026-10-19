function K = ncx2_cgf(m, lam)
%NCX2_CGF Noncentral chi-squared chi^2_m(lam): K(t) = -(m/2) log(1 - 2t) + lam t/(1 - 2t), t < 1/2.
if nargin < 2, lam = 0; end
K = CGF(@(t) -m/2*log(1 - 2*t(1)) + lam*t(1)/(1 - 2*t(1)), 1, 'domain', @(t) t(1) < 1/2);
end
