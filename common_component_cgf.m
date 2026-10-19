function K = common_component_cgf(m, lam)
%COMMON_COMPONENT_CGF Q_j = X_j + X_0, j = 1..d, with independent X_i ~ chi^2_{m_i}(lam_i),
%   m = [m_1 ... m_d m_0] and lam likewise (default zeros): K(t) = sum_j K_j(t_j) + K_0(sum_j t_j),
%   the K_i noncentral chi-squared cgfs.  The pairs of quadratic forms of Section 6.1 of the paper
%   and the trivariate example are of this form.
if nargin < 2, lam = zeros(size(m)); end
d = numel(m) - 1;
if d < 1, error('common_component_cgf:m', 'common_component_cgf: m must have at least two entries'); end
k1 = @(u, mi, li) -mi/2*log(1 - 2*u) + li*u./(1 - 2*u);
K = CGF(@(t) ccval(t, m, lam, d, k1), d, 'domain', @(t) all(t < 1/2) && sum(t) < 1/2);
end
function v = ccval(t, m, lam, d, k1)
v = k1(sum(t), m(d+1), lam(d+1));
for j = 1:d, v = v + k1(t(j), m(j), lam(j)); end
end
