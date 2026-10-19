function K = normal_cgf(mu, Sigma)
%NORMAL_CGF N(mu, Sigma): K(t) = mu't + t'Sigma t/2 (the expansions are exact at order one).
%   normal_cgf(mu, sigma2) with scalars gives the univariate normal.
mu = reshape(mu, [], 1); d = numel(mu);
if ~isequal(size(Sigma), [d d]), error('normal_cgf:Sigma', 'normal_cgf: Sigma must be %d x %d', d, d); end
K = CGF(@(t) mu.'*t + sum(t.*(Sigma*t))/2, d);
end
