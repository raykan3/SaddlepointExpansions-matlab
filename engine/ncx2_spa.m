function [cdf,pdf,upp,kappa] = ncx2_spa(x,nu,lam,order)
%NCX2_SPA Saddlepoint approximations of orders 1..order of the distribution function and the
%   density of the noncentral chi-squared distribution chi^2_nu(lam) at x > 0 (Section S2.1 of the
%   Supplementary Material): Hauschildt-Robinson expansion of P[X <= x] (and of P[X > x], upp, computed directly)
%   and Daniels expansion of the density.  The saddlepoint solves
%   K'(s) = nu/(1-2s) + lam/(1-2s)^2 = x, a quadratic in y = 1/(1-2s), whose root is written
%   in the form that is stable for small lam.  kappa(M): cancellation indicator of the cdf sum
%   (see spa_uni).  The values are not clipped to [0,1].
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Uniform saddlepoint expansions for multivariate tail probabilities"; free to use and modify with a citation of the paper.
if ~(isscalar(x) && x > 0 && isfinite(x)), error('ncx2_spa: x must be a positive scalar'); end
if ~(isscalar(nu) && nu > 0 && isscalar(lam) && lam >= 0), error('ncx2_spa: invalid nu or lam'); end
if ~(isscalar(order) && order >= 1 && order == round(order)), error('ncx2_spa: order must be a positive integer'); end
y = 2*x/(nu+sqrt(nu^2+4*lam*x));
s = (1-1/y)/2;
Kd = ncx2_Kder(s,nu,lam,2*order+1);
[pdf,cdf,~,upp,kappa] = spa_uni(Kd,s,x,order);
end
