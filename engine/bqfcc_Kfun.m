function v = bqfcc_Kfun(nu,t,m,lam)
%BQFCC_KFUN Derivatives of the joint cgf of Q1 = X1+X0, Q2 = X2+X0, X_i ~ chi^2_{m(i)}(lam(i)):
%   K(t1,t2) = K1(t1) + K2(t2) + K0(t1+t2).  nu = [nu1 nu2], nu = [0 0] gives K.
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Saddlepoint expansions of arbitrary order for multivariate distribution functions"; free to use and modify with a citation of the paper.
if 1-2*t(1) <= 0 || 1-2*t(2) <= 0 || 1-2*t(1)-2*t(2) <= 0, v = NaN; return; end
K0 = ncx2_Kder(t(1)+t(2),m(3),lam(3),nu(1)+nu(2));
v = K0(end);
if nu(2) == 0, K1 = ncx2_Kder(t(1),m(1),lam(1),nu(1)); v = v+K1(end); end
if nu(1) == 0, K2 = ncx2_Kder(t(2),m(2),lam(2),nu(2)); v = v+K2(end); end
