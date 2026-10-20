function v = mvcc_Kfun(nu,t,m,lam)
%MVCC_KFUN Derivative of order nu (1 x d) at t (1 x d) of the joint cgf of Q_j = X_j + X_0,
%   j = 1..d, X_i ~ chi^2_{m(i)}(lam(i)) independent, m = [m_1 ... m_d m_0], lam likewise:
%   K(t) = sum_j K_j(t_j) + K_0(t_1 + ... + t_d).  NaN outside the domain.  Works in double
%   precision and in variable precision (vpa t).  For d = 2 it is bqfcc_Kfun.
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Uniform saddlepoint expansions for multivariate tail probabilities"; free to use and modify with a citation of the paper.
d = numel(t);
if any(double(1-2*t) <= 0) || double(1-2*sum(t)) <= 0, v = NaN; return; end
k = sum(nu);
v = kchi(k,sum(t),m(d+1),lam(d+1));
nz = find(nu > 0);
if isempty(nz)
   for j = 1:d, v = v+kchi(0,t(j),m(j),lam(j)); end
elseif numel(nz) == 1
   v = v+kchi(k,t(nz),m(nz),lam(nz));
end
end

function v = kchi(k,u,mm,ll)
%  k-th derivative of the cgf of chi^2_mm(ll) at u
if k == 0
   v = -mm/2*log(1-2*u)+ll*u/(1-2*u);
else
   v = 2^(k-1)*factorial(k-1)*mm/(1-2*u)^k+2^(k-1)*factorial(k)*ll/(1-2*u)^(k+1);
end
end
