function Kd = ncx2_Kder(u,m,lam,N)
%NCX2_KDER Derivatives K^{(k)}(u), k = 0..N, of the cgf of chi^2_m(lam):
%   K(u) = -(m/2) log(1-2u) + lam u/(1-2u).
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Uniform saddlepoint expansions for multivariate tail probabilities"; free to use and modify with a citation of the paper.
Kd = zeros(N+1,1);
Kd(1) = -m/2*log1p(-2*u)+lam*u/(1-2*u);
for k = 1:N
   Kd(k+1) = 2^(k-1)*factorial(k-1)*m/(1-2*u)^k + 2^(k-1)*factorial(k)*lam/(1-2*u)^(k+1);
end
