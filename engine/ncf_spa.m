function [cdf,pdf,upp,kappa] = ncf_spa(f,m,n,lam,order)
%NCF_SPA Saddlepoint approximations of orders 1..order of the distribution function and the
%   density of the singly noncentral F distribution, F = (Q1/m)/(Q2/n), Q1 ~ chi^2_m(lam),
%   Q2 ~ chi^2_n, at f > 0 (Section S2.2 of the Supplementary Material).  P[F <= f] = P[W <= 0], W = Q1 - c Q2,
%   c = m f/n, by the Hauschildt-Robinson expansion (upp: the same expansion of P[F > f],
%   computed directly); the density by the Rice expansion of the
%   density of R = Q1/Q2 at c (spa_ratio_pdf), times m/n.  The saddlepoint is the root in
%   (-1/(2c),1/2) of the quadratic obtained from K_W'(s) = 0 by clearing denominators,
%      -4c(m+n) s^2 + 2(cm - m + c lam + 2cn) s + (m + lam - cn) = 0.
%   kappa(M): cancellation indicator of the cdf sum (see spa_uni): about log10(kappa) digits of
%   cdf(M) are lost to rounding in double precision (an arithmetic error, distinct from the
%   asymptotic error of the expansion).  The values are not clipped to [0,1].
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Uniform saddlepoint expansions for multivariate tail probabilities"; free to use and modify with a citation of the paper.
validate(f,m,n,lam,order);
c = m*f/n;
A = -4*c*(m+n); B = 2*(c*m-m+c*lam+2*c*n); C = m+lam-c*n;
r = roots([A B C]); r = r(abs(imag(r)) < 1e-12 & real(r) > -1/(2*c) & real(r) < 1/2);
shat = real(r(1));
N = 2*order+1;                                  % derivatives needed by spa_uni
Kd = ncx2_Kder(shat,m,lam,N)+((-c).^(0:N)').*ncx2_Kder(-c*shat,n,0,N);
[~,cdf,~,upp,kappa] = spa_uni(Kd,shat,0,order);
K2d = ncx2_Kder(-c*shat,n,0,2*order);                  % g(s) = K_{Q2}'(-c s)
gd = ((-c).^(0:2*order-1)').*K2d(2:2*order+1);
pdf = m/n*spa_ratio_pdf(Kd,gd,shat,order);
end

function validate(f,m,n,lam,order)
if ~(isscalar(f) && f > 0 && isfinite(f)), error('ncf_spa: f must be a positive scalar'); end
if ~(isscalar(m) && m > 0 && isscalar(n) && n > 0), error('ncf_spa: the degrees of freedom must be positive'); end
if ~(isscalar(lam) && lam >= 0 && isfinite(lam)), error('ncf_spa: lam must be a nonnegative scalar'); end
if ~(isscalar(order) && order >= 1 && order == round(order)), error('ncf_spa: order must be a positive integer'); end
end
