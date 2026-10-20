function [cdf,pdf,upp,kappa] = dncf_spa(f,m,n,lam1,lam2,order)
%DNCF_SPA Saddlepoint approximations of orders 1..order of the cdf and pdf of the
%   doubly noncentral F, F = (Q1/m)/(Q2/n), Q1 ~ chi^2_m(lam1), Q2 ~ chi^2_n(lam2).
%   cdf: Hauschildt-Robinson expansion of P[Q1 - c Q2 <= 0], c = m f/n; upp: the same
%   expansion of P[F > f] computed directly (for small upper tails); kappa: cancellation
%   indicator of the cdf sum (see spa_uni; an arithmetic diagnostic, distinct from the
%   asymptotic error; the values are not clipped to [0,1]);
%   pdf: Rice expansion of the density of R = Q1/Q2 at c, times m/n.
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Uniform saddlepoint expansions for multivariate tail probabilities"; free to use and modify with a citation of the paper.
if ~(isscalar(f) && f > 0 && isfinite(f)), error('dncf_spa: f must be a positive scalar'); end
if ~(isscalar(order) && order >= 1 && order == round(order)), error('dncf_spa: order must be a positive integer'); end
if ~(lam1 >= 0 && lam2 >= 0 && m > 0 && n > 0), error('dncf_spa: invalid degrees of freedom or noncentralities'); end
c = m*f/n;
N = 2*order+1;                                  % derivatives needed by spa_uni
KW = @(s) ncx2_Kder(s,m,lam1,N) + ((-c).^(0:N)').*ncx2_Kder(-c*s,n,lam2,N);
shat = dncf_saddle(c,m,n,lam1,lam2);                 % root of K_W'(s) = 0 (cubic + Newton)
Kd = KW(shat);
[~,cdf,~,upp,kappa] = spa_uni(Kd,shat,0,order);
% g(s) = K_2'(-c s): derivatives g^{(l)}(s) = (-c)^l K_2^{(l+1)}(-c s)
K2d = ncx2_Kder(-c*shat,n,lam2,2*order);
gd = ((-c).^(0:2*order-1)').*K2d(2:2*order+1);
pdf = m/n*spa_ratio_pdf(Kd,gd,shat,order);
end
