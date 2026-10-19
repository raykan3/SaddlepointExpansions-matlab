function [Fup,pdf,Flo] = bqfcc_exact(x1,x2,m,lam)
%BQFCC_EXACT Upper orthant probability P[Q1 >= x1, Q2 >= x2], density and lower corner
%   probability P[Q1 <= x1, Q2 <= x2] for Q1 = X1 + X0, Q2 = X2 + X0 with
%   X_i ~ chi^2_{m(i)}(lam(i)) independent, m = [m1 m2 m0], lam = [lam1 lam2 lam0],
%   by one-dimensional integration over X0 in double precision (relative tolerance 1e-12;
%   the upper tails are computed directly, so that small probabilities keep their relative
%   accuracy).  The values in bqfcc_exact.csv were computed in 30-digit arithmetic.
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Saddlepoint expansions of arbitrary order for multivariate distribution functions"; free to use and modify with a citation of the paper.
f0 = @(s) ncx2pdf(s,m(3),lam(3));
opts = {'AbsTol',1e-300,'RelTol',1e-12};
Fup = integral(@(s) f0(s).*ncx2cdf(x1-s,m(1),lam(1),'upper').*ncx2cdf(x2-s,m(2),lam(2),'upper'),0,Inf,opts{:}) ...
    + 0;   % for s >= min(x1,x2) one of the upper tails is 1 (ncx2cdf(negative,...,'upper') = 1)
Flo = integral(@(s) f0(s).*ncx2cdf(x1-s,m(1),lam(1)).*ncx2cdf(x2-s,m(2),lam(2)),0,min(x1,x2),opts{:});
pdf = integral(@(s) f0(s).*ncx2pdf(x1-s,m(1),lam(1)).*ncx2pdf(x2-s,m(2),lam(2)),0,min(x1,x2),opts{:});
