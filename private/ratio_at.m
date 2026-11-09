function a = ratio_at(K, s, loc, order)
%RATIO_AT The distribution function (spa_uni of W = X_1 - r X_2 at 0) and the density
%   (spa_ratio_pdf) of the ratio expansions at the point of the saddlepoint locus with abscissa s
%   (loc from ratio_locus); [] outside the domain.  a.cdf, a.upp, a.f (orders 1..order), a.KW2.
r = loc.r; N = 2*order + 1; a = [];
P = taylor(K, [s; -r*s], N, [1 0; -r 1]);            % K(s + v1, -r(s + v1) + v2)
if isempty(P), return, end
KWd = coef(P, [(0:N)' zeros(N+1, 1)]);
if ~(all(isfinite(KWd)) && KWd(3) > 0), return, end
gd = coef(P, [(0:2*order-2)' ones(2*order-1, 1)]);
[~, cdf, ~, upp] = spa_uni(KWd, s, 0, order);
f = spa_ratio_pdf(KWd(1:2*order+1), gd, s, order);
a = struct('cdf', cdf(:), 'upp', upp(:), 'f', f(:), 'KW2', KWd(3));
end
