function r = quantile_ratio(K, p, order, lower, tol)
%QUANTILE_RATIO The quantile of the saddlepoint distribution function of order 'order' of the ratio
%   X_1/X_2 (K the joint cgf, d = 2), by the safeguarded Newton iteration in the saddlepoint shat of
%   W = X_1 - r X_2 along the locus s K_1 + t K_2 = 0 (ratio_locus, continued from the previous
%   iterate); the derivative is the ratio density expansion times dr/dshat.  r.r, r.shat, r.F,
%   r.iterations, r.status.
last = [0 0 0];                                       % (s, t, t') of the previous iterate
[s, e, it, status] = quant_newton(@ev, 0, @(e) 1/sqrt(e.KW2), tol*min(p, 1 - p), 'ratio_quantile');
r = struct('r', e.r, 'shat', s, 'F', e.F, 'iterations', it, 'status', status);

   function out = ev(s)
      out = [];
      loc = ratio_locus(K, s, last(2) + last(3)*(s - last(1)));
      if isempty(loc), return, end
      a = ratio_at(K, s, loc, order);
      if isempty(a), return, end
      last = [s loc.t loc.tp];
      if lower, F = a.cdf(end); g = F - p; else, F = a.upp(end); g = p - F; end
      out = struct('g', g, 'dg', a.f(end)*loc.rp, 'r', loc.r, 'F', F, 'KW2', a.KW2);
   end
end
