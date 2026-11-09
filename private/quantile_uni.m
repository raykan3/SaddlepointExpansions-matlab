function r = quantile_uni(K, p, order, lower, tol)
%QUANTILE_UNI The quantile of the saddlepoint distribution function (spa_uni) of order 'order' of a
%   univariate CGF, by the safeguarded Newton iteration in saddlepoint space (quant_newton): the
%   point x = K'(s) and its distribution function come with s, and the derivative in s is the
%   density expansion times K''(s).  r.q, r.tau, r.F, r.iterations, r.status.
N = 2*order + 1;
[s, e, it, status] = quant_newton(@ev, 0, @(e) 1/sqrt(e.K2), tol*min(p, 1 - p), 'quantile');
r = struct('q', e.x, 'tau', s, 'F', e.F, 'iterations', it, 'status', status);

   function out = ev(s)
      out = [];
      Kd = uni_derivs(K, s, N);
      if isempty(Kd), return, end
      [pdf, cdf, ~, upp] = spa_uni(Kd, s, Kd(2), order);
      if lower, F = cdf(end); g = F - p; else, F = upp(end); g = p - F; end
      out = struct('g', g, 'dg', pdf(end)*Kd(3), 'x', Kd(2), 'F', F, 'K2', Kd(3));
   end
end
