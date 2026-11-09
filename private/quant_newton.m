function [s, e, it, status] = quant_newton(ev, s0, scale, tol, name)
%QUANT_NEWTON Safeguarded Newton iteration for an increasing function of s (the quantiles in
%   saddlepoint space): ev(s) returns a struct with g (the function) and dg (its derivative), or
%   [] outside the domain.  Steps are limited to 4 scale(e), pulled back into the domain by
%   halving, and replaced by bisection once the root is bracketed and a step leaves the bracket.
%   Converged when |g| <= tol (status 'ok'), or when the bracket has shrunk to rounding level
%   (status 'bracket at rounding level').
maxit = 200;
s = s0; e = ev(s);
if isempty(e), error('CGF:domain', '%s: the starting point is outside the domain of K', name); end
lo = -Inf; hi = Inf; status = 'ok';
for it = 1:maxit
   if abs(e.g) <= tol, return, end
   if e.g < 0, lo = s; else, hi = s; end
   smax = 4*scale(e);
   if e.dg > 0, step = -e.g/e.dg; elseif e.g < 0, step = smax; else, step = -smax; end
   sn = s + min(max(step, -smax), smax);
   if isfinite(lo) && isfinite(hi) && ~(lo < sn && sn < hi), sn = (lo + hi)/2; end
   en = ev(sn);
   while isempty(en)                                % outside the domain: back toward s
      sn = (s + sn)/2;
      if abs(sn - s) <= 4*eps*max(abs(s), 1), error('CGF:domain', '%s: the iteration reached the boundary of the domain of K', name); end
      en = ev(sn);
   end
   if isfinite(lo) && isfinite(hi) && hi - lo <= 8*eps*max([abs(lo), abs(hi), 1])
      s = sn; e = en; status = 'bracket at rounding level';   % bracket at rounding level
      return
   end
   s = sn; e = en;
end
error('CGF:quantile', '%s: the iteration did not converge', name);
end
