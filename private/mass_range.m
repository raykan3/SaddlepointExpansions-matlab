function [lo, hi] = mass_range(logw, h, c)
%MASS_RANGE The interval of v (the line s = s0 + v e) where the integrand of a normalization is not
%   negligible: from v = 0, which must be inside the domain, steps doubling from h in each
%   direction until logw(v) ([] outside the domain) falls c below the largest value met, then
%   bisection toward the boundary of the domain; an end at which the integrand still has not
%   fallen after 2^12 h (an algebraic tail, as where a gamma variable approaches zero) is
%   infinite.
l0 = logw(0);
if isempty(l0), error('CGF:normalize', 'normalize: the starting point is outside the domain of K'); end
ends = zeros(1, 2); dirs = [-1 1];
for id = 1:2
   dir = dirs(id);
   inside = 0; v = h; lmax = l0;
   while true
      l = logw(dir*v);
      if isempty(l)                                 % outside: bisect toward the boundary
         out = v;
         for it = 1:60
            mid = (inside + out)/2;
            if isempty(logw(dir*mid)), out = mid; else, inside = mid; end
            if out - inside <= 1e-12*max(out, 1), break, end
         end
         ends(id) = dir*inside; break
      end
      lmax = max(lmax, l);
      if l < lmax - c, ends(id) = dir*v; break, end
      inside = v; v = 2*v;
      if v > 4096*h, ends(id) = dir*Inf; break, end
   end
end
lo = ends(1); hi = ends(2);
end
