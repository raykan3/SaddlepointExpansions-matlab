function mass = mass_ratio(K, order)
%MASS_RATIO The masses of the density expansions of orders 1..order of a ratio X_1/X_2 (K the
%   joint cgf, d = 2; ratio_density with 'normalize', true): the integral over r of f_R(r) written
%   as the integral over the saddlepoint shat of f_R(r(shat)) dr/dshat along the locus
%   s K_1 + t K_2 = 0 (ratio_locus), each node continued from the nearest abscissa solved so far;
%   adaptive Gauss-Kronrod quadrature (gk_quad) with the relative tolerance 1e-10, the tails
%   truncated where the leading term has fallen by e^-50 (mass_range).
cs = 0; ctt = 0; ctp = 0;                            % the abscissas solved so far, t and t'
a0 = vals(0);
if isempty(a0), error('CGF:normalize', 'normalize: the ratio expansion cannot be evaluated at the centre'); end
[lo, hi] = mass_range(@lw, 1/sqrt(a0.a.KW2), 50);
mass = gk_quad(@fq, [lo 0 hi], 1e-10);

   function loc = locus(s)
      [~, i] = min(abs(cs - s)); sn = cs(i); t0 = ctt(i); tp0 = ctp(i);
      loc = ratio_locus(K, s, t0 + tp0*(s - sn));
      if isempty(loc)                                % in steps from the nearest one
         m = 8; tt = t0; tp = tp0;
         for j = 1:m
            sj = sn + (s - sn)*j/m;
            lj = ratio_locus(K, sj, tt + tp*(s - sn)/m);
            if isempty(lj), loc = []; return, end
            tt = lj.t; tp = lj.tp; loc = lj;
         end
      end
      k = find(cs == s, 1);
      if isempty(k), cs(end+1) = s; ctt(end+1) = loc.t; ctp(end+1) = loc.tp;
      else, ctt(k) = loc.t; ctp(k) = loc.tp; end
   end
   function v = vals(s)
      v = [];
      loc = locus(s);
      if isempty(loc), return, end
      a = ratio_at(K, s, loc, order);
      if isempty(a), return, end
      v = struct('f', a.f*loc.rp, 'a', a);
   end
   function l = lw(s)
      l = []; v = vals(s);
      if isempty(v) || ~(v.f(1) > 0), return, end
      l = log(v.f(1));
   end
   function y = fq(s)
      v = vals(s);
      if isempty(v), y = zeros(order, 1); else, y = v.f; end
   end
end
