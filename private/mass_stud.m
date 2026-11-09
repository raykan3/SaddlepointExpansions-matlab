function mass = mass_stud(K, order)
%MASS_STUD The masses of the density expansions of orders 1..order of the Studentized mean
%   Q = X/sqrt(Y) (studentized_density with 'normalize', true): the integral over q written as the
%   integral over the saddlepoint shat along the locus s K_s + 2 t K_t = 0 (stud_locus), with
%   dq/dshat along it; the two sides of shat = 0 (where t = 0 and the expansion is not evaluated)
%   separately, each from |shat| = 0.05/E(Y)^(1/2), by adaptive Gauss-Kronrod quadrature (gk_quad)
%   with the relative tolerance 1e-9, the tails truncated where the leading term has fallen by
%   e^-50 (mass_range).  The expansions at the nodes are those of studentized_density, refined in
%   double-double where they cancel.
P0 = stud_jet(K, 0, 0, 2);
EU = coef(P0, [1 0]); EV = coef(P0, [0 1]);
slope = -EU/(2*EV);                                  % t'(0) along the locus
cs = zeros(1, 0); ctv = zeros(1, 0);                 % the abscissas solved so far and their t
mass = zeros(order, 1);
for dir = [-1 1]
   s0 = dir*(1/20)/sqrt(EV);
   if isempty(vals(s0)), error('CGF:normalize', 'normalize: the Studentized expansion cannot be evaluated near the centre'); end
   [lo, hi] = mass_range(@(v) lw(s0 + v), abs(s0)*4, 50);
   if dir == -1, a = lo; b = -s0; else, a = -s0; b = hi; end   % the side of shat = 0, in v = s - s0
   if a < b, mass = mass + gk_quad(@(v) fq(s0 + v), [a b], 1e-9); end
end

   function y = vals(s)
      y = [];
      if isempty(cs), sn = []; else, [~, i] = min(abs(cs - s)); sn = cs(i); end
      if isempty(sn) || abs(sn - s) > abs(s), tg = slope*s; else, tg = ctv(i); end
      try
         r = stud_locus(K, s, tg);
      catch
         return
      end
      k = find(cs == s, 1);
      if isempty(k), cs(end+1) = s; ctv(end+1) = r.t; else, ctv(k) = r.t; end
      P = taylor(K, [s; r.t], 2);
      if isempty(P), return, end
      Ks = coef(P, [1 0]); Kt = coef(P, [0 1]);
      Kss = coef(P, [2 0]); Kst = coef(P, [1 1]); Ktt = coef(P, [0 2]);
      Gs = Ks + s*Kss + 2*r.t*Kst; Gt = s*Kst + 2*Kt + 2*r.t*Ktt;
      tp = -Gs/Gt;
      vp = (Kst + Ktt*tp)/(2*r.v);
      qp = ((Kss + Kst*tp)*r.v - Ks*vp)/r.v^2;
      try
         f = studentized_density(K, r.q, 'order', order, 'saddle', struct('s', s, 't', r.t, 'v', r.v)).f;
      catch
         return
      end
      y = f(:)*qp;
   end
   function l = lw(s)
      l = []; y = vals(s);
      if isempty(y) || ~(y(1) > 0), return, end
      l = log(y(1));
   end
   function y = fq(s)
      y = vals(s);
      if isempty(y), y = zeros(order, 1); end
   end
end
