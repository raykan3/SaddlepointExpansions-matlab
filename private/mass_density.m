function mass = mass_density(K, order)
%MASS_DENSITY The masses c_M = int f_M(x) dx, M = 1..order, of the density expansions of a CGF of
%   dimension d (density with 'normalize', true): the integral over x of f_M(x) written as the
%   integral over s of f_M(K'(s)) det K''(s), nested over the coordinates (the last outermost);
%   every line starts from a point inside the domain (the regression of its coordinate on the
%   outer ones under K''(0), or the nearest point inside along the line), and its range is where
%   the leading term, for the inner lines, or the inner integral, for the outer ones, has not
%   fallen by e^-50 from its largest value (mass_range); adaptive Gauss-Kronrod quadrature
%   (gk_quad) with the relative tolerance 1e-12 in one dimension and 1e-9 in more, as in the Julia
%   package.  In more than one dimension the expansion at a node is that of spa_cond in whitened
%   coordinates, evaluated at the known saddlepoint s.
d = K.d; maxdeg = 2*order + 1;
if d == 1, rtol = 1e-12; else, rtol = 1e-9; end
Kf = cgf_kfun(K, maxdeg); Kw = cgf_kwhite(K, maxdeg);
E = eye(d);
H0 = zeros(d);
for i0 = 1:d, for j0 = 1:d, H0(i0, j0) = Kf(E(i0, :) + E(j0, :), zeros(1, d)); end, end
if ~all(isfinite(H0(:))), error('CGF:normalize', 'normalize: K is not finite at the origin'); end
c = 50;
mass = level(d, zeros(0, 1));

   function tf = inside(s), tf = ~isempty(taylor(K, s, 2)); end
   function l = lead(s)                              % log of f_1(K'(s)) det K''(s)
      l = [];
      P = taylor(K, s, 2);
      if isempty(P), return, end
      [g, H] = gradhess(P);
      [Lc, flag] = chol(H);
      if ~all(isfinite(H(:))) || flag, return, end
      l = coef(P, zeros(1, d)) - s.'*g - (d/2)*log(2*pi) + sum(log(diag(Lc)));
   end
   function v = full(s)                              % f_M(K'(s)) det K''(s), M = 1..order
      v = zeros(order, 1);
      if d == 1
         Kd = uni_derivs(K, s, maxdeg);
         if isempty(Kd), return, end
         pdf = spa_uni(Kd, s, Kd(2), order);
         v = pdf(:)*Kd(3);
         return
      end
      % the gradient and the Hessian from Kf, whose Taylor polynomial at s is then reused by spa_cond
      if ~isfinite(Kf(zeros(1, d), s)), return, end
      g = zeros(d, 1); H = zeros(d);
      for i = 1:d
         g(i) = Kf(E(i, :), s);
         for j = 1:d, H(i, j) = Kf(E(i, :) + E(j, :), s); end
      end
      % where K'' is numerically singular (at the boundary of the domain, where the density is
      % negligible) the node counts as outside
      [~, flag] = chol(H);
      if ~all(isfinite(H(:))) || flag, return, end
      [~, fM] = spa_cond(Kf, g.', 1:d, order, struct('tail', false, 'Kwhite', Kw, 'tauf', s.'));
      v = fM*det(H);
   end
   function [g, H] = gradhess(P)
      g = zeros(d, 1); H = zeros(d);
      for i = 1:d
         g(i) = coef(P, E(i, :));
         for j = 1:d, H(i, j) = coef(P, E(i, :) + E(j, :)); end
      end
   end
   function I = level(k, outer)
      % the integral over s_k (and the inner coordinates) with s(k+1:d) = outer
      if k == d, ck = 0; else, ck = -(H0(k, k+1:d)*outer)/H0(k, k); end
      pt = @(v) [zeros(k-1, 1); v; outer];
      h = 1/sqrt(H0(k, k));
      if ~inside(pt(ck))                             % a point of the line inside the domain
         found = false;
         for j = 0:40
            for dir = [-1 1]
               v = ck + dir*h*2^(j/2);
               if inside(pt(v)), ck = v; found = true; break, end
            end
            if found, break, end
         end
         if ~found, I = zeros(order, 1); return, end
      end
      if k == 1
         lw = @(v) lead(pt(ck + v));
         fv = @(v) full(pt(ck + v));
      else
         memo = containers.Map('KeyType', 'double', 'ValueType', 'any');
         fv = @(v) inner(memo, k, ck, outer, v);
         lw = @(v) loginner(fv(v));
      end
      [lo, hi] = mass_range(lw, h, c);
      I = gk_quad(fv, [lo 0 hi], rtol);
   end
   function Iv = inner(memo, k, ck, outer, v)
      if isKey(memo, v), Iv = memo(v); return, end
      Iv = level(k-1, [ck + v; outer]);
      memo(v) = Iv;
   end
end

function l = loginner(I)
if I(1) > 0, l = log(I(1)); else, l = []; end
end
