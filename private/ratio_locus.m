function loc = ratio_locus(K, s, tguess)
%RATIO_LOCUS The point of the saddlepoint locus of a ratio R = X_1/X_2 with abscissa s: t with
%   s K_1(s, t) + t K_2(s, t) = 0, by Newton's method in t from tguess, with r = K_1/K_2 there and its
%   derivative along the locus, t' = -G_s/G_t; [] if the iteration leaves the domain or does not
%   converge.  loc.t, loc.r, loc.rp (dr/ds), loc.tp (dt/ds).
if s == 0, tguess = 0; end
t = tguess; loc = [];
for it = 1:60
   P = taylor(K, [s; t], 2);
   if isempty(P), return, end
   K1 = coef(P, [1 0]); K2 = coef(P, [0 1]);
   K11 = coef(P, [2 0]); K12 = coef(P, [1 1]); K22 = coef(P, [0 2]);
   G = s*K1 + t*K2; Gt = s*K12 + K2 + t*K22;
   if ~(K2 > 0 && Gt > 0), return, end
   dt = G/Gt;
   if abs(dt) <= 50*eps*(1 + abs(t)) || G == 0
      Gs = K1 + s*K11 + t*K12;
      tp = -Gs/Gt;
      r = K1/K2;
      rp = ((K11 + K12*tp)*K2 - K1*(K12 + K22*tp))/K2^2;
      loc = struct('t', t, 'r', r, 'rp', rp, 'tp', tp);
      return
   end
   t = t - dt;
end
end
