function r = stud_quantile(K, prob, M, tol)
%STUD_QUANTILE The quantile of the saddlepoint distribution function of order M of
%   Q = X/sqrt(Y): P(Q <= q) = prob.  The search runs over shat along the saddlepoint locus
%   s K_s + 2 t K_t = 0 (stud_locus), where each trial point gives its q and its distribution
%   function directly, so that no saddlepoint equation is solved: a bracket in shat from the
%   origin, then the Illinois iteration.  r.q, r.saddle.
P0 = stud_jet(K, 0, 0, 2);
EU = coef(P0, [1 0]); EV = coef(P0, [0 1]);
slope = -EU/(2*EV);                                      % t'(0) along the locus
step = 0.5; s = 0.1; t = slope*s;
[F, ~, t] = at(s, t);
if F < prob, dir = 1; else, dir = -1; end
if dir == -1
   s = -s; t = slope*s; [F, ~, t] = at(s, t);
end
sa = s; Fa = F; ta = t;
while (dir == 1 && F < prob) || (dir == -1 && F > prob)
   sa = s; Fa = F; ta = t;
   s = s + dir*step; [F, ~, t] = at(s, t);
   if abs(s) > 1e4, error('CGF:studentized', 'studentized_quantile: no bracket found'); end
end
sb = s; Fb = F; tb = t;
fa = Fa - prob; fb = Fb - prob; side = 0;
for it = 1:200
   sm = (sa*fb - sb*fa)/(fb - fa);
   tg = ta + (tb - ta)*(sm - sa)/(sb - sa);
   [Fm, qm, tm] = at(sm, tg);
   fm = Fm - prob;
   if abs(fm) < tol
      loc = stud_locus(K, sm, tm);
      r = struct('q', qm, 'saddle', struct('s', sm, 't', loc.t, 'v', loc.v));
      return
   end
   if sign(fm) == sign(fb)
      sb = sm; fb = fm; tb = tm;
      if side == 1, fa = fa/2; end
      side = 1;
   else
      sa = sm; fa = fm; ta = tm;
      if side == -1, fb = fb/2; end
      side = -1;
   end
end
error('CGF:studentized', 'studentized_quantile: the iteration did not converge');

   function [Fv, qv, tv] = at(sv, tgv)
      loc = stud_locus(K, sv, tgv);
      rr = stud_tailprob(K, loc.q, M, true, struct('s', sv, 't', loc.t, 'v', loc.v));
      Fv = rr.F(end); qv = loc.q; tv = loc.t;
   end
end
