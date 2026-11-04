function r = stud_locus(K, s, tguess, tol)
%STUD_LOCUS The point of the saddlepoint locus s K_s + 2 t K_t = 0 with abscissa s, by Newton's
%   method in t from tguess; r.t, r.v = K_t^(1/2) and r.q = K_s/v, the value of q whose
%   saddlepoint it is.
if nargin < 4 || isempty(tol), tol = 50*eps; end
t = tguess; dprev = Inf;
for it = 1:60
   P = stud_jet(K, s, t, 2);
   Ks = coef(P, [1 0]); Kt = coef(P, [0 1]); Kst = coef(P, [1 1]); Ktt = coef(P, [0 2]);
   g = s*Ks + 2*t*Kt; gp = s*Kst + 2*Kt + 2*t*Ktt;
   t = t - g/gp;
   dn = abs(g/gp);
   stalled = dn < 1e-7*(1 + abs(t)) && dn >= dprev/2;
   dprev = dn;
   if dn < tol*(1 + abs(t)) || stalled
      P = stud_jet(K, s, t, 2);
      Ks = coef(P, [1 0]); Kt = coef(P, [0 1]);
      if ~(Kt > 0), error('CGF:studentized', 'studentized_locus: K_t is not positive at the point found'); end
      v = sqrt(Kt);
      r = struct('t', t, 'v', v, 'q', Ks/v);
      return
   end
end
error('CGF:studentized', 'studentized_locus: the iteration did not converge at s = %g', s);
end
