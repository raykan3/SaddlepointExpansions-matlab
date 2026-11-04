function sd = stud_refine_dd(K, q, sd)
%STUD_REFINE_DD the saddlepoint (s, t, v) refined to double-double accuracy: Newton's method with
%   the residual of K_s = q v, K_t = v^2, q s + 2 t v = 0 in double-double and the Jacobian in double
s = dd(sd.s); t = dd(sd.t); v = dd(sd.v); qd = dd(q);
for it = 1:8
   P = stud_jet_dd(K, s, t, 2);
   Ks = coef(P, [1 0]); Kt = coef(P, [0 1]);
   Kss = double(coef(P, [2 0])); Kst = double(coef(P, [1 1])); Ktt = double(coef(P, [0 2]));
   F = [double(Ks - qd*v); double(Kt - v*v); double(qd*s + 2*t*v)];
   A = [Kss Kst -q; Kst Ktt -2*double(v); q 2*double(v) 2*double(t)];
   d = A\F;
   s = s - d(1); t = t - d(2); v = v - d(3);
   if sum(abs(d)) < 1e-30*(1 + abs(double(s)) + abs(double(t)) + abs(double(v))), break; end
end
sd = struct('s', s, 't', t, 'v', v);
end
