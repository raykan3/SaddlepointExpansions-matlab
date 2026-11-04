function r = stud_density_dd(K, q, M, sd)
%STUD_DENSITY_DD the density expansion of stud_density in double-double arithmetic: the Taylor
%   polynomial of K, the saddlepoint (refined), the whitening and all sums carry about 32 digits.
%   The whitening is real: W'*psi''*W = diag(sigma), sigma = +-1, from the eigen-decomposition of
%   psi'', and the Hermite moments are those of the indefinite form diag(sigma), which gives the
%   same terms as the complex factorization of the double-precision code.
sd = stud_refine_dd(K, q, sd);
s = sd.s; t = sd.t; v = sd.v;
if double(t) == 0, error('CGF:studentized', 'studentized_density: the saddlepoint has t = 0'); end
D = max(6*(M - 1), 2);
P = stud_jet_dd(K, s, t, D);
w0 = tpsdd.vars(2, D);
qq = dd(q)*dd(q)/4;
psi = P + (s + w0(1))^2*qq/(t + w0(2));
psi0 = ddconst(psi);
H = [coef(psi, [2 0]) coef(psi, [1 1]); coef(psi, [1 1]) coef(psi, [0 2])];
detH = H(1,1)*H(2,2) - H(1,2)*H(1,2);
if ~(double(t*detH) > 0)
   error('CGF:studentized', 'studentized_density: that*det(psi'''') is not positive at the saddlepoint');
end
pidd = dd(vpa(pi, 40));
pref = v*v*exp(psi0)/sqrt(pidd*t*detH);
[W, sigma] = realfactor(H);
u = tpsdd.vars(2, D);
w = W*u;                                       % psi'' in the coordinates u is diag(sigma)
psiu = subs2(psi, w, D);
T = tps.tables(2, D);
sel = T.deg >= 3 & T.deg <= min(2*M, D);
Theta = tpsdd(psiu.H.*sel, psiu.L.*sel, [1 1], 2, D);
R = 2*(M - 1);
a0 = trunc(((s + w(1))/s)^2*((t + w(2))/t)^(-2.5), R, T, D);
a1 = trunc(((t + w(2))/t)^(-1.5)/(2*t*v*v), R, T, D);
he = hemoment(T.mi, sigma);
Pr = cell(R + 1, 1);
Pr{1} = tpsdd.const(1, 2, D);
for k = 1:R, Pr{k + 1} = Pr{k}*Theta/k; end
f = zeros(M, 1); kappa = zeros(M, 1);
br = dd(1); ab = 1; f(1) = double(pref); kappa(1) = 1;
for m = 1:M - 1
   [b0, s0] = bcoef(m, a0); [b1, s1] = bcoef(m - 1, a1);
   br = br + b0 + b1; ab = ab + s0 + s1;
   f(m + 1) = double(pref*br); kappa(m + 1) = ab/abs(double(br));
end
if kappa(end)*2^-104 > 1e-10, status = 'coefficients inaccurate'; else, status = 'ok'; end
r = struct('f', f, 'saddle', struct('s', double(s), 't', double(t), 'v', double(v)), 'status', status, ...
   'info', struct('pref', double(pref), 'psi0', double(psi0), 'H', double(H), 'kappa', kappa, 'refined', 'double-double'));

   function [tot, abssum] = bcoef(m, a)
      tot = dd(0); abssum = 0;
      for rr = 0:2*m
         Qp = Pr{rr + 1}*a;
         ix = find(T.deg == 2*m + 2*rr);
         terms = dd(Qp.H(ix), Qp.L(ix)).*he(ix);
         tot = tot + sum(terms); abssum = abssum + sum(abs(double(terms)));
      end
   end
end

function p = trunc(p, deg, T, D)
keep = T.deg <= deg;
p = tpsdd(p.H.*keep, p.L.*keep, [1 1], 2, D);
end

function [W, sigma] = realfactor(H)
% W with W'*H*W = diag(sigma), sigma = +-1, in double-double, from the eigen-decomposition of the
% symmetric 2 x 2 matrix H (as the complex factorization of stud_density, so the terms of the sums
% have the same magnitudes): H = V diag(lambda) V', W = V diag(1/sqrt(|lambda|)), sigma = sign(lambda)
a = H(1,1); b = H(1,2); c = H(2,2);
dlt = a - c;
r = sqrt(dlt*dlt + 4*b*b);
if double(r) == 0
   cs = dd(1); sn = dd(0);                     % H is a multiple of the identity
elseif double(dlt/r) >= 0
   c2 = dlt/r; s2 = 2*b/r;                     % cos(2 theta), sin(2 theta)
   cs = sqrt((1 + c2)/2); sn = s2/(2*cs);
else
   c2 = dlt/r; s2 = 2*b/r;
   sn = sqrt((1 - c2)/2); cs = s2/(2*sn);
end
l1 = a*cs*cs + 2*b*sn*cs + c*sn*sn;
l2 = a*sn*sn - 2*b*sn*cs + c*cs*cs;
detH = a*c - b*b;
if abs(double(l1)) >= abs(double(l2)), l2 = detH/l1; else, l1 = detH/l2; end
if double(l1) == 0 || double(l2) == 0, error('CGF:studentized', 'studentized_density: psi'''' is singular at the saddlepoint'); end
V = [cs, -sn; sn, cs];
W = V.*[1/sqrt(abs(l1)), 1/sqrt(abs(l2))];
sigma = [sign(double(l1)), sign(double(l2))];
end

function out = subs2(P, w, N)
% the bivariate series P evaluated at the series w(1), w(2): Horner in w(1), powers of w(2)
T = tps.tables(2, N);
pw = cell(N + 1, 1); pw{1} = tpsdd.const(1, 2, N);
w2 = w(2); w1 = w(1);
for j = 1:N, pw{j + 1} = pw{j}*w2; end
inner = cell(N + 1, 1);
for i = 0:N
   acc = tpsdd.const(0, 2, N);
   for j = 0:N - i
      idx = T.lookup([i j]*T.base + 1);
      if P.H(idx) == 0 && P.L(idx) == 0, continue; end
      acc = acc + pw{j + 1}*rawcoef(P, idx);
   end
   inner{i + 1} = acc;
end
out = inner{N + 1};
for i = N-1:-1:0, out = out*w1 + inner{i + 1}; end
end

function he = hemoment(mi, sigma)
% He_nu(0, diag(sigma)): prod_i (-1)^(n_i/2) sigma_i^(n_i/2) (n_i - 1)!! for even n_i, else 0
he = ones(size(mi, 1), 1);
for i = 1:size(mi, 1)
   for c = 1:2
      n = mi(i, c);
      if mod(n, 2), he(i) = 0; break; end
      he(i) = he(i)*(-1)^(n/2)*sigma(c)^(n/2)*prod(n-1:-2:1);
   end
end
end
