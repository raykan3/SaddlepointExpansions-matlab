function r = stud_tailprob_dd(K, q, M, lower, sd)
%STUD_TAILPROB_DD the distribution function expansion of stud_tailprob in double-double arithmetic
%   (saddlepoint refined, Taylor polynomial of K, the reduction to one dimension, the sums), with
%   the integrals I_m in variable precision.  Where psi_tt < 0 the square root of psi_tt is
%   imaginary; the code works with the real root of -psi_tt and the signs (-1)^j that the powers of
%   i contribute, which gives the same (real) coefficients as the complex arithmetic of the
%   double-precision code.
sd = stud_refine_dd(K, q, sd);
s = sd.s; t = sd.t; v = sd.v;
if double(t) == 0, error('CGF:studentized', 'studentized_tailprob: the saddlepoint has t = 0'); end
p = 2*M - 1;
[Psi_c, Hk_c] = reduction(K, q, s, t, p);
[Fp, kappa] = hrsum(Psi_c, Hk_c, s, p);
if lower, Fp = 1 - Fp; end
Fd = double(Fp);
F = Fd(2*(1:M));
if kappa(end)*2^-104 > 1e-10, status = 'coefficients inaccurate'; else, status = 'ok'; end
r = struct('F', F(:), 'saddle', struct('s', double(s), 't', double(t), 'v', double(v)), 'status', status, ...
   'info', struct('Fp', Fd(:), 'kappa', kappa(:), 'Psi_c', double(Psi_c), 'Hk_c', {cellfun(@double, Hk_c, 'UniformOutput', false)}, ...
   'refined', 'double-double'));
end

function [Psi_c, Hk_c] = reduction(K, q, s0, t0, p)
jmax = floor(p/2);
N = max(p + 2, 2*jmax + 2 + p + 2);
J = stud_jet_dd(K, s0, t0, N);
w = tpsdd.vars(2, N);
qq = dd(q)*dd(q)/4;
psi = J + (s0 + w(1))^2*qq/(t0 + w(2));
A = coefmat(psi, N); AJ = coefmat(J, N);
At = dtmat(A, 1, N); Att = dtmat(A, 2, N);
tau = dd(zeros(1, N + 1));
for it = 1:N + 1
   g = compose(At, tau, N); gt = compose(Att, tau, N);
   tau = tau - sdiv(g, gt, N);
   tau = subsasgn(tau, substruct('()', {1}), dd(0));
end
Psi = compose(A, tau, N);
su = dd([0 1 zeros(1, N - 1)]); su = subsasgn(su, substruct('()', {1}), s0);
tu = subsasgn(tau, substruct('()', {1}), t0);
vs = sdiv(su, tu, N)*(-dd(q)/2);
Ktt = compose(dtmat(AJ, 2, N), tau, N);
arg = 2*smul(tu, Ktt, N) + 4*smul(vs, vs, N);
if ~(double(at1(arg)) > 0)
   error('CGF:studentized', 'studentized_tailprob: 2 t K_tt + 4 v^2 is not positive at the saddlepoint');
end
H0 = sdiv(2*vs, ssqrt(arg, N), N);
p2 = compose(Att, tau, N);
sg = sign(double(at1(p2)));                        % psi_tt > 0: sg = 1; < 0: sg = -1 (sqrt imaginary)
rho = ssqrt(sg*p2, N);
theta = cell(1, 2*jmax + 2);
for k = 3:2*jmax + 2
   theta{k} = sdiv(compose(dtmat(A, k, N), tau, N), spow(rho, k, N), N)/dd(factorial(k));
end
itu = sdiv(dd([1 zeros(1, N)]), tu, N);
gam = @(l) sdiv(spow(itu, l, N), spow(rho, l, N), N)*gbinom(l);
P2 = 2*jmax;
d = cell(P2 + 1, P2 + 1);
for i = 1:numel(d), d{i} = dd(zeros(1, N + 1)); end
d{1, 1} = dd([1 zeros(1, N)]);
for rr = 1:P2
   for j = rr:P2
      acc = dd(zeros(1, N + 1));
      for k = 1:j - rr + 1, acc = acc + smul(theta{k + 2}, d{rr, j - k + 1}, N); end
      d{rr + 1, j + 1} = acc/rr;
   end
end
hh = cumprod([1, 1 - 2*(1:2*P2 + 2)]);              % h_i = He_2i(0), exact
beta = cell(1, jmax);
for j = 1:jmax
   acc = dd(zeros(1, N + 1));
   for k = 0:2*j
      aij = dd(zeros(1, N + 1));
      for kk = 0:k
         sgn = 1; if sg < 0, sgn = (-1)^kk; end
         aij = aij + d{kk + 1, k + 1}*(sgn*hh(j + kk + 1));
      end
      acc = acc + smul(gam(2*j - k), aij, N);
   end
   if sg < 0, acc = acc*(-1)^j; end
   beta{j} = acc;
end
Psi_c = subsref(Psi, substruct('()', {1:p + 3}));
Hk_c = cell(1, jmax + 1);
Hk_c{1} = subsref(H0, substruct('()', {1:p + 1}));
for k = 1:jmax
   Hk = smul(H0, beta{k}, N);
   Hk_c{k + 1} = subsref(Hk, substruct('()', {1:p - 2*k + 1}));
end
end

function [out, kappa] = hrsum(Psi_c, Hk_c, s0, p)
if double(s0) >= 0, sg = 1; else, sg = -1; end
Psi2 = 2*at(Psi_c, 3);
if ~(double(Psi2) > 0), error('CGF:studentized', 'studentized_tailprob: Psi''''(shat) is not positive'); end
sP = sqrt(Psi2);
theta = cell(1, p + 2);
for k = 3:p + 2, theta{k} = at(Psi_c, k + 1)*(sg^k)/sP^k; end
gam = cell(1, numel(Hk_c));
for k = 1:numel(Hk_c)
   n = numel(Hk_c{k}); g = dd(zeros(1, n));
   for l = 0:n - 1, g = subsasgn(g, substruct('()', {l + 1}), at(Hk_c{k}, l + 1)*(sg^l)/sP^l); end
   gam{k} = g;
end
dmat = cell(p + 1, p + 1); for i = 1:numel(dmat), dmat{i} = dd(0); end
dmat{1, 1} = dd(1);
for rr = 1:p
   for j = rr:p
      acc = dd(0);
      for k = 1:j - rr + 1, acc = acc + theta{k + 2}*dmat{rr, j - k + 1}; end
      dmat{rr + 1, j + 1} = acc/rr;
   end
end
w = sqrt(-2*at(Psi_c, 1)); u = abs(s0)*sP;
I = Itable(u, w, 3*p + 1);                          % E_n I_m(u), m = 0..3p+1
out = dd(zeros(p + 1, 1)); kappa = zeros(p + 1, 1);
tot = dd(0); abssum = 0;
for j = 0:p
   for k = 0:floor(j/2)
      for m = 0:j - 2*k
         l = j - 2*k - m;
         for rr = 0:m
            if rr == 0 && m ~= 0, continue; end
            term = dmat{rr + 1, m + 1}*at(gam{k + 1}, l + 1)*at(I, m + l + 2*rr + 1);
            if double(term) == 0, continue; end
            tot = tot + term; abssum = abssum + abs(double(term));
         end
      end
   end
   if sg == 1, val = tot; else, val = 1 - tot; end
   out = subsasgn(out, substruct('()', {j + 1}), val);
   kappa(j + 1) = abssum/abs(double(tot));
end
end

function I = Itable(u, w, mmax)
% E_n I_m(u), m = 0..mmax, E_n = exp((u^2 - w^2)/2), in variable precision: I_0 = Phi(-u),
% I_{m+1} = He_m(0) phi(u) - u I_m (forward; the digits lost are covered by the working precision)
ud = double(u);
lost = max(0, (2*mmax + 1)*log10(max(ud, 1)) - sum(log10(1:mmax)));
digs = 50 + ceil(lost);
uv = vpa(u, digs); wv = vpa(w, digs);
En = exp((uv^2 - wv^2)/2); phi = exp(-uv^2/2)/sqrt(2*vpa(pi, digs));
Iv = vpa(zeros(mmax + 1, 1), digs);
Iv(1) = erfc(uv/sqrt(vpa(2, digs)))/2;
h = 1;
for m = 0:mmax - 1
   if mod(m, 2) == 0, hm = h; h = h*(-(m + 1)); else, hm = 0; end
   Iv(m + 2) = hm*phi - uv*Iv(m + 1);
end
I = dd(En*Iv);
end

%---------------------------------------------------------------- double-double series helpers
function x = at(a, i), x = subsref(a, substruct('()', {i})); end
function x = at1(a), x = at(a, 1); end
function g = gbinom(l), g = 1; for i = 0:l-1, g = g*(-1.5 - i)/(i + 1); end, g = dd(g); end
function A = coefmat(P, N)
T = tps.tables(2, N);
h = zeros(N + 1); l = h;
for idx = 1:T.n
   h(T.mi(idx, 1) + 1, T.mi(idx, 2) + 1) = P.H(idx); l(T.mi(idx, 1) + 1, T.mi(idx, 2) + 1) = P.L(idx);
end
A = dd(h, l);
end
function B = dtmat(A, k, N)
f = zeros(1, N + 1);
for j = 0:N - k, f(j + 1) = prod(j + 1:j + k); end
idx = min((0:N) + k, N) + 1;                         % columns j + k (zeroed by f beyond N - k)
B = subsref(A, substruct('()', {':', idx})).*f;
end
function r = compose(A, tau, N)
r = transpose(subsref(A, substruct('()', {':', N + 1})));
for j = N-1:-1:0
   r = smul(r, tau, N) + transpose(subsref(A, substruct('()', {':', j + 1})));
end
end
function c = smul(a, b, N)
% truncated product of two double-double series (row vectors of length N + 1)
P = transpose(a).*b;                                 % P(i+1, j+1) = a_i b_j
[I, J] = ndgrid(0:N, 0:N); dg = I + J;
c = dd(zeros(1, N + 1));
h = P.hi; l = P.lo;
for k = 0:N
   sel = dg == k;
   [sh, sl] = dd.colsum(h(sel), l(sel));
   c = subsasgn(c, substruct('()', {k + 1}), dd(sh, sl));
end
end
function c = sdiv(a, b, N)
c = dd(zeros(1, N + 1)); b0 = at(b, 1);
for k = 0:N
   acc = at(a, k + 1);
   if k > 0
      bi = subsref(b, substruct('()', {2:k + 1})); ci = subsref(c, substruct('()', {k:-1:1}));
      acc = acc - sum(bi.*ci);
   end
   c = subsasgn(c, substruct('()', {k + 1}), acc/b0);
end
end
function c = ssqrt(a, N)
c = dd(zeros(1, N + 1)); c0 = sqrt(at(a, 1)); c = subsasgn(c, substruct('()', {1}), c0);
for k = 1:N
   acc = at(a, k + 1);
   if k > 1
      ci = subsref(c, substruct('()', {2:k})); cj = subsref(c, substruct('()', {k:-1:2}));
      acc = acc - sum(ci.*cj);
   end
   c = subsasgn(c, substruct('()', {k + 1}), acc/(2*c0));
end
end
function c = spow(a, k, N)
c = dd([1 zeros(1, N)]);
for i = 1:k, c = smul(c, a, N); end
end
