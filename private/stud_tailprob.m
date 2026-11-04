function r = stud_tailprob(K, q, M, lower, saddle)
%STUD_TAILPROB Saddlepoint expansions of orders 1..M of P(Q >= q) (lower: P(Q <= q)) for
%   Q = X/sqrt(Y) (Theorem 3 of Bao, Broda and Kan): Hauschildt-Robinson type expansion in s after
%   the t-integral is expanded by Laplace's method, with Psi(s) = psi(s, t(s)), psi_t(s, t(s)) = 0.
%   All Taylor coefficients of Psi and of the amplitudes H_k at the saddlepoint are obtained from
%   the Taylor polynomial of K by Newton's method and composition on power series.  r.F(M) keeps
%   the terms of orders n^(-m/2), m <= 2M - 1; r.info.Fp(j+1), j = 0..2M-1, those up to n^(-j/2);
%   r.info.kappa the cancellation indicator.
if isempty(saddle), sd = stud_saddle(K, q); else, sd = saddle; end
s = sd.s; t = sd.t; v = sd.v;
if t == 0
   error('CGF:studentized', ['studentized_tailprob: the saddlepoint has t = 0 (q is the point ' ...
      'q0 = E(X)/E(Y)^(1/2)); the expansion is not evaluated there']);
end
p = 2*M - 1;
[Psi_c, Hk_c] = reduction(K, q, s, t, p);
[Fp, kappa] = hrsum(Psi_c, Hk_c, s, p);
if lower, Fp = 1 - Fp; end
F = Fp(2*(1:M));
if kappa(end) > 1e8, status = 'coefficients inaccurate'; else, status = 'ok'; end
r = struct('F', F(:), 'saddle', struct('s', s, 't', t, 'v', v), 'status', status, ...
   'info', struct('Fp', Fp(:), 'kappa', kappa(:), 'Psi_c', Psi_c, 'Hk_c', {Hk_c}));
end

function [Psi_c, Hk_c] = reduction(K, q, s0, t0, p)
% the Taylor coefficients Psi_c(k+1), k = 0..p+2, of Psi at shat and Hk_c{k+1}(l+1), l = 0..p-2k,
% of the amplitudes H_k
jmax = floor(p/2);
N = max(p + 2, 2*jmax + 2 + p + 2);
J = stud_jet(K, s0, t0, N);
w = tps.vars(2, N);
psi = J + (s0 + w(1))^2*(q*q/4)/(t0 + w(2));
A = coefmat(psi, N); AJ = coefmat(J, N);
At = dtmat(A, 1, N); Att = dtmat(A, 2, N);
tau = zeros(1, N + 1);                                   % t(s) - that, nilpotent in ds
for it = 1:N + 1
   g = compose(At, tau, N); gt = compose(Att, tau, N);
   tau = tau - sdiv(g, gt, N);
   tau(1) = 0;
end
Psi = compose(A, tau, N);
su = [s0 1 zeros(1, N - 1)];
tu = tau; tu(1) = t0;
vs = -q/2*sdiv(su, tu, N);
Ktt = compose(dtmat(AJ, 2, N), tau, N);
arg = 2*smul(tu, Ktt, N) + 4*smul(vs, vs, N);
if ~(arg(1) > 0)
   error('CGF:studentized', 'studentized_tailprob: 2 t K_tt + 4 v^2 is not positive at the saddlepoint');
end
H0 = sdiv(2*vs, ssqrt(arg, N), N);
p2 = compose(Att, tau, N);
sq = ssqrt(complex(p2), N);                              % complex where psi_tt < 0
theta = cell(1, 2*jmax + 2);
for k = 3:2*jmax + 2
   theta{k} = sdiv(complex(compose(dtmat(A, k, N), tau, N)), spow(sq, k, N), N)/factorial(k);
end
itu = sdiv([1 zeros(1, N)], tu, N);
gam = @(l) sdiv(complex(spow(itu, l, N)), spow(sq, l, N), N)*gbinom(-3/2, l);
P2 = 2*jmax;
d = cell(P2 + 1, P2 + 1);                                % d{r+1, j+1}: series
for i = 1:numel(d), d{i} = complex(zeros(1, N + 1)); end
d{1, 1}(1) = 1;
for rr = 1:P2
   for j = rr:P2
      acc = complex(zeros(1, N + 1));
      for k = 1:j - rr + 1
         acc = acc + smul(theta{k + 2}, d{rr, j - k + 1}, N);
      end
      d{rr + 1, j + 1} = acc/rr;
   end
end
hh = hcoef(2*P2 + 2);                                    % hh(i+1) = h_i = He_2i(0)
beta = cell(1, jmax);
for j = 1:jmax
   acc = complex(zeros(1, N + 1));
   for k = 0:2*j
      aij = complex(zeros(1, N + 1));
      for kk = 0:k, aij = aij + d{kk + 1, k + 1}*hh(j + kk + 1); end
      acc = acc + smul(gam(2*j - k), aij, N);
   end
   beta{j} = acc;
end
Psi_c = Psi(1:p + 3);
Hk_c = cell(1, jmax + 1);
Hk_c{1} = H0(1:p + 1);
for k = 1:jmax
   Hk = smul(complex(H0), beta{k}, N);
   c = Hk(1:p - 2*k + 1);
   if any(abs(imag(c)) > sqrt(eps)*(1 + abs(c)))
      error('CGF:studentized', 'studentized_tailprob: a coefficient of H_%d is not real; the expansion does not apply at this q', k);
   end
   Hk_c{k + 1} = real(c);
end
end

function [out, kappa] = hrsum(Psi_c, Hk_c, s0, p)
% the Hauschildt-Robinson type sum: approximations to P(Q > q) with the terms up to n^(-j/2),
% j = 0..p, with the reflection for shat < 0
if s0 >= 0, sg = 1; else, sg = -1; end
Psi2 = 2*Psi_c(3);
if ~(Psi2 > 0), error('CGF:studentized', 'studentized_tailprob: Psi''''(shat) is not positive'); end
theta = zeros(1, p + 2);
for k = 3:p + 2, theta(k) = sg^k*Psi_c(k + 1)/Psi2^(k/2); end
gam = cell(1, numel(Hk_c));
for k = 1:numel(Hk_c)
   l = 0:numel(Hk_c{k}) - 1;
   gam{k} = sg.^l.*Hk_c{k}./Psi2.^(l/2);
end
d = zeros(p + 1, p + 1);                                 % d(r+1, j+1)
d(1, 1) = 1;
for rr = 1:p
   for j = rr:p
      acc = 0;
      for k = 1:j - rr + 1, acc = acc + theta(k + 2)*d(rr, j - k + 1); end
      d(rr + 1, j + 1) = acc/rr;
   end
end
w = sqrt(-2*Psi_c(1)); u = abs(s0)*sqrt(Psi2);
En = exp((u^2 - w^2)/2);
I = Itable(u, 3*p + 1);
out = zeros(p + 1, 1); kappa = zeros(p + 1, 1);
tot = 0; abssum = 0;
for j = 0:p
   for k = 0:floor(j/2)
      for m = 0:j - 2*k
         l = j - 2*k - m;
         for rr = 0:m
            if rr == 0 && m == 0, drm = 1; elseif rr == 0, drm = 0; else, drm = d(rr + 1, m + 1); end
            if drm == 0, continue; end
            term = drm*gam{k + 1}(l + 1)*I(m + l + 2*rr + 1);
            tot = tot + term; abssum = abssum + abs(term);
         end
      end
   end
   val = En*tot;
   if sg == 1, out(j + 1) = val; else, out(j + 1) = 1 - val; end
   kappa(j + 1) = abssum/abs(tot);
end
end

function I = Itable(u, mmax)
% I_m(u), m = 0..mmax, u >= 0: forward recursion, or 80-point Gauss-Hermite for u > 2.5
phi = exp(-u^2/2)/sqrt(2*pi);
I = zeros(mmax + 1, 1);
if u > 2.5
   I = phi*Iherm(u, mmax);
   return
end
h = hcoef(floor(mmax/2) + 1);
I(1) = 0.5*erfc(u/sqrt(2));
for m = 0:mmax - 1
   if mod(m, 2) == 0, hm = h(m/2 + 1); else, hm = 0; end
   I(m + 2) = hm*phi - u*I(m + 1);
end
end

function R = Iherm(u, mmax)
% R(m+1) = I_m(u)/phi(u): R(2k+1) = (-1)^k u M_k, R(2k+2) = (-1)^k M_{k+1}, M_j = E[Z^(2j)/(u^2+Z^2)]
persistent xg wg
if isempty(xg)
   n = 80; b = sqrt(1:n-1);
   [V, Dg] = eig(diag(b, 1) + diag(b, -1));
   [xg, o] = sort(diag(Dg)); wg = V(1, o)'.^2;
end
qv = wg./(u^2 + xg.^2); x2 = xg.^2;
Jn = floor(mmax/2) + 1; Mm = zeros(Jn + 1, 1);
for j = 0:Jn, Mm(j + 1) = sum(qv); qv = qv.*x2; end
R = zeros(mmax + 1, 1);
for m = 0:mmax
   k = floor(m/2);
   if mod(m, 2) == 0, R(m + 1) = (-1)^k*u*Mm(k + 1); else, R(m + 1) = (-1)^k*Mm(k + 2); end
end
end

function h = hcoef(n)
% h(i+1) = He_2i(0) = prod_{j=1}^i (1 - 2j), i = 0..n
h = cumprod([1, 1 - 2*(1:n)]);
end

function g = gbinom(a, k)
g = 1; for i = 0:k-1, g = g*(a - i)/(i + 1); end
end

function A = coefmat(P, N)
% A(i+1, j+1): the coefficient of ds^i dt^j of the bivariate series P
T = tps.tables(2, N);
A = zeros(N + 1, N + 1);
for idx = 1:T.n, A(T.mi(idx, 1) + 1, T.mi(idx, 2) + 1) = P.C(idx); end
end

function B = dtmat(A, k, N)
% the coefficient matrix of the k-th t-derivative
B = zeros(N + 1, N + 1);
for j = 0:N - k
   B(:, j + 1) = A(:, j + k + 1)*prod(j + 1:j + k);
end
end

function r = compose(A, tau, N)
% P(shat + ds, that + tau(ds)) as a univariate series in ds (tau nilpotent): Horner in t
r = A(:, N + 1).';
for j = N - 1:-1:0
   r = smul(r, tau, N) + A(:, j + 1).';
end
end

function c = smul(a, b, N)
c = conv(a, b); c = c(1:N + 1);
end

function c = sdiv(a, b, N)
% a/b for series, b(1) ~= 0
c = zeros(1, N + 1) + 0*a(1)*b(1);
for k = 0:N
   acc = a(k + 1);
   for i = 1:k, acc = acc - b(i + 1)*c(k - i + 1); end
   c(k + 1) = acc/b(1);
end
end

function c = ssqrt(a, N)
% sqrt of a series (principal branch at the constant term, complex allowed)
c = zeros(1, N + 1) + 0*a(1);
c(1) = sqrt(a(1));
for k = 1:N
   acc = a(k + 1);
   for i = 1:k - 1, acc = acc - c(i + 1)*c(k - i + 1); end
   c(k + 1) = acc/(2*c(1));
end
end

function c = spow(a, k, N)
c = zeros(1, N + 1) + 0*a(1); c(1) = 1;
for i = 1:k, c = smul(c, a, N); end
end
