function [f, info] = ratio_mv(K2, r, order)
%RATIO_MV Saddlepoint expansion of the joint density of R_j = X_j/Y_j, j = 1..d (Theorem 4 and
%   Corollary 1 of Broda and Kan), orders 1..order.
%   K2:  CGF of dimension 2d of (X_1..X_d, Y_1..Y_d), Y_j > 0;  r: the point (d values).
%   With W = X - r.*Y, K_W(s) = K(s, -r.*s), K_W'(shat) = 0, K_W''(shat) = L L', the density is
%      f(r) = (2 pi i)^(-d) int exp(K_W(s)) g(s) ds,  g = sum over the partitions pi of 1..d of
%             prod_{B in pi} K_B(s, -r.*s),  K_B = d^|B| K / prod_{j in B} dt_j,
%   and g = g_0 + ... + g_{d-1}, g_k the partitions into d - k blocks.  The expansion is
%      f ~ exp(K_W(shat)) / ((2 pi)^(d/2) |K_W''(shat)|^(1/2)) sum_{m=0}^{M-1} a_m,
%      a_m = sum_{k=0}^{min(m,d-1)} b_{m-k}(g_k),
%      b_m(h) = sum_{r=0}^{2m} sum_{|nu|+|mu| = 2m+2r} c_{r,nu} gamma_mu He_{nu+mu}(0),
%   in the whitened coordinates s = shat + L^{-T} u: theta_nu (|nu| >= 3) and gamma_mu are the
%   Taylor coefficients of K_W and of h in u, and c_{r,nu} those of Theta^r/r!, Theta the sum of
%   the theta_nu u^nu.  All Taylor coefficients come from one Taylor polynomial of
%   K(s, -r.*s + tau) in (u, tau), the K_B being its coefficients multilinear in tau.
%   info: shat, a (the terms a_m), logC (the log of the leading factor), saddle.
d = numel(r); r = reshape(r, [], 1); M = order;
B = [eye(d); -diag(r)];
KW = linear_combination(K2, B);
[s, sinfo] = spa_saddle(cgf_kfun(KW, 2), zeros(1, d), []);
s = reshape(double(s), [], 1);
P2 = taylor(KW, s, 2);
if isempty(P2), error('CGF:domain', 'ratio_density: the saddlepoint is outside the domain of K'); end
H = zeros(d);
for i = 1:d, for j = 1:d, e = zeros(1, d); e(i) = e(i) + 1; e(j) = e(j) + 1; H(i, j) = coef(P2, e); end, end
[L, flag] = chol((H + H')/2, 'lower');
if flag, error('CGF:hessian', 'ratio_density: K_W'''' is not positive definite at the saddlepoint'); end
Lit = inv(L)';
N = max(2*M, 2*M - 2 + d);                       % degree in (u, tau)
Mmap = [Lit, zeros(d); -diag(r)*Lit, eye(d)];
P = taylor(K2, [s; -r.*s], N, Mmap);
if isempty(P), error('CGF:domain', 'ratio_density: the saddlepoint is outside the domain of K'); end
T = tps.tables(2*d, N);
U = T.mi(:, 1:d); Tau = T.mi(:, d+1:end); du = sum(U, 2);
D = max(6*(M - 1), 2*M);                         % degree of the series in u
T2 = tps.tables(d, D);
pos = @(A) full(T2.lookup(A*T2.base + 1));
mk = @(C) tps(C, [1 1], d, D);
% Theta: the terms of degree 3..2M of K_W(shat + L^{-T} u)
rows = all(Tau == 0, 2) & du >= 3 & du <= 2*M;
C = zeros(T2.n, 1); C(pos(U(rows, :))) = P.C(rows); Theta = mk(C);
K0 = P.C(1);
% K_B(shat + L^{-T} u) for every nonempty subset B (bit mask), degree <= 2(M-1) in u
G = cell(2^d - 1, 1);
for mask = 1:2^d - 1
   tv = bitget(mask, 1:d);
   rows = all(Tau == tv, 2) & du <= 2*(M - 1);
   C = zeros(T2.n, 1); C(pos(U(rows, :))) = P.C(rows); G{mask} = mk(C);
end
% g_k: the partitions into d - k blocks
g = cell(d, 1); for k = 1:d, g{k} = mk(zeros(T2.n, 1)); end
parts = setparts(d);
for i = 1:numel(parts)
   bl = parts{i}; q = G{bl(1)};
   for j = 2:numel(bl), q = q.*G{bl(j)}; end
   k = d - numel(bl); g{k+1} = g{k+1} + q;
end
lowdeg = T2.deg <= 2*(M - 1);
for k = 1:d, C = g{k}.C; C(~lowdeg) = 0; g{k} = mk(C); end
% He_nu(0)
he = ones(T2.n, 1);
for j = 1:d
   n = T2.mi(:, j); h1 = zeros(size(n)); ev = mod(n, 2) == 0;
   h1(ev) = (-1).^(n(ev)/2) .* arrayfun(@dfact, n(ev) - 1);
   he = he.*h1;
end
% c_{r,.} = Theta^r / r!
R = 2*(M - 1); Pr = cell(R + 1, 1); Pr{1} = mk([1; zeros(T2.n - 1, 1)]);
for rr = 1:R, Pr{rr+1} = Pr{rr}.*Theta/rr; end
a = zeros(M, 1);
for k = 0:min(M - 1, d - 1)
   h = g{k+1};
   for rr = 0:R
      Q = Pr{rr+1}.*h;
      for m = ceil(rr/2):M - 1 - k
         sel = T2.deg == 2*m + 2*rr;
         a(m + k + 1) = a(m + k + 1) + sum(Q.C(sel).*he(sel));
      end
   end
end
logC = K0 - d/2*log(2*pi) - sum(log(diag(L)));
f = exp(logC)*cumsum(a);
info = struct('shat', s.', 'a', a, 'logC', logC, 'saddle', sinfo);
end

function v = dfact(n)
v = prod(n:-2:1);
end

function parts = setparts(d)
%SETPARTS all partitions of 1..d, each a vector of block bit masks
parts = {1};
for i = 2:d
   new = {}; b = 2^(i - 1);
   for p = 1:numel(parts)
      bl = parts{p};
      for j = 1:numel(bl), q = bl; q(j) = q(j) + b; new{end+1} = q; end %#ok<AGROW>
      new{end+1} = [bl b]; %#ok<AGROW>
   end
   parts = new;
end
end
