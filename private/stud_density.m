function r = stud_density(K, q, M, saddle)
%STUD_DENSITY Saddlepoint expansions of orders 1..M of the density of Q = X/sqrt(Y) at q
%   (Theorem 2 of Bao, Broda and Kan): the bivariate amplitude expansion about the saddlepoint of
%   psi(s, t) = K(s, t) + q^2 s^2/(4t), evaluated in the coordinates of a complex factorization
%   psi'' = L L' (psi'' is indefinite where t < 0).  r.f(M), r.saddle, r.status,
%   r.info.kappa (cancellation indicator), r.info.pref, r.info.psi0, r.info.H.
if isempty(saddle), sd = stud_saddle(K, q); else, sd = saddle; end
s = sd.s; t = sd.t; v = sd.v;
if t == 0
   error('CGF:studentized', ['studentized_density: the saddlepoint has t = 0 (q is the point ' ...
      'q0 = E(X)/E(Y)^(1/2)); the expansion is not evaluated there']);
end
D = max(6*(M - 1), 2);
P = stud_jet(K, s, t, D);
w0 = tps.vars(2, D);
psi = P + (s + w0(1))^2*(q*q/4)/(t + w0(2));
psi0 = double(psi);
H = [coef(psi, [2 0]) coef(psi, [1 1]); coef(psi, [1 1]) coef(psi, [0 2])];
detH = H(1,1)*H(2,2) - H(1,2)^2;
if ~(t*detH > 0)
   error('CGF:studentized', 'studentized_density: that*det(psi'''') is not positive at the saddlepoint');
end
pref = v^2*exp(psi0)/sqrt(pi*t*detH);
Lit = complexfactor(H);                        % inv(L)'
u = tps.vars(2, D);
w = Lit*u;                                     % the whitened coordinates: w = L^(-T) u
psiu = subs(psi, w, D);
T = tps.tables(2, D);
% Theta: the terms of degree 3..2M of psi in the whitened coordinates
Cth = zeros(T.n, 1);
sel = T.deg >= 3 & T.deg <= min(2*M, D);
Cth(sel) = psiu.C(sel);
Theta = tps(Cth, [1 1], 2, D);
% the amplitudes, polynomials in u of degree <= 2(M-1)
a0 = trunc(((s + w(1))/s)^2*((t + w(2))/t)^(-5/2), 2*(M - 1));
a1 = trunc(((t + w(2))/t)^(-3/2)/(2*t*v*v), 2*(M - 1));
he = hemoment(T.mi);
R = 2*(M - 1);
Pr = cell(R + 1, 1);
Pr{1} = tps.const(1, 2, D);
for k = 1:R, Pr{k + 1} = Pr{k}*Theta/k; end
f = zeros(M, 1); kappa = zeros(M, 1);
br = 1; ab = 1; f(1) = pref; kappa(1) = 1;
for m = 1:M - 1
   [b0, s0] = bcoef(m, a0); [b1, s1] = bcoef(m - 1, a1);
   br = br + real(b0 + b1); ab = ab + s0 + s1;
   f(m + 1) = pref*br; kappa(m + 1) = ab/abs(br);
end
if kappa(end) > 1e10, status = 'coefficients inaccurate'; else, status = 'ok'; end
r = struct('f', f, 'saddle', struct('s', s, 't', t, 'v', v), 'status', status, ...
   'info', struct('pref', pref, 'psi0', psi0, 'H', H, 'kappa', kappa));

   function [tot, abssum] = bcoef(m, a)
      % b_m(a) = sum_{r=0}^{2m} sum_{|nu| = 2m+2r} [Theta^r/r! a]_nu He_nu(0)
      tot = 0; abssum = 0;
      for rr = 0:2*m
         Qp = Pr{rr + 1}*a;
         ix = T.deg == 2*m + 2*rr;
         terms = Qp.C(ix).*he(ix);
         tot = tot + sum(terms); abssum = abssum + sum(abs(terms));
      end
   end
   function p = trunc(p, deg)
      Cp = p.C; Cp(T.deg > deg) = 0; p = tps(Cp, [1 1], 2, D);
   end
end

function Lit = complexfactor(H)
% inv(L)' for a complex L with H = L L', from the eigen-decomposition of the symmetric 2 x 2
% matrix H (indefinite where t < 0; its (1,1) entry vanishes in symmetric cases, so a triangular
% factorization is not used): L = V diag(sqrt(lambda)), inv(L)' = V diag(1/sqrt(lambda))
a = H(1,1); b = H(2,1); c = H(2,2);
th = atan2(2*b, a - c)/2; cs = cos(th); sn = sin(th);
V = [cs -sn; sn cs];
l1 = a*cs*cs + 2*b*sn*cs + c*sn*sn; l2 = a*sn*sn - 2*b*sn*cs + c*cs*cs;
if l1 == 0 || l2 == 0, error('CGF:studentized', 'studentized_density: psi'''' is singular at the saddlepoint'); end
Lit = V*diag([1/sqrt(complex(l1)), 1/sqrt(complex(l2))]);
end

function out = subs(P, w, N)
% the scalar series P evaluated at the vector w of series (composition), degree N
T = tps.tables(2, N);
pw = cell(2, N + 1);
for i = 1:2
   pw{i, 1} = tps.const(1, 2, N);
   for k = 1:N, pw{i, k + 1} = pw{i, k}*w(i); end
end
Cout = zeros(T.n, 1);
for idx = 1:T.n
   cf = P.C(idx);
   if cf == 0, continue; end
   nu = T.mi(idx, :);
   term = pw{1, nu(1) + 1}*pw{2, nu(2) + 1};
   Cout = Cout + cf*term.C;
end
out = tps(Cout, [1 1], 2, N);
end

function he = hemoment(mi)
% He_nu(0) of the standard bivariate normal: prod_i (-1)^(n_i/2) (n_i - 1)!! for even n_i, else 0
he = ones(size(mi, 1), 1);
for i = 1:size(mi, 1)
   for n = mi(i, :)
      if mod(n, 2), he(i) = 0; break; end
      he(i) = he(i)*(-1)^(n/2)*prod(n-1:-2:1);
   end
end
end
