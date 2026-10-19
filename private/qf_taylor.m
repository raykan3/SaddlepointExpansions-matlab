function p = qf_taylor(B, m, scale, t, N, M)
%QF_TAYLOR Taylor polynomial of degree N in v of K(t + M v) for the joint cgf of the quadratic
%   forms w'B_j w, w ~ N(m, I): K(t) = -1/2 log det(I - 2C) + 1/2 m'{(I - 2C)^{-1} - I}m,
%   C = sum_j t_j B_j, multiplied by scale.  Closed form (trace recursion): with D = (I - 2C)^{-1}
%   = L L', the derivative of order a in the directions Dir_i = sum_j M(j,i) B_j is
%   2^{|a|-1} a!/|a| {tr F_a + |a| y'F_a y}, F_a the sum of the products of G_i = L'Dir_i L in
%   every order (symmetric), y = L'm; F is formed up to degree ceil(N/2) and the higher traces
%   by splitting the words, tr(F_g F_r) and (F_g y)'(F_r y).  [] outside the domain.
d = numel(B); n = size(B{1}, 1);
if isempty(M), M = eye(d); end
k = size(M, 2);
C = zeros(n); for j = 1:d, C = C + t(j)*B{j}; end
[R, flag] = chol(eye(n) - 2*C);                 % I - 2C = R'R
if flag, p = []; return; end
L = R \ eye(n);                                  % D = L L'
y = L.'*m;
T = tps.tables(k, N); coefs = zeros(T.n, 1);
coefs(1) = scale*(-sum(log(diag(R))) + 0.5*(y.'*y - m.'*m));
G = cell(1, k);
for i = 1:k, Dir = zeros(n); for j = 1:d, if M(j, i) ~= 0, Dir = Dir + M(j, i)*B{j}; end, end, Gi = L.'*Dir*L; G{i} = (Gi + Gi.')/2; end
h = ceil(N/2);
F = cell(T.n, 1); Fy = cell(T.n, 1);
hasMean = any(m ~= 0);
for idx = 2:T.n
   a = T.mi(idx, :); l = T.deg(idx);
   if l > h, break; end
   if l == 1
      Fa = G{find(a, 1)};
   else
      Fa = zeros(n);
      for i = 1:k
         if a(i) > 0, b = a; b(i) = b(i) - 1; Fa = Fa + G{i}*F{T.lookup(b*T.base + 1)}; end
      end
      Fa = (Fa + Fa.')/2;
   end
   F{idx} = Fa;
   val = trace(Fa);
   if hasMean, Fy{idx} = Fa*y; val = val + l*(y.'*Fy{idx}); end
   coefs(idx) = scale*2^(l-1)/l*val;
end
for idx = 2:T.n
   a = T.mi(idx, :); l = T.deg(idx);
   if l <= h, continue; end
   kk = l - h;
   g = T.mi(T.deg == kk, :); g = g(all(g <= a, 2), :);
   tval = 0; mq = 0;
   for q = 1:size(g, 1)
      ig = T.lookup(g(q, :)*T.base + 1); ir = T.lookup((a - g(q, :))*T.base + 1);
      tval = tval + sum(F{ig}(:).*F{ir}(:));
      if hasMean, mq = mq + Fy{ig}.'*Fy{ir}; end
   end
   coefs(idx) = scale*2^(l-1)/l*(tval + l*mq);
end
p = tps(coefs, [1 1], k, N);
end
