function [K,alist] = qf_Kmulti(theta,Mats,L)
%QF_KMULTI Derivatives of the joint cgf of the quadratic forms z'A_i z, z ~ N(0,I),
%   K(theta) = -1/2 log det(I - 2C), C = sum_i theta_i A_i, at theta, for all multi-indices
%   alpha with |alpha| <= L:  K_alpha = 2^{|alpha|-1} alpha! tr(F_alpha)/|alpha|, with
%   F_alpha = sum_i D A_i F_{alpha-e_i}, D = (I-2C)^{-1}, F_0 = I.
%   K is a containers.Map from the string of alpha (e.g. '1 0 2') to the value; K('0 0 0') = K(theta).
%   Returns K = [] if I - 2C is not positive definite (theta outside the domain).
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Saddlepoint expansions of arbitrary order for multivariate distribution functions"; free to use and modify with a citation of the paper.
p = numel(Mats); N = size(Mats{1},1);
C = zeros(N); for i = 1:p, C = C+theta(i)*Mats{i}; end
Q = eye(N)-2*C;
[~,flag] = chol(Q); if flag, K = []; alist = []; return; end
D = inv(Q);
DA = cell(p,1); for i = 1:p, DA{i} = D*Mats{i}; end
alist = multiindices(p,L);
nA = size(alist,1);
keys = cell(nA,1); for a = 1:nA, keys{a} = strtrim(sprintf('%d ',alist(a,:))); end
pos = containers.Map(keys,num2cell(1:nA));
F = cell(nA,1); vals = zeros(nA,1);
for a = 1:nA
   al = alist(a,:); m = sum(al);
   if m == 0
      F{a} = eye(N); vals(a) = -0.5*sum(log(eig(Q)));   % log det via eigenvalues (Q spd)
      continue;
   end
   Fa = zeros(N);
   for i = 1:p
      if al(i) > 0
         b = al; b(i) = b(i)-1; Fa = Fa+DA{i}*F{pos(strtrim(sprintf('%d ',b)))};
      end
   end
   F{a} = Fa;
   vals(a) = 2^(m-1)*prod(factorial(al))*trace(Fa)/m;
end
K = containers.Map(keys,num2cell(vals));
end

function A = multiindices(p,L)
% all nonnegative integer p-vectors with sum <= L, ordered by total degree
A = zeros(0,p);
for m = 0:L
   A = [A; compositions(m,p)]; %#ok<AGROW>
end
end

function C = compositions(m,p)
if p == 1, C = m; return; end
C = zeros(0,p);
for k = 0:m
   R = compositions(m-k,p-1);
   C = [C; k*ones(size(R,1),1) R]; %#ok<AGROW>
end
end
