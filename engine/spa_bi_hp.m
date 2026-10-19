function [pdf,Fup,canc] = spa_bi_hp(Kd,tau,x,order,G)
%SPA_BI_HP Variable-precision version of spa_bi (Symbolic Math Toolbox): bivariate saddlepoint
%   approximations of orders 1..order of the density and of the orthant probability
%   P[X1 >= x1, X2 >= x2] (Sections 3-4 of the paper), computed with the current digits().
%   [pdf,Fup,canc] = spa_bi_hp(Kd,tau,x,order,G)
%   Kd:    vpa array, Kd(n1+1,n2+1) = K^{(n1,n2)}(tau), n1+n2 <= 2*order+1, Kd(1,1) = K(tau).
%   tau:   the saddlepoint (vpa 1x2), K'(tau) = x;  x: the point (1x2).
%   G:     optional {G0,G1}, Taylor coefficients at tau of the amplitude functions of a pair of
%          ratios (see spa_bi and acf_gam); pdf is then the density of the pair of ratios.
%   pdf(M), M = 1..order: density expansion with terms m <= M-1.
%   Fup(M), M = 1..order: orthant probability, terms m <= 2M-1; requires tau >= 0 (NaN otherwise).
%   canc(M): cancellation indicator (double), as in spa_bi.
%   Unlike spa_bi, the integrals I(nu) are computed exactly from the binomial sum (14) and the
%   recursion of Lemma 1, which are free of truncation error; the cancellation that makes them
%   inaccurate in double precision for large t and |nu| is harmless with enough digits.
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Saddlepoint expansions of arbitrary order for multivariate distribution functions"; free to use and modify with a citation of the paper.
if nargin < 5, G = []; end
x = vpa(x); tau = vpa(tau);
p = 2*order-1;
N = max(3*p,6*(order-1));
K0 = Kd(1,1);
K2 = [Kd(3,1) Kd(2,2); Kd(2,2) Kd(1,3)];
%
%  theta(nu1+1,nu2+1) = K_nu/nu!, |nu| >= 3;  a{r}(nu1+1,nu2+1) = [t^nu] Theta^r/r!
%
theta = vpa(zeros(N+1,N+1));
for k = 3:p+2
   for n1 = 0:k
      theta(n1+1,k-n1+1) = Kd(n1+1,k-n1+1)/(factorial(sym(n1))*factorial(sym(k-n1)));
   end
end
R = max(p,2*(order-1));
a = cell(R,1); a{1} = theta;
sz = [N+1 N+1];
for r = 2:R
   A = vpa(zeros(N+1,N+1));
   for k = 3*r:N
      for n1 = 0:k
         n2 = k-n1;
         [L1,L2] = ndgrid(0:n1,0:n2);
         L1 = L1(:); L2 = L2(:); l = L1+L2;
         keep = l >= 3 & l <= k-3*(r-1);
         L1 = L1(keep); L2 = L2(keep);
         if isempty(L1), continue; end
         i1 = sub2ind(sz,L1+1,L2+1); i2 = sub2ind(sz,n1-L1+1,n2-L2+1);
         A(n1+1,n2+1) = sum(theta(i1).*a{r-1}(i2))/r;
      end
   end
   a{r} = A;
end
Si = inv(K2);
%
%  Density: Hermite polynomials at 0
%
He = hermite2(N,Si,vpa([0 0]));
f0 = exp(K0-tau*x.')/(2*vpa(pi)*sqrt(det(K2)));
pdf = vpa(zeros(order,1));
if isempty(G)
   s = vpa(1); pdf(1) = f0;
   for M = 2:order
      m = M-1;
      for r = 1:2*m
         k = 2*m+2*r;
         n1 = (0:k)';
         i1 = sub2ind(sz,n1+1,k-n1+1);
         s = s+sum(a{r}(i1).*He(i1));
      end
      pdf(M) = f0*s;
   end
else
   szG = size(G{1}); s = vpa(0);
   for M = 1:order
      m = M-1;
      for k = 0:min(m,1)
         for r = 0:2*(m-k)
            tot = 2*(m-k+r);
            if r == 0
               m1 = (0:tot)';
               s = s+sum(G{k+1}(sub2ind(szG,m1+1,tot-m1+1)).*He(sub2ind(sz,m1+1,tot-m1+1)));
            else
               for l = 3*r:tot
                  for n1 = 0:l
                     anu = a{r}(n1+1,l-n1+1);
                     if isAlways(anu == 0), continue; end
                     m1 = (0:tot-l)';
                     gg = G{k+1}(sub2ind(szG,m1+1,tot-l-m1+1));
                     hh = He(sub2ind(sz,n1+m1+1,l-n1+tot-l-m1+1));
                     s = s+anu*sum(gg.*hh);
                  end
               end
            end
         end
      end
      pdf(M) = f0*s;
   end
end
%
%  Orthant probability (Theorem 1), with I(nu) from (14) and Lemma 1
%
Fup = vpa(nan(order,1)); canc = nan(order,1);
if any(isAlways(tau < 0)), return; end
t = (K2*tau.').';
E = exp(K0-tau*x.'+tau*K2*tau.'/2);
L = 3*p;
Jm = Jtable(L,K2,Si,t);                       % Jm(r1+2,r2+2) = J(r), r >= -1, |r| <= L-2
Inu = vpa(zeros(L+1,L+1));
for k = 0:L
   for n1 = 0:k
      n2 = k-n1;
      v1 = arrayfun(@(m) nchoosek(sym(n1),m),0:n1).*(-tau(1)).^(0:n1);   % exact binomials
      v2 = arrayfun(@(m) nchoosek(sym(n2),m),0:n2).*(-tau(2)).^(0:n2);
      Inu(n1+1,n2+1) = v1*Jm(n1+1:-1:1,n2+1:-1:1)*v2.';
   end
end
s = Inu(1,1); m = 0; big = 0;
szI = size(Inu);
for M = 1:order
   while m < 2*M-1
      m = m+1;
      for r = 1:m
         k = m+2*r;
         n1 = (0:k)';
         terms = a{r}(sub2ind(sz,n1+1,k-n1+1)).*Inu(sub2ind(szI,n1+1,k-n1+1));
         s = s+sum(terms);
         big = max(big,double(max(abs(terms))));
      end
   end
   Fup(M) = E*s; canc(M) = big/double(Inu(1,1));
end
end

function He = hermite2(N,Si,x)
%  He(n1+1,n2+1) = He_nu(x,Sigma), |nu| <= N, by the recursion of Takemura (1988)
He = vpa(zeros(N+1,N+1)); He(1,1) = 1;
y = Si*x(:);
for k = 1:N
   for n1 = 0:k
      n2 = k-n1;
      if n1 > 0
         j = 1; prev = [n1-1 n2];
      else
         j = 2; prev = [n1 n2-1];
      end
      v = y(j)*He(prev(1)+1,prev(2)+1);
      for kk = 1:2
         if prev(kk) > 0
            pp = prev; pp(kk) = pp(kk)-1;
            v = v-Si(j,kk)*prev(kk)*He(pp(1)+1,pp(2)+1);
         end
      end
      He(n1+1,n2+1) = v;
   end
end
end

function Jm = Jtable(L,S,Si,t)
%  Jm(r1+2,r2+2) = J(r) = prod_j (-D_j)^{r_j+1} Pbar(t,S), r_j >= -1, r1+r2 <= L-2: phi(t,S)He_r(t,S)
%  for r >= 0 and Lemma 1 for r with a component -1.
Jm = vpa(zeros(L+1,L+1));
He = hermite2(L,Si,t);
phi2 = exp(-t*Si*t.'/2)/(2*vpa(pi)*sqrt(det(S)));
for k = 0:L-1                                % J(r) = phi He_r, r >= 0 (|r| <= L-1 is used below)
   for n1 = 0:k
      Jm(n1+2,k-n1+2) = phi2*He(n1+1,k-n1+1);
   end
end
for i = 1:2                                  % I = {i}, J = {j}
   j = 3-i;
   Sjj = S(j,j); Sij = S(i,j);
   cv = S(i,i)-Sij^2/Sjj;
   g = exp(-t(j)^2/(2*Sjj))/sqrt(2*vpa(pi)*Sjj)*erfc((t(i)-Sij/Sjj*t(j))/sqrt(2*cv))/2;
   Jr = vpa(zeros(L,1)); Jr(1) = g;          % Jr(rj+1) = J(r), r_i = -1
   for rj = 0:L-2
      % J(r+e_j) = (t_j/S_jj) J(r) - r_j/S_jj J(r-e_j) - (S_ij/S_jj) J(r+e_i), r+e_i has r_i = 0
      if i == 1, Jri = Jm(2,rj+2); else, Jri = Jm(rj+2,2); end
      v = t(j)/Sjj*Jr(rj+1)-Sij/Sjj*Jri;
      if rj > 0, v = v-rj/Sjj*Jr(rj); end
      Jr(rj+2) = v;
   end
   if i == 1
      Jm(1,2:L+1) = Jr.';
   else
      Jm(2:L+1,1) = Jr;
   end
end
Jm(1,1) = orthant2(t,S);
end

function P = orthant2(t,S)
%  P[Z1 >= t1, Z2 >= t2], Z ~ N(0,S), by one-dimensional numerical integration in vpa
sd1 = sqrt(S(1,1)); sd2 = sqrt(S(2,2)); r = S(1,2)/(sd1*sd2);
h = t(1)/sd1; k = t(2)/sd2;
syms z real
f = exp(-z^2/2)/sqrt(2*vpa(pi))*erfc((k-r*z)/sqrt(2*(1-r^2)))/2;
up = max(double(h),0)+40;
P = vpaintegral(f,z,h,up,'RelTol',10^(-(digits-8)),'AbsTol',0,'MaxFunctionCalls',1e6);
end
