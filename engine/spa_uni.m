function [pdf,cdf,aij,upp,kappa] = spa_uni(Kd,shat,xbar,order,uthr)
%SPA_UNI Higher-order saddlepoint approximations of the pdf and cdf of X
%   from the derivatives of its cumulant generating function at the saddlepoint.
%   [pdf,cdf,aij,upp,kappa] = spa_uni(Kd,shat,xbar,order)
%   Kd(k+1) = K^{(k)}(shat), k = 0,...,N, with N >= 2*order+1 (= p+2, p = 2*order-1);
%   shat solves K'(shat) = xbar.
%   pdf(M), M = 1..order: Daniels' expansion with M terms (relative error O(n^-M));
%   cdf(M), M = 1..order: Hauschildt-Robinson expansion of P[X <= xbar] with the
%            terms m <= 2M-1 (relative error O(n^-M)), both signs of shat;
%   upp(M): the same expansion of P[X > xbar], computed directly (cdf + upp = 1 up to
%            rounding; use upp for small upper tails);
%   kappa(M): cancellation indicator of the cdf sum, (sum of |terms|)/|sum|; about
%            log10(kappa) digits are lost to cancellation in double precision.
%   The "n" is absorbed in K.  Notation of Broda and Kan, "Saddlepoint
%   expansions of arbitrary order", Section 5.1.
%
%   Evaluation of the integrals I_m(u), u = shat*sqrt(K''(shat)): for |u| <= uthr (default
%   2.5) by the forward recursion (36) of the paper, which is accurate there; for |u| > uthr
%   from the representation, for u > 0,
%      I_{2k}(u)/phi(u) = (-1)^k u M_k(u),  I_{2k+1}(u)/phi(u) = (-1)^k M_{k+1}(u),
%      M_j(u) = E[Z^(2j)/(u^2+Z^2)],  Z ~ N(0,1),
%   obtained by moving the contour of (35) to Re v = u, whose integrands are positive, by
%   80-point Gauss-Hermite quadrature (relative error below 2e-15 for |u| > 2.5 and m <= 33).
%   The forward recursion loses about log10(u^(2m+1)/m!) digits at index m for large |u|,
%   because the I_m are its minimal solution.  For shat < 0 the integrals along the contour to
%   the left of the pole are -(-1)^m I_m(|u|).  Both branches are scaled by phi(u), so that
%   nothing overflows for large |u|.
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Uniform saddlepoint expansions for multivariate tail probabilities"; free to use and modify with a citation of the paper.
if nargin < 5, uthr = 2.5; end
if ~(isscalar(order) && double(order) >= 1 && order == round(order)), error('spa_uni: order must be a positive integer'); end
if ~(isscalar(shat) && isscalar(xbar)), error('spa_uni: shat and xbar must be scalars'); end
Kd = Kd(:);
p = 2*order-1;
N = p+2;                                          % theta_k is needed for k <= p+2 only
if numel(Kd) < N+1
   error('spa_uni: need K^{(k)} for k = 0..%d',N);
end
K2 = Kd(3);
k = (3:N)';
theta = Kd(k+1)./(factorial(k).*K2.^(k/2));      % theta(k-2) = theta_k, k = 3..N
%
%  d(r,j) = [s^j] Theta(s)^r/r!, Theta(s) = sum_k theta_{k+2} s^k
%
J = N-2;
d = zeros(J,J);                                   % d(r,j), r,j = 1..J (d(0,0)=1 implicit)
d(1,1:J) = theta(1:J)';
for r = 2:J
   for j = r:J
      d(r,j) = theta(1:j-r+1)'*d(r-1,j-1:-1:r-1)'/r;
   end
end
h = cumprod([1; -(1:2:4*J-1)']);                  % h(i+1) = h_i = (-2)^i (1/2)_i
aij = @(i,j) acoef(d,h,i,j);
%
%  Daniels density
%
f0 = exp(Kd(1)-shat*xbar)/sqrt(2*pi*K2);
pdf = zeros(order,1);
s = 0;
for M = 1:order
   if M == 1
      s = 1;
   else
      s = s+aij(M-1,2*(M-1));
   end
   pdf(M) = f0*s;
end
%
%  Hauschildt-Robinson cdf: R(m+1) = I_m(u)/phi(u) along the contour used (right of the
%  pole for shat >= 0, left of it for shat < 0), and E*phi(u) = exp(K - shat*xbar)/sqrt(2 pi)
%
u = shat*sqrt(K2);
Ephi = exp(Kd(1)-shat*xbar)/sqrt(2*pi);
nI = 3*p+1;
if abs(u) <= uthr
   R = zeros(nI,1);
   if shat >= 0
      R(1) = normcdf(-u)/normpdf(u);
   else
      R(1) = -normcdf(u)/normpdf(u);                % contour to the left of the pole
   end
   for i = 0:3*p-1
      if rem(i,2) == 0
         R(i+2) = h(i/2+1)-u*R(i+1);
      else
         R(i+2) = -u*R(i+1);
      end
   end
else
   R = Iherm(abs(u),nI-1);                         % I_m(|u|)/phi(|u|), m = 0..nI-1
   if shat < 0
      R = -((-1).^(0:nI-1)').*R;
   end
end
cdf = zeros(order,1); upp = cdf; kappa = cdf;
s = R(1); sa = abs(R(1));
m = 0;
for M = 1:order
   while m < 2*M-1
      m = m+1;
      for r = 1:m
         term = d(r,m)*R(m+2*r+1);
         s = s+term; sa = sa+abs(term);
      end
   end
   if shat >= 0
      upp(M) = Ephi*s; cdf(M) = 1-upp(M);
   else
      cdf(M) = -Ephi*s; upp(M) = 1-cdf(M);
   end
   kappa(M) = sa/abs(s);
end
end

function R = Iherm(u,mmax)
%  R(m+1) = I_m(u)/phi(u), m = 0..mmax, u > 0:  R(2k+1) = (-1)^k u M_k, R(2k+2) = (-1)^k M_{k+1},
%  M_j = E[Z^(2j)/(u^2+Z^2)], Z ~ N(0,1), by 80-point Gauss-Hermite quadrature (positive terms).
persistent xg wg
if isempty(xg)
   n = 80; b = sqrt(1:n-1);                         % Golub-Welsch, weight exp(-x^2/2)/sqrt(2 pi)
   [V,D] = eig(diag(b,1)+diag(b,-1));
   [xg,o] = sort(diag(D)); wg = V(1,o)'.^2;
end
q = wg./(u^2+xg.^2); x2 = xg.^2;
J = floor(mmax/2)+1; M = zeros(J+1,1);
for j = 0:J
   M(j+1) = sum(q); q = q.*x2;
end
R = zeros(mmax+1,1);
for m = 0:mmax
   k = floor(m/2);
   if rem(m,2) == 0
      R(m+1) = (-1)^k*u*M(k+1);
   else
      R(m+1) = (-1)^k*M(k+2);
   end
end
end

function a = acoef(d,h,i,j)
%  a_{i,j} = sum_{k=0}^j d(k,j) h_{i+k}, with d(0,0) = 1, d(0,j) = 0 for j > 0
if j == 0
   a = h(i+1);
else
   a = sum(d(1:j,j).*h(i+(1:j)'+1));
end
end
