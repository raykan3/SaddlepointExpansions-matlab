function pdf = spa_ratio_pdf(KWd,gd,shat,order)
%SPA_RATIO_PDF Rice (Laplace) expansion of (1/2 pi i) int exp(K_W(s)) g(s) ds
%   through the saddlepoint shat of K_W (K_W'(shat) = 0), to orders 1..order.
%   KWd(k+1) = K_W^{(k)}(shat), k = 0..2*order;  gd(l+1) = g^{(l)}(shat), l = 0..2*order-2.
%   For the density of R = Xbar/Ybar with Y > 0: K_W(s) = K(s,-r s), g(s) = K_2(s,-r s).
%   Notation of Broda and Kan, "Saddlepoint expansions of arbitrary order", Section 5.2.
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Saddlepoint expansions of arbitrary order for multivariate distribution functions"; free to use and modify with a citation of the paper.
if ~(isscalar(order) && order >= 1 && order == round(order)), error('spa_ratio_pdf: order must be a positive integer'); end
if numel(KWd) < 2*order+1
   error('spa_ratio_pdf: need K_W^{(k)}(shat) for k = 0..%d (got %d values)',2*order,numel(KWd));
end
if numel(gd) < 2*order-1
   error('spa_ratio_pdf: need g^{(l)}(shat) for l = 0..%d (got %d values)',2*order-2,numel(gd));
end
N = 2*order;
K2 = KWd(3);
k = (3:N+2)';
theta = Kd_pad(KWd,N+2,k)./(factorial(k).*K2.^(k/2));
J = N;
d = zeros(J,J);
d(1,1:J) = theta(1:J)';
for r = 2:J
   for j = r:J
      d(r,j) = theta(1:j-r+1)'*d(r-1,j-1:-1:r-1)'/r;
   end
end
h = cumprod([1; -(1:2:4*J-1)']);
aij = @(i,j) acoef(d,h,i,j);
g0 = gd(1);
gam = zeros(N+1,1);                               % gam(l+1) = gamma~_l
for l = 0:N-2
   gam(l+1) = gd(l+1)/(factorial(l)*g0*K2^(l/2));
end
lead = exp(KWd(1))*g0/sqrt(2*pi*K2);
pdf = zeros(order,1);
s = 0;
for M = 1:order
   j = M-1;
   for kk = 0:2*j
      s = s+gam(2*j-kk+1)*aij(j,kk);
   end
   pdf(M) = lead*s;
end
end
function a = acoef(d,h,i,j)
if j == 0
   a = h(i+1);
else
   a = sum(d(1:j,j).*h(i+(1:j)'+1));
end
end
function v = Kd_pad(Kd,N,k)
%  the coefficients theta_k for k > 2*order are not used by the expansion (the table d is
%  computed further than needed); they are set to zero
   v = zeros(size(k));
   for i = 1:numel(k)
      if k(i)+1 <= numel(Kd), v(i) = Kd(k(i)+1); end
   end
end
