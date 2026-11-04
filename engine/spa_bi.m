function [pdf,Fup,tau,canc,kappa,qerr,info] = spa_bi(Kfun,x,order,tau0,gamfun,nG,opts)
%SPA_BI Bivariate saddlepoint approximations of arbitrary order (Broda and Kan,
%   "Uniform saddlepoint expansions", Sections 3-4), with n = 1.
%   [pdf,Fup,tau,canc,kappa,qerr,info] = spa_bi(Kfun,x,order,tau0,gamfun,nG,opts)
%   Kfun(nu,t): derivative of order nu = [nu1 nu2] of the cgf K at t = [t1 t2]
%               (nu = [0 0] gives K(t)); must return NaN outside the domain of K.
%   x:          the point [x1 x2];  tau0: starting value for the saddlepoint (default [0 0]).
%   gamfun:     optional.  gamfun(tau) returns a cell array {G0,G1} of Taylor coefficients
%               at tau of the amplitude functions g_0, g_1 of a pair of ratios (Appendix A.1),
%               G_k(mu1+1,mu2+1) = coefficient gamma_mu of t^mu in g_k(tau+t), which enters
%               b_j(g_k) of Corollary 3, |mu| <= 2(order-1); pdf is then the
%               density of the pair of ratios (x should be [0 0]).
%   pdf(M), M = 1..order: density expansion with terms m <= M-1 (relative error O(n^-M)).
%   Fup(M), M = 1..order: P[X1 >= x1, X2 >= x2] with terms m <= 2M-1 (relative error O(n^-M)),
%               valid when the saddlepoint tau is in the closed positive orthant (NaN otherwise;
%               use inclusion-exclusion, Section 3.3, or spa_orthant2).
%   canc(M):    cancellation indicator of Fup(M): the largest |term| among its terms divided by
%               I(0).
%   kappa(M):   (sum of |terms| of Fup(M))/|Fup(M)/E|; about log10(kappa) digits are lost to
%               cancellation in the final sum in double precision.  kappa does not measure the
%               accuracy of the Taylor coefficients themselves (see info.coef_err_cdf).
%   nG:         number of Gauss-Legendre nodes in each coordinate (per piece, see below) for the
%               orthant quadrature (case (b) below; default 200).
%   qerr(M):    quadrature check of Fup(M): |Fup(M) computed with half the nodes - Fup(M)|/|Fup(M)|,
%               all retained terms included (NaN when the leading integral is not resolved).  It
%               compares two quadratures and is not a bound; it says nothing about the
%               asymptotic error of the expansion.
%   info:       struct: status ('ok', 'negative saddlepoint component (use spa_orthant2)',
%               'leading integral not resolved', 'integration not resolved' when qerr(order)
%               exceeds opts.qtol after the node escalation below, 'coefficients inaccurate'
%               when info.coef_err_pdf or info.coef_err_cdf exceeds 1e-6, 'probability below the
%               double-precision range (use opts.logscale)' when the Chernoff bound
%               exp(K(tau) - tau'x - c) is below realmin, Fup then being returned as zero;
%               several are joined by '; '); nG (the number of nodes used); coef_err_pdf(M),
%               coef_err_cdf(M): estimates
%               of the relative errors of pdf(M) and Fup(M) caused by rounding in the formation of
%               the whitened Taylor coefficients (first-order propagation of random
%               perturbations of the size of the rounding errors of the substitution below; with
%               opts.Kwhite only the rounding of the supplied derivatives is included); saddle
%               (see spa_saddle).
%   opts:       optional struct.  opts.Kwhite = @(nu,u,T): derivative of order nu at u of the
%               cgf K_Y(u) = K(T u) of Y = T' X for a 2 x 2 matrix T (see below; with
%               T = L^{-T}, Y = L^{-1} X); when it is absent the Taylor coefficients of K_Y are
%               obtained from those of K by polynomial substitution, which loses digits when
%               K'' is ill conditioned (info.coef_err_pdf, info.coef_err_cdf estimate the loss).
%               opts.gamwhite = @(u,T): the amplitude coefficients of gamfun expressed in the
%               whitened variable (otherwise substitution).
%               opts.direct = true evaluates the orthant expansion also when tau has negative
%               components, by the orthant-integral form of I(nu) (the analytic continuation of
%               the formula; this is the "direct" evaluation of Section 3.3 and Table S6, which
%               has the wrong exponent; use spa_orthant2 for the probability itself).
%               opts.logscale = c returns Fup scaled by exp(-c), for probabilities beyond the
%               range of double precision (default 0).
%               opts.qtol (default 1e-8) and opts.nGmax (default 4 nG): in case (b), while
%               qerr(order) > qtol the number of nodes is doubled, up to nGmax.
%   Fup is NaN, and qerr is NaN, when the leading integral I(0) cannot be resolved by the
%   quadrature (it comes out zero or not finite), instead of a misleading zero.
%   The density prefactor exp(K - tau'x)/(2 pi det(K'')^(1/2)) is formed in the log domain from
%   the Cholesky factor of K'', so that it does not overflow or underflow when det(K'') does.
%   The saddlepoint is found by spa_saddle (Newton's method with backtracking, convergence
%   measured by the Newton decrement); an error is raised if it does not converge.
%
%   Whitened evaluation.  With K2 = K''(tau) = L L' (Cholesky, L lower triangular) and
%   T = L^{-T}, the terms of each order are evaluated for Y = L^{-1} X: the Taylor coefficients
%   of K_Y at uhat = L' tau replace those of K at tau, the Hermite polynomials He_nu(w,K2) are
%   replaced by products of univariate ones in v = L^{-1} w, and the orthant integrals I(nu) by
%   integrals of these products over the same orthant {z >= t}, at the same quadrature nodes.
%   The sums of the terms of a given order are the same in exact arithmetic; the terms are not,
%   and in the whitened form they cancel much less when the components of X are strongly
%   correlated under the tilted distribution (Section 5 of the paper on conditional
%   expansions, and Appendix C of this paper), provided the whitened coefficients are formed
%   accurately (directly by opts.Kwhite, or by substitution when info.coef_err_* is small).
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Uniform saddlepoint expansions for multivariate tail probabilities"; free to use and modify with a citation of the paper.
if nargin < 4 || isempty(tau0), tau0 = [0 0]; end
if nargin < 5, gamfun = []; end
if nargin < 6 || isempty(nG), nG = 200; end
if nargin < 7 || isempty(opts), opts = struct(); end
if ~isfield(opts,'Kwhite'), opts.Kwhite = []; end
if ~isfield(opts,'gamwhite'), opts.gamwhite = []; end
if ~isfield(opts,'direct'), opts.direct = false; end
if ~isfield(opts,'logscale'), opts.logscale = 0; end
if ~isfield(opts,'qtol'), opts.qtol = 1e-8; end
if ~isfield(opts,'nGmax'), opts.nGmax = 4*nG; end
if ~isa(Kfun,'function_handle'), error('spa_bi: Kfun must be a function handle'); end
if ~isnumeric(x) || numel(x) ~= 2 || ~all(isfinite(x)), error('spa_bi: x must be a finite 1 x 2 vector'); end
if ~isscalar(order) || order < 1 || order ~= round(order), error('spa_bi: order must be a positive integer'); end
if ~isscalar(nG) || nG < 4 || nG ~= round(nG), error('spa_bi: nG must be an integer >= 4'); end
x = reshape(double(x),1,2);
[tau,sinfo] = spa_saddle(Kfun,x,tau0);
info = struct('status','ok','nG',NaN,'coef_err_pdf',nan(order,1),'coef_err_cdf',nan(order,1), ...
   'saddle',sinfo);
K0 = Kfun([0 0],tau);
K2 = [Kfun([2 0],tau) Kfun([1 1],tau); Kfun([1 1],tau) Kfun([0 2],tau)];
L = chol(K2,'lower'); T = inv(L.');                     % s = T u, T upper triangular
logdet2 = sum(log(diag(L)));                            % log det(K2)^(1/2)
uh = (L.'*tau.').';                                     % saddlepoint of K_Y
p = 2*order-1;
N = max(3*p, 6*(order-1));                              % largest |nu| needed
%
%  theta(nu1+1,nu2+1) = Taylor coefficient of K_Y at uhat, |nu| >= 3;
%  a{r}(nu1+1,nu2+1) = [u^nu] G^r/r!
%
theta = zeros(N+1,N+1);
[D1,D2] = ndgrid(0:N,0:N); degm = D1+D2;                % total degree of each entry
if isempty(opts.Kwhite)
   for k = 3:p+2
      for n1 = 0:k
         theta(n1+1,k-n1+1) = Kfun([n1 k-n1],tau)/(factorial(n1)*factorial(k-n1));
      end
   end
   thabs = subs2(abs(theta),abs(T));                    % rounding-error scale of the substitution
   theta = subs2(theta,T);
   dth = (degm+2)*eps.*thabs;
else
   for k = 3:p+2
      for n1 = 0:k
         theta(n1+1,k-n1+1) = opts.Kwhite([n1 k-n1],uh,T)/(factorial(n1)*factorial(k-n1));
      end
   end
   dth = 2*eps*abs(theta);
end
R = max(p,2*(order-1));
a = cell(R,1); a{1} = theta;
for r = 2:R
   a{r} = zeros(N+1,N+1);
   for k = 3*r:N
      for n1 = 0:k
         n2 = k-n1; s = 0;
         for j = 3:k-3*(r-1)
            for l1 = max(0,j-n2):min(j,n1)
               l2 = j-l1;
               s = s+theta(l1+1,l2+1)*a{r-1}(n1-l1+1,n2-l2+1);
            end
         end
         a{r}(n1+1,n2+1) = s/r;
      end
   end
end
%
%  Density: He_mu(0,I) = He_{mu1}(0) He_{mu2}(0)
%
h0 = zeros(N+1,1); h0(1) = 1;
for m = 2:2:N, h0(m+1) = -(m-1)*h0(m-1); end
He0 = h0*h0.';
f0 = exp(K0-tau*x'-log(2*pi)-logdet2);
%  first-order propagation of perturbations of the whitened coefficients of the size of their
%  rounding errors: da{r} = a{r-1} * dtheta (product of polynomials), the derivative of G^r/r!
st = RandStream('mt19937ar','Seed',1); ndraw = 2;
da = cell(ndraw,1);
for q = 1:ndraw
   dt = dth.*sign(rand(st,N+1,N+1)-0.5);
   da{q} = cell(R,1); da{q}{1} = dt;
   for r = 2:R
      P = conv2(a{r-1},dt); da{q}{r} = P(1:N+1,1:N+1);
   end
end
pdf = zeros(order,1);
if isempty(gamfun)
   s = 1; pdf(1) = f0; ds = zeros(ndraw,1); info.coef_err_pdf(1) = 0;
   for M = 2:order
      m = M-1;
      for r = 1:2*m
         k = 2*m+2*r;
         for n1 = 0:k
            s = s+a{r}(n1+1,k-n1+1)*He0(n1+1,k-n1+1);
            for q = 1:ndraw, ds(q) = ds(q)+da{q}{r}(n1+1,k-n1+1)*He0(n1+1,k-n1+1); end
         end
      end
      pdf(M) = f0*s; info.coef_err_pdf(M) = max(abs(ds))/abs(s);
   end
else
   % density of a pair of ratios (Corollary 3): coefficient of n^-m = sum_{k<=min(m,1)} b_{m-k}(g_k),
   %    b_j(g_k) = sum_r sum_{|nu|+|mu|=2(j+r)} a_{r,nu} gamma_mu He_{nu+mu}(0,I), gamma_mu the
   %    Taylor coefficients of g_k, a_{0,0} = 1, in the whitened variable
   if isempty(opts.gamwhite)
      G = gamfun(tau); dG = cell(1,2);
      for k = 1:2
         Gk = padsq(G{k},N+1);
         dG{k} = (degm+2)*eps.*subs2(abs(Gk),abs(T)); G{k} = subs2(Gk,T);
      end
   else
      G = opts.gamwhite(uh,T);
      dG = cell(1,2);
      for k = 1:2
         G{k} = padsq(G{k},N+1); dG{k} = 2*eps*abs(G{k});
      end
   end
   dGq = cell(ndraw,2);
   for q = 1:ndraw, for k = 1:2, dGq{q,k} = dG{k}.*sign(rand(st,N+1,N+1)-0.5); end, end
   s = 0; ds = zeros(ndraw,1);
   for M = 1:order
      m = M-1;
      for k = 0:min(m,1)
         for r = 0:2*(m-k)
            tot = 2*(m-k+r);
            if r == 0
               P = G{k+1};
            else
               P = conv2(a{r},G{k+1}); P = P(1:N+1,1:N+1);       % product polynomial
            end
            for m1 = 0:tot
               s = s+P(m1+1,tot-m1+1)*He0(m1+1,tot-m1+1);
            end
            for q = 1:ndraw                                    % its perturbation
               if r == 0
                  dP = dGq{q,k+1};
               else
                  dP = conv2(da{q}{r},G{k+1})+conv2(a{r},dGq{q,k+1}); dP = dP(1:N+1,1:N+1);
               end
               for m1 = 0:tot
                  ds(q) = ds(q)+dP(m1+1,tot-m1+1)*He0(m1+1,tot-m1+1);
               end
            end
         end
      end
      pdf(M) = f0*s; info.coef_err_pdf(M) = max(abs(ds))/abs(s);
   end
end
%
%  Orthant probability.  The integrals I(nu) of Theorem 1 are computed in one of two ways.
%  (a) If u_j = tau_j/sd(W_j) > 2.5 for both j, where W ~ N(0,Sigma^{-1}), from the
%      representation obtained by moving the contour to Re s = tau,
%         I(nu) = phi(t,Sigma) Re E[(iW)^nu/((tau1+iW1)(tau2+iW2))],
%      with W = T V, V standard normal, and (iW)^nu replaced by (iV)^mu in the whitened form, by
%      80 x 80 Gauss-Hermite quadrature; the integrand is analytic in a strip of half-width at
%      least min(u_j), so the quadrature converges exponentially.
%  (b) Otherwise from the orthant-integral form int_{z >= t} He_mu(L^{-1}(z-t)) phi(z,Sigma) dz
%      by Gauss-Legendre quadrature in conditioned coordinates: G nodes in z1 over
%      [t1, t1 + 16 sd1] (around the mode of the restricted density when tau has a negative
%      component and opts.direct is set), and for each node G nodes in z2 over the part of
%      [mu(z1) - 16 s2, mu(z1) + 16 s2] above t2, where mu(z1) and s2 are the conditional mean
%      and standard deviation of z2 given z1; this resolves the Gaussian ridge for correlations
%      up to about 0.999 in absolute value (it does not for correlations within 1e-6 of +-1).
%      The exponential factor is combined with the normal density in the log domain.
%  Inu(mu1+1,mu2+1) holds the whitened integral, or its value divided by phi(t,Sigma) in case
%  (a), with E scaled to match.
%
Fup = nan(order,1); canc = nan(order,1); kappa = nan(order,1); qerr = nan(order,1);
if max(info.coef_err_pdf) > 1e-6, info.status = 'coefficients inaccurate'; end
if any(tau < 0) && ~opts.direct                  % tau_j = 0 is allowed (limiting form)
   info.status = addstatus(info.status,'negative saddlepoint component (use spa_orthant2)'); return
end
if all(tau >= 0) && K0-tau*x'-opts.logscale < log(realmin)
   %  Chernoff bound P[X >= x] <= exp(K(tau) - tau'x): the (scaled) probability is below the
   %  double-precision range, and is returned as zero, as an underflow
   Fup = zeros(order,1); qerr = zeros(order,1); canc = zeros(order,1); kappa = ones(order,1);
   info.status = addstatus(info.status,'probability below the double-precision range (use opts.logscale)');
   return
end
Lh = 3*p;
Dk = diag(1./sqrt(diag(K2))); Si = Dk*inv(Dk*K2*Dk)*Dk;   % inverse insensitive to the units
uu = tau./sqrt(diag(Si)');                     % tau_j / sd(W_j)
if all(uu > 2.5)
   [Inu,Ih] = Ifreq(T,tau,Lh,80);
   E = exp(K0-tau*x'-opts.logscale-log(2*pi)-logdet2);
   info.nG = 80;
   if resolved(Inu)
      [Fup,canc,kappa,dF] = osums(Inu); Fh = osums(Ih); qerr = abs(Fh-Fup)./abs(Fup);
   end
else
   %  the exponential factor E = exp(K0 - tau x + tau' K2 tau/2) and the normal density at the
   %  nodes are combined in the log domain, so that their product, the quantity of interest,
   %  is formed without overflow or underflow of the factors (E can overflow while the
   %  integral underflows when one coordinate is far in its tail and another near its mean)
   t = (K2*tau')';                             % t = Sigma tau, Sigma = K2
   logE = K0-tau*x'+tau*K2*tau'/2-opts.logscale; E = 1;
   G = nG; Ih = [];
   while true
      Inu = Ileg(K2,L,t,G,logE,Lh);
      if ~resolved(Inu), break, end
      if isempty(Ih), Ih = Ileg(K2,L,t,floor(G/2),logE,Lh); end
      [Fup,canc,kappa,dF] = osums(Inu); Fh = osums(Ih); qerr = abs(Fh-Fup)./abs(Fup);
      if ~(qerr(order) > opts.qtol) || 2*G > opts.nGmax, break, end
      G = 2*G; Ih = Inu;                       % escalate; the previous grid is the half grid
   end
   info.nG = G;
end
if ~resolved(Inu)                              % the leading integral is not resolved
   info.status = addstatus(info.status,'leading integral not resolved'); return
end
info.coef_err_cdf = dF;
if max(dF) > 1e-6 && ~contains(info.status,'coefficients'), info.status = addstatus(info.status,'coefficients inaccurate'); end
if ~(qerr(order) <= opts.qtol), info.status = addstatus(info.status,'integration not resolved'); end

   function [F,cn,kp,dFr] = osums(In)
      %  Fup(M), M = 1..order, from the integrals In, with the cancellation indicators and the
      %  relative coefficient-error estimates
      oS = In(1,1); oSa = abs(oS); om = 0; oBig = 0; oD = zeros(ndraw,1);
      F = zeros(order,1); cn = F; kp = F; dFr = F;
      for oM = 1:order
         while om < 2*oM-1
            om = om+1;
            for orr = 1:om
               ok_ = om+2*orr;
               for on1 = 0:ok_
                  oT = a{orr}(on1+1,ok_-on1+1)*In(on1+1,ok_-on1+1);
                  oS = oS+oT; oSa = oSa+abs(oT); oBig = max(oBig,abs(oT));
                  for oq = 1:ndraw, oD(oq) = oD(oq)+da{oq}{orr}(on1+1,ok_-on1+1)*In(on1+1,ok_-on1+1); end
               end
            end
         end
         F(oM) = E*oS; cn(oM) = oBig/In(1,1); kp(oM) = oSa/abs(oS); dFr(oM) = max(abs(oD))/abs(oS);
      end
   end
end

function ok = resolved(Inu)
ok = isfinite(Inu(1,1)) && Inu(1,1) > 0;
end

function s = addstatus(s,msg)
if strcmp(s,'ok'), s = msg; else, s = [s '; ' msg]; end
end

function Inu = Ileg(K2,L,t,G,logE,Lh)
%  whitened orthant integrals by Gauss-Legendre quadrature in conditioned coordinates (condgrid)
[W1,W2,phiW] = condgrid(K2,t,G,logE);
V1 = W1/L(1,1); V2 = (W2-L(2,1)*V1)/L(2,2);          % v = L^{-1} w
H1 = hermite1(V1,Lh); H2 = hermite1(V2,Lh);         % H1{m+1} = He_m(v1)
Inu = zeros(Lh+1,Lh+1);
for k = 0:Lh
   for n1 = 0:k
      Inu(n1+1,k-n1+1) = sum(H1{n1+1}(:).*H2{k-n1+1}(:).*phiW(:));
   end
end
end

function Q = padsq(P,n)
%  P padded with zeros, or truncated, to n x n
Q = zeros(n,n); m1 = min(n,size(P,1)); m2 = min(n,size(P,2)); Q(1:m1,1:m2) = P(1:m1,1:m2);
end

function Q = subs2(P,T)
%  coefficients of P(T u) for the bivariate polynomial P (P(n1+1,n2+1) = [s^(n1,n2)] P),
%  T upper triangular: s1 = T11 u1 + T12 u2, s2 = T22 u2
n = size(P,1); Q = zeros(n);
C = abs(pascal(n,1));                                   % C(n+1,k+1) = nchoosek(n,k)
for n1 = 0:n-1
   for n2 = 0:n-1-n1
      c = P(n1+1,n2+1); if c == 0, continue, end
      for j = 0:n1
         Q(j+1,n1-j+n2+1) = Q(j+1,n1-j+n2+1)+c*C(n1+1,j+1)*T(1,1)^j*T(1,2)^(n1-j)*T(2,2)^n2;
      end
   end
end
end

function H = hermite1(V,L)
%  H{m+1} = He_m(V), m = 0..L, by the three-term recursion
H = cell(L+1,1); H{1} = ones(size(V)); if L == 0, return, end
H{2} = V;
for m = 2:L, H{m+1} = V.*H{m}-(m-1)*H{m-1}; end
end

function [Inu,Ih] = Ifreq(T,tau,L,G)
%  Inu(mu1+1,mu2+1) = Re E[(iV)^mu/((tau1+iW1)(tau2+iW2))], V ~ N(0,I), W = T V ~ N(0,Sigma^{-1}),
%  |mu| <= L, by G x G Gauss-Hermite quadrature, and Ih with G/2 nodes
Inu = run(G); Ih = run(floor(G/2));
   function I = run(G)
      [xh,wh] = gauher(G);
      [Z1,Z2] = ndgrid(xh,xh); WW = wh*wh';
      W1 = T(1,1)*Z1+T(1,2)*Z2; W2 = T(2,2)*Z2;
      q = WW./((tau(1)+1i*W1).*(tau(2)+1i*W2));
      P1 = cell(L+1,1); P2 = P1; P1{1} = ones(G); P2{1} = ones(G);
      for k = 1:L
         P1{k+1} = P1{k}.*(1i*Z1); P2{k+1} = P2{k}.*(1i*Z2);
      end
      I = zeros(L+1,L+1);
      for k = 0:L
         for n1 = 0:k
            I(n1+1,k-n1+1) = real(sum(q.*P1{n1+1}.*P2{k-n1+1},'all'));
         end
      end
   end
end

function [x,w] = gauher(n)
%  Gauss-Hermite nodes and weights for the weight exp(-x^2/2)/sqrt(2 pi) (Golub-Welsch)
b = sqrt(1:n-1);
[V,D] = eig(diag(b,1)+diag(b,-1));
[x,i] = sort(diag(D)); w = V(1,i)'.^2;
end

function [W1,W2,phiW] = condgrid(S,t,G,logE)
%  Quadrature grid for E int_{z >= t} h(z-t) phi(z,S) dz in conditioned coordinates:
%  W1, W2 = z - t at the nodes, phiW = E phi(z,S) times the quadrature weights, with
%  E = exp(logE) combined with phi in the log domain.  The nodes in z1 cover 16 standard
%  deviations on either side of the z1-coordinate m1 of the mode of phi(z,S) on {z >= t}
%  (m1 = t1 when the saddlepoint is in the positive orthant, so that the range is
%  [t1, t1 + 16 sd1]), split, with G nodes on each piece, at the edges of the layer in which the
%  truncation of z2 given z1 at t2 is partial when that layer is narrow (correlations near +-1,
%  layerbp) and where the density of z1 has decayed when t1 is far in its tail (tailbp); for each
%  z1 the G nodes in z2 cover the part above t2 of an interval of 16 conditional standard
%  deviations around the conditional mean of z2 given z1.
if nargin < 4, logE = 0; end
[xg,wg] = gauleg(G);
sd1 = sqrt(S(1,1)); b = S(1,2)/S(1,1); s2 = sqrt(S(2,2)-S(1,2)^2/S(1,1));
m1 = modez1(S,t);
a1 = max(t(1),m1-16*sd1); b1 = m1+16*sd1;
[z1,w1] = glpieces(a1,b1,[layerbp(a1,b1,b,0,s2,t(2),16) tailbp(a1,b1,0,sd1,a1,16)],xg,wg);
n1 = numel(z1);
mu = b*z1;
lo = max(t(2),mu-16*s2); hi = max(mu,t(2))+16*s2;
Z2 = (lo+hi)'/2+(hi-lo)'/2*xg;                             % n1 x G, row i: nodes for z1(i)
W2w = (hi-lo)'/2*wg;
Z1 = repmat(z1',1,G);
lphi = logE-(z1'.^2)/(2*S(1,1))-log(sqrt(2*pi)*sd1) ...         % log E phi(z1)
     -(Z2-mu').^2/(2*s2^2)-log(sqrt(2*pi)*s2);                   %  + log phi(z2 | z1)
phiW = exp(lphi).*w1'.*W2w;
W1 = Z1-t(1); W2 = Z2-t(2);
if n1 == 0, W1 = zeros(1,G); W2 = zeros(1,G); phiW = zeros(1,G); end
end

function [z,w] = glpieces(lo,hi,bp,xg,wg)
%  Gauss-Legendre nodes and weights (rows) on [lo,hi], split at the breakpoints bp that lie inside
%  (lo,hi), with the full number of nodes on each piece.  Nothing is dropped: the split only adds
%  nodes where the integrand changes on a scale much shorter than the interval (layerbp, tailbp).
%  Without breakpoints the nodes are those of [lo,hi].  Double or variable precision (decisions
%  in double, nodes in the class of the inputs).
xg = xg(:).'; wg = wg(:).';
e = {lo}; bd = zeros(1,0); bk = cell(1,0);
for c = 1:numel(bp)
   v = double(bp(c));
   if v > double(lo) && v < double(hi), bd(end+1) = v; bk{end+1} = bp(c); end %#ok<AGROW>
end
[~,o] = sort(bd); e = [e bk(o) {hi}];
z = xg(1:0); w = wg(1:0);
for k = 1:numel(e)-1
   if ~(double(e{k+1}) > double(e{k})), continue, end
   z = [z (e{k}+e{k+1})/2+(e{k+1}-e{k})/2*xg]; w = [w (e{k+1}-e{k})/2*wg]; %#ok<AGROW>
end
end

function bp = layerbp(lo,hi,slope,icpt,sd,tj,R)
%  edges of the layer of the outer coordinate z in which a later coordinate, normal given z with
%  mean icpt + slope z and standard deviation sd, has its truncation point tj within R sd of its
%  mean (width 2 R sd/|slope|), when the layer is narrower than a quarter of [lo,hi], as when a
%  correlation is near +-1; otherwise none
bp = zeros(1,0,'like',lo+icpt);
if double(slope) == 0 || ~isfinite(double(slope)), return, end
zA = (tj-R*sd-icpt)/slope; zB = (tj+R*sd-icpt)/slope;
if abs(double(zB-zA)) < double(hi-lo)/4, bp = [zA zB]; end
end

function bp = tailbp(lo,hi,m,s,t,R)
%  for a coordinate N(m,s^2) truncated to [t,Inf) with t far above m, the point
%  m + {(t - m)^2 + R^2 s^2}^(1/2) beyond which its density is below exp(-R^2/2) times that at t,
%  when it is within a quarter of [lo,hi] of lo; otherwise none
bp = zeros(1,0,'like',lo+m);
if double(t) <= double(m), return, end
h = m+sqrt((t-m)^2+R^2*s^2);
if double(h-lo) < double(hi-lo)/4, bp = h; end
end

function m1 = modez1(S,t)
%  first coordinate of the maximizer of -z'S^{-1}z/2 on {z >= t} (2 x 2): the vertex when
%  S^{-1} t >= 0, otherwise the best point on the two edges or the origin
Dk = diag(1./sqrt(diag(S))); P = Dk*inv(Dk*S*Dk)*Dk; cand = t(:);
if ~all(P*t(:) >= 0)
   cand = [];
   if all(t <= 0), cand = [0;0]; end
   z2 = max(t(2),S(1,2)/S(1,1)*t(1)); cand = [cand [t(1);z2]];   % edge z1 = t1
   z1 = max(t(1),S(1,2)/S(2,2)*t(2)); cand = [cand [z1;t(2)]];   % edge z2 = t2
   q = sum(cand.*(P*cand),1); [~,i] = min(q); cand = cand(:,i);
end
m1 = cand(1);
end

function [x,w] = gauleg(n)
%  Gauss-Legendre nodes and weights on [-1,1] (Golub-Welsch)
k = 1:n-1; b = k./sqrt(4*k.^2-1);
[V,D] = eig(diag(b,1)+diag(b,-1));
[x,i] = sort(diag(D)'); w = 2*V(1,i).^2;
end
