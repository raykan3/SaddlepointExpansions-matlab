function [P,f,info] = spa_cond(Kfun,x,A,order,opts)
%SPA_COND Conditional saddlepoint expansions of arbitrary order: the conditional distribution
%   function P[X_A >= x_A | X_B = x_B] and the conditional density f(x_A | x_B) of a random
%   d-vector X with cumulant generating function K, B the complement of A, for continuous and
%   lattice variables (Broda and Kan, "Conditional and lattice saddlepoint expansions of
%   arbitrary order", Theorems 3 to 5), with n = 1.
%   [P,f,info] = spa_cond(Kfun,x,A,order,opts)
%   Kfun(nu,t): derivative of order nu (1 x d nonnegative integers) of K at t (1 x d); nu = 0 gives
%               K(t); NaN outside the domain of K.
%   x:          the point (1 x d).
%   A:          indices (within 1..d) of the components whose tail is wanted; the others, B, are
%               conditioned on.  With B empty, P is the orthant probability P[X >= x] (for lattice
%               variables Corollary 1).
%   P(M):       P[X_A >= x_A | X_B = x_B] of order M = 1..order (Theorem 3): the terms m <= 2M-1
%               of the numerator, the orthant-probability integral with the poles in the
%               components of A only, over the density expansion of X_B at its own saddlepoint
%               with M terms; relative error O(n^-M).  When the saddlepoint has negative components
%               in A the inclusion-exclusion of Remark 1 of the paper is applied: every term is the
%               conditional probability, given the same X_B, of the components of A that are kept,
%               some negated, with the others marginalized, and its own saddlepoint; the term with
%               no tail component is 1.
%   f(M):       the conditional density f(x_A | x_B) of order M (Theorem 4), the ratio of the
%               density expansions of X and of X_B with the terms m <= M-1.
%   info:       struct: tau (the saddlepoint of K at x, at the continuity-corrected point in the
%               lattice case), tauf (the saddlepoint at x used by the density), taut (the
%               saddlepoint of the marginal of X_B, embedded as (0_A, s_B)); kappaP, kappaf, kappaB
%               (cancellation indicators, sum of the absolute values of the terms over the
%               absolute value of their sum, of the numerator of P, the terms of the
%               inclusion-exclusion included, of the density expansion of X and of that of X_B);
%               err(M) (the change of P(M) with half the quadrature nodes: a comparison of two
%               quadratures, not a bound, and not the asymptotic error); Pp(j+1), j = 0..2*order-1
%               (the expansions with the terms of the numerator up to n^(-j/2) and floor(j/2)+1
%               terms of the denominator, P(M) = Pp(2M); Pp(1) is the leading term, for one tail
%               component the conditional analogue (u/z) exp((u^2-w^2)/2) Phibar(u) of the
%               Hauschildt-Robinson approximation in the notation of Skovgaard (1987)); status
%               ('ok' or one or more of 'large cancellation in the numerator' (or 'in the
%               density', 'in the denominator': an indicator above 1e-8/eps) and 'integration not
%               resolved' (err(order) above 1e-8 |P(order)|), joined by '; ').
%   opts (optional struct):
%     lattice   (false) the components in A are integer-valued and so is x_A (Theorem 5): P is
%               then P[S_A >= k_A | S_B = k_B] with the continuity correction (the saddlepoint of
%               the numerator is at k_A - 1/2) and the amplitude prod_{j in A} s_j/(2 sinh(s_j/2)),
%               and f is P[S_A = k_A | S_B = k_B] (the same formula as the density).  The
%               components in B may be lattice or continuous.
%     sinh      ('saddle') in the lattice case the amplitude is expanded about the saddlepoint
%               (Section 4 of the paper); 'origin' expands s/(2 sinh(s/2)) about zero as
%               1 - s^2/24 + 7 s^4/5760 - ..., keeping the terms up to s^(2M-2) in the
%               approximation of order M, as in Kolassa (2003, 2006).
%     nG        (64) Gauss-Legendre nodes per panel for the orthant integrals.
%     tau0      starting value of the saddlepoint iterations (default zeros).
%     Kwhite    ([]) optional function handle @(nu,u,M): derivative of order nu (1 x k) at u
%               (1 x k) of K_Y(u) = K(M u), for a d x k matrix M (as for spa_mv and spa_orthant).
%               When given, the Taylor coefficients in the canonical and whitened coordinates are
%               taken from it; otherwise they are obtained by substituting the linear maps into
%               the Taylor polynomial of K at the saddlepoint, which loses digits when K'' is ill
%               conditioned.
%     tail, density  (true, true) compute P and f (what is not computed is NaN).
%     tauf      ([]) the saddlepoint of x, when it is known: the density is then evaluated there
%               without the saddlepoint iteration.
%   Computation (Section 5 of the paper): the Taylor coefficients of K at the saddlepoint are
%   taken in the canonical coordinates Y_A = X_A - C X_B, Y_B = L^{-1} X_B, C = K''_AB K''_BB^{-1},
%   L L' = K''_BB, in which the tilted covariance matrix is block diagonal, diag(K''_{AA.B}, I);
%   the conditional probability is unchanged, and the terms no longer cancel (by up to sixteen
%   orders of magnitude in the original coordinates in the examples of the paper).  The tail
%   components are whitened as well, K''_{AA.B} = L_A L_A' (Appendix B of the first paper; for one
%   tail component a scaling, which changes nothing).  The coefficients c_{r,nu} of G^r/r! follow
%   from the recursion (3), and the integrals I_A(nu) factor into the products of He_{nu_B}(0) and
%   an integral over the orthant of the tail components, which is one-dimensional for one tail
%   component, whatever the number of conditioning variables: three Gauss-Legendre panels of nG
%   nodes in the standardized variable v, split at 1/(1+u) and 4/(1+u), on [0, v_max], where v_max
%   is the point beyond which v^L exp(-u v - v^2/2), L = 3(2 order - 1) the highest degree of the
%   Hermite polynomials, has fallen by e^-40 from its maximum (the rule u v_max + v_max^2/2 = 40
%   leaves out the factor v^L, which costs up to ten digits in the integrals of high degree when u
%   is small), and the same with half the nodes for err.  For more tail components the orthant
%   integral is computed in sequentially conditioned coordinates, each later coordinate with three
%   panels placed by its own conditional truncation point (nG nodes each for two tail components,
%   nG/2 for more), with the Hermite polynomials products of univariate ones in the whitened
%   variable L_A^{-1}(z_A - t_A).  The density expansions are evaluated in whitened coordinates.
%   MATLAB counterpart of spa_cond of SaddlepointExpansions.jl (the same nodes and the same
%   formulas, so that the numbers agree to rounding).
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026.  Programs accompanying Broda and Kan,
%   "Conditional and lattice saddlepoint expansions of arbitrary order"; free to use and modify
%   with a citation of the paper.
if nargin < 5, opts = struct(); end
def = struct('lattice',false,'sinh','saddle','nG',64,'tau0',[],'Kwhite',[],'tail',true,'density',true,'tauf',[]);
fn = fieldnames(def);
for i = 1:numel(fn), if ~isfield(opts,fn{i}), opts.(fn{i}) = def.(fn{i}); end, end
if ~isa(Kfun,'function_handle'), error('spa_cond:Kfun','spa_cond: Kfun must be a function handle'); end
if ~(isscalar(order) && isnumeric(order) && order >= 1 && order == round(order)), error('spa_cond:order','spa_cond: order must be a positive integer'); end
if ~isnumeric(x) || isempty(x), error('spa_cond:x','spa_cond: x must be a nonempty vector'); end
if ~all(isfinite(x(:))), error('spa_cond:x','spa_cond: x must be finite'); end
if ~(ischar(opts.sinh) || isstring(opts.sinh)) || ~any(strcmp(opts.sinh,{'saddle','origin'})), error('spa_cond:sinh','spa_cond: sinh must be ''saddle'' or ''origin'''); end
if ~(isscalar(opts.nG) && opts.nG >= 8 && opts.nG == round(opts.nG)), error('spa_cond:nG','spa_cond: nG must be an integer of at least 8'); end
d = numel(x); x = reshape(double(x),1,d);
A = condset(A,d);
B = setdiff(1:d,A); b = numel(B);
t0 = []; if ~isempty(opts.tau0), t0 = reshape(double(opts.tau0),1,d); end
mode = char(opts.sinh); lattice = logical(opts.lattice);
% ---- the denominator: the density (or probability) expansion of X_B at its own saddlepoint ----
if b > 0
   KB = @(nu,s) Kfun(embed(nu,B,d),embed(s,B,d));
   KwB = [];
   if ~isempty(opts.Kwhite)
      EB = zeros(d,b); EB(sub2ind([d b],B,1:b)) = 1;
      KwB = @(nu,u,M) opts.Kwhite(nu,u,EB*M);
   end
   if isempty(t0), tb0 = zeros(1,b); else, tb0 = t0(B); end
   taut = spa_saddle(KB,x(B),tb0);
   den = dens(KB,KwB,taut,x(B),order);
   tautf = zeros(1,d); tautf(B) = taut;
else
   den = struct('logf0',0,'s',ones(order,1),'kappa',ones(order,1)); tautf = zeros(1,d);
end
status = 'ok';
if canc(den.kappa), status = addstatus(status,'large cancellation in the denominator'); end
% ---- the conditional density: the density expansion of X at its saddlepoint ----
f = NaN(order,1); kappaf = NaN(order,1); tauf = NaN(1,d);
if opts.density
   if ~isempty(opts.tauf)
      tauf = reshape(double(opts.tauf),1,d);
   else
      if isempty(t0), tf0 = zeros(1,d); else, tf0 = t0; end
      tauf = spa_saddle(Kfun,x,tf0);
   end
   num = dens(Kfun,opts.Kwhite,tauf,x,order);
   f = exp(num.logf0-den.logf0)*num.s./den.s;
   kappaf = num.kappa;
   if canc(kappaf), status = addstatus(status,'large cancellation in the density'); end
end
% ---- the conditional tail probability ----
P = NaN(order,1); kappaP = NaN(order,1); err = NaN(order,1); Pp = NaN(2*order,1); tau = NaN(1,d);
if opts.tail
   if opts.density, tt0 = tauf; else, tt0 = t0; end
   r = condtail(Kfun,opts.Kwhite,x,A,order,den,tt0,lattice,mode,opts.nG);
   Pp = r.P; tau = r.tau;
   P = Pp(2:2:end); err = r.err(2:2:end);
   kappaP = r.abssum(2:2:end)./abs(P);
   if canc(kappaP), status = addstatus(status,'large cancellation in the numerator'); end
   if all(isfinite(err)) && err(order) > 1e-8*abs(P(order)), status = addstatus(status,'integration not resolved'); end
end
info = struct('tau',tau,'tauf',tauf,'taut',tautf,'kappaP',kappaP,'kappaf',kappaf,'kappaB',den.kappa, ...
   'err',err,'Pp',Pp,'status',status);
end

% ======================================================================================
function r = condtail(Kfun,Kwhite,x,A,order,den,tau0,lattice,mode,nG)
%  P = D/D_M with the numerator terms m <= j, j = 0..2*order-1, and M = floor(j/2)+1 terms in the
%  denominator (D scaled by the leading density term of X_B, den.logf0), the sums of the absolute
%  values of the terms (divided by |D_M|) and the error estimates; when the saddlepoint has
%  negative components in A, by the inclusion-exclusion (Remark 1 of the paper): every term is the
%  conditional probability, given the same X_B, of the components of A that are kept, those in T
%  negated, with the components of N \ T marginalized
d = numel(x);
xc = x;
if lattice, xc(A) = xc(A)-1/2; end
if isempty(tau0), tau0 = zeros(1,d); end
try
   tau = spa_saddle(Kfun,xc,tau0);
catch e
   if ~lattice, rethrow(e); end
   % no saddlepoint at k - 1/2: the corrected point is not inside the convex hull of the support,
   % so that the probability is zero or one (P(S >= 0) for a count, for example)
   error('spa_cond:support',['spa_cond: no saddlepoint at the continuity-corrected point; it is outside ' ...
      'the interior of the convex hull of the support (the probability is 0 or 1 there)']);
end
Nn = A(tau(A) < 0);
if isempty(Nn)
   q = numer(Kfun,Kwhite,tau,xc,A,order,den.logf0,lattice,mode,nG);
   sj = den.s(floor((0:2*order-1)/2)+1); sj = sj(:);
   r = struct('P',q.D./sj,'abssum',q.Dabs./abs(sj),'err',q.err./abs(sj),'tau',tau);
   return
end
B = setdiff(1:d,A); Apos = setdiff(A,Nn);
P = zeros(2*order,1); abssum = zeros(2*order,1); err = zeros(2*order,1);
for mask = 0:2^numel(Nn)-1
   Tset = Nn(bitget(mask,1:numel(Nn)) == 1);
   sgn = (-1)^numel(Tset);
   tail = sort([Apos Tset]);
   if isempty(tail), P = P+sgn; abssum = abssum+1; continue, end
   keep = sort([tail B]); k = numel(keep);
   flip = ones(1,d); flip(Tset) = -1; fk = flip(keep);
   Km = @(nu,t) prod(fk.^nu)*Kfun(embed(nu,keep,d),flip.*embed(t,keep,d));
   Kw = [];
   if ~isempty(Kwhite)
      Bm = zeros(d,k); Bm(sub2ind([d k],keep,1:k)) = flip(keep);
      Kw = @(nu,u,M) Kwhite(nu,u,Bm*M);                      % K_B(M u) = K(B M u)
   end
   x2 = flip.*x;
   if lattice, x2(Tset) = x2(Tset)+1; end                    % X_j < x_j <=> -X_j >= -x_j + 1
   [~,A2] = ismember(tail,keep);
   ft = flip.*tau;
   q = condtail(Km,Kw,x2(keep),A2,order,den,ft(keep),lattice,mode,nG);
   P = P+sgn*q.P; abssum = abssum+q.abssum; err = err+q.err;
end
r = struct('P',P,'abssum',abssum,'err',err,'tau',tau);
end

function q = numer(Kfun,Kwhite,tau,x,A,order,logscale,lattice,mode,nG)
%  E_n [sum_{m <= j} sum_r sum_|nu|=m+2r ct_{r,nu} I_A(nu)], j = 0..2*order-1, at the saddlepoint tau
%  of the (continuity-corrected) point x, all tau_A >= 0, scaled by exp(-logscale); with the sums of
%  the absolute values of the terms and the change with half the quadrature nodes.  The
%  coordinates are the canonical ones of Section 5.2 of the paper, s_A = u'_A,
%  s_B = -C' u'_A + LB^{-T} u_B, followed by the whitening u'_A = LA^{-T} u_A of the tail
%  components, LA LA' = K''_{AA.B}
d = numel(x); B = setdiff(1:d,A); a = numel(A); b = numel(B);
p = 2*order-1; L = 3*p;
[K0,K2] = k02(Kfun,tau);
T1 = eye(d); uh = tau;
if b > 0
   Lb = chol(K2(B,B),'lower');
   C = K2(A,B)/K2(B,B);
   T1(B,A) = -C.'; T1(B,B) = inv(Lb.');
   SA = K2(A,A)-C*K2(B,A);
   uh(B) = (Lb.'*(tau(B).'+C.'*tau(A).')).';
   logdetL = sum(log(diag(Lb)));
else
   SA = K2; logdetL = 0;
end
SA = (SA+SA.')/2;
[LA,flag] = chol(SA,'lower');
if flag, error('spa_cond:domain','spa_cond: K''''_{AA.B} is not positive definite at the saddlepoint'); end
UA = inv(LA.');
T2 = eye(d); T2(A,A) = UA;
Tm = T1*T2;
uh(A) = (LA.'*tau(A).').';
IL = mindex(d,L); nN = size(IL.list,1);
% the coefficients of G^r/r! in the coordinates s = Tm u
subs = cell(1,d); ns = 0;
for j = fliplr(B), ns = ns+1; subs{ns} = {j, T1(j,:)}; end
for j = fliplr(A), ns = ns+1; subs{ns} = {j, T2(j,:)}; end
theta = ctheta(Kfun,Kwhite,tau,uh,Tm,IL,p+2,subs);
c = ccoeffs(theta,IL,p,@(r) min(L,p+2*r));
delta = zeros(nN,1); delta(1) = 1;
if ~lattice
   ctM = repmat({[{delta} c]},1,order);
else
   % the amplitude prod_{j in A} g(s_j): its Taylor coefficients at tau in s_A, in the whitened
   % variables u_A (s_A = tau_A + UA (u_A - uh_A)) by the same substitution
   ctM = cell(1,order);
   for M = 1:order
      if strcmp(mode,'saddle') && M > 1, ctM{M} = ctM{1}; continue, end
      if strcmp(mode,'origin'), dg = 2*M-2; else, dg = 0; end
      G = cell(1,a);
      for i = 1:a, G{i} = sinhcoef(tau(A(i)),L,mode,dg); end
      ctM{M} = latticect(c,G,IL,A,UA,p,L);
   end
end
% the integrals I_A(nu) = I_A(nu_A) He_{nu_B}(0): under the Gaussian of the integrand the tail and
% the conditioning components are independent in these coordinates
logE0 = K0-tau*x.'-logdetL-(b/2)*log(2*pi)-logscale;
IA = mindex(a,L);
[Iv,Ih] = condIA(IA,SA,LA,tau(A),logE0,nG);
hB = hcoef(floor(L/2)+1);
hb = ones(nN,1);
for j = B
   nj = IL.list(:,j); od = mod(nj,2) == 1;
   hb(od) = 0;
   ev = ~od & hb ~= 0; hb(ev) = hb(ev).*hB(nj(ev)/2+1);
end
nzb = find(hb ~= 0);
rA = mi_rank(IL.list(nzb,A),a,IA.B);
[D,Dabs] = sums(full_(Iv));
err = abs(sums(full_(Ih))-D);
q = struct('D',D,'Dabs',Dabs,'err',err);
   function out = full_(Iv)
      out = zeros(nN,1); out(nzb) = hb(nzb).*Iv(rA);
   end
   function [F,ab] = sums(Inu)
      % partial sums over m <= j, j = 0..p, each with the amplitude of the order M = floor(j/2)+1
      F = zeros(p+1,1); ab = zeros(p+1,1);
      for jj = 0:p
         ct = ctM{floor(jj/2)+1}; s = 0; sa = 0;
         for m = 0:jj
            for r = 0:m
               k = degrange(IL,m+2*r);
               v = ct{r+1}(k); w = Inu(k);
               nz = v ~= 0 & w ~= 0;
               tt = v(nz).*w(nz);
               s = s+sum(tt); sa = sa+sum(abs(tt));
            end
         end
         F(jj+1) = s; ab(jj+1) = sa;
      end
   end
end

function ct = latticect(c,G,IL,A,UA,p,L)
%  ct_r = sum_mu c_{r, . - mu} gamma_mu, r = 0..p, with c_0 = delta_0 and gamma the Taylor
%  coefficients of h(s) = prod_{j in A} g(s_j) at tau (G{i} those of g at tau_{A_i}), expressed in
%  the whitened variables of the tail components, s_A - tau_A = UA (u_A - uh_A) (multi-indices
%  supported on A)
a = numel(A); d = size(IL.list,2);
IA = mindex(a,L);
h = G{1}(IA.list(:,1)+1);
for i = 2:a, h = h.*G{i}(IA.list(:,i)+1); end
for i = a:-1:1, h = polysub(h,IA,i,UA(i,:)); end      % s_{A_i} -> sum_{k >= i} UA(i,k) u_{A_k}
kg = find(h ~= 0);
gmu = zeros(numel(kg),d); gmu(:,A) = IA.list(kg,:);
gval = h(kg); gdeg = IA.deg(kg);
nN = size(IL.list,1);
ct = cell(1,p+1);
for r = 0:p
   if r == 0, dm = p; else, dm = min(L,p+2*r); end
   v = zeros(nN,1);
   if r == 0
      sel = gdeg <= dm;
      tgt = mi_rank(gmu(sel,:),d,IL.B);
      v(tgt) = v(tgt)+gval(sel);
   else
      prev = c{r};
      kk = find(prev(1:IL.cnt(dm+1)) ~= 0);
      dk = IL.deg(kk); Ak = IL.list(kk,:);
      for q = 1:numel(kg)
         sel = dk+gdeg(q) <= dm;
         if ~any(sel), continue, end
         tgt = mi_rank(Ak(sel,:)+gmu(q,:),d,IL.B);
         v(tgt) = v(tgt)+gval(q)*prev(kk(sel));
      end
   end
   ct{r+1} = v;
end
end

% ----- the density expansion (Theorem 5 of the first paper) in whitened coordinates

function den = dens(Kfun,Kwhite,tau,x,order)
%  log of the leading term and the bracket 1 + sum_m n^-m ... of orders 1..order at the
%  saddlepoint tau of x, with the cancellation indicators of the bracket
d = numel(x);
[K0,K2] = k02(Kfun,tau);
Lc = chol(K2,'lower');
logf0 = K0-tau*x.'-(d/2)*log(2*pi)-sum(log(diag(Lc)));
s = ones(order,1); kappa = ones(order,1);
den = struct('logf0',logf0,'s',s,'kappa',kappa);
if order == 1, return, end
Tm = inv(Lc.');
uh = (Lc.'*tau.').';
Lx = 6*(order-1);
I = mindex(d,Lx);
subs = cell(1,d);
for j = d:-1:1, subs{d-j+1} = {j, Tm(j,:)}; end
theta = ctheta(Kfun,Kwhite,tau,uh,Tm,I,2*order,subs);
R = 2*(order-1);
c = ccoeffs(theta,I,R,@(r) min(Lx,2*(order-1)+2*r));
He0 = he0list(I);
acc = 1; ab = 1;
for M = 2:order
   m = M-1;
   for r = 1:2*m
      k = degrange(I,2*m+2*r);
      v = c{r}(k); h = He0(k);
      nz = v ~= 0 & h ~= 0;
      t = v(nz).*h(nz);
      acc = acc+sum(t); ab = ab+sum(abs(t));
   end
   s(M) = acc; kappa(M) = ab/abs(acc);
end
den.s = s; den.kappa = kappa;
end

function [K0,K2] = k02(Kfun,tau)
%  value and Hessian of K at tau from Kfun
d = numel(tau); E = eye(d);
K0 = Kfun(zeros(1,d),tau);
K2 = zeros(d);
for i = 1:d, for j = 1:d, K2(i,j) = Kfun(E(i,:)+E(j,:),tau); end, end
if ~(isfinite(K0) && all(isfinite(K2(:)))), error('spa_cond:domain','spa_cond: the saddlepoint is outside the domain of K'); end
K2 = (K2+K2.')/2;
end

function theta = ctheta(Kfun,Kwhite,tau,uh,Tm,I,nmax,subs)
%  Taylor coefficients theta(k) = [v^nu_k] K(tau + Tm v), 3 <= |nu_k| <= nmax, on the list of I:
%  from Kwhite (the derivatives of K_Y(u) = K(Tm u) at uh, Tm uh = tau), or from the derivatives of
%  K at tau by the substitutions subs = {{j, ell}, ...}, applied in turn: the variable in slot j is
%  replaced by the linear form ell*u, which may involve only the variables of slots already
%  substituted, or of slots that keep their variable
theta = zeros(size(I.list,1),1);
for k = 1:I.cnt(nmax+1)
   if I.deg(k) < 3, continue, end
   nu = I.list(k,:);
   fa = prod(factorial(nu));
   if isempty(Kwhite), theta(k) = Kfun(nu,tau)/fa; else, theta(k) = Kwhite(nu,uh,Tm)/fa; end
end
if ~isempty(Kwhite), return, end
for i = 1:numel(subs), theta = polysub(theta,I,subs{i}{1},subs{i}{2}); end
end

function new = polysub(th,I,j,ell)
%  substitution of the variable in slot j by the linear form ell*u in the polynomial with the
%  coefficients th on the list of I (all degrees are kept: the substitution is homogeneous)
d = size(I.list,2);
supp = find(ell ~= 0); ns = numel(supp);
new = zeros(size(th));
m = I.list(:,j);
q0 = th ~= 0 & m == 0; new(q0) = th(q0);
if ns == 0, return, end
nzm = m(th ~= 0);
for mm = 1:max([0; nzm])
   idx = find(th ~= 0 & m == mm);
   if isempty(idx), continue, end
   Cm = comps(mm,ns);
   cf = factorial(mm)*ones(size(Cm,1),1);
   for i = 1:ns, cf = cf.*(ell(supp(i)).^Cm(:,i)./factorial(Cm(:,i))); end
   base = I.list(idx,:); base(:,j) = 0;
   for q = 1:size(Cm,1)
      mu = base; mu(:,supp) = mu(:,supp)+Cm(q,:);
      tgt = mi_rank(mu,d,I.B);
      new(tgt) = new(tgt)+th(idx)*cf(q);
   end
end
end

function c = ccoeffs(theta,I,R,dmax)
%  c{r}(k) = [u^nu_k] G^r/r!, r = 1..R, from the coefficients theta of G (degrees >= 3), computed
%  only for 3r <= |nu| <= dmax(r): the terms of the expansions use no others
c = cell(1,max(R,0));
if R < 1, return, end
c{1} = theta;
d = size(I.list,2);
supp = find(theta ~= 0); dsupp = I.deg(supp);       % in increasing degree (graded list)
for r = 2:R
   dm = dmax(r);
   Ar = zeros(size(theta)); prev = c{r-1};
   kk = find(prev(1:I.cnt(dm-3+1)) ~= 0);
   dk = I.deg(kk); Ak = I.list(kk,:);
   if isempty(kk), c{r} = Ar; continue, end
   mind = min(dk);
   for i = 1:numel(supp)
      if mind+dsupp(i) > dm, break, end
      sel = dk+dsupp(i) <= dm;
      tgt = mi_rank(Ak(sel,:)+I.list(supp(i),:),d,I.B);
      Ar(tgt) = Ar(tgt)+theta(supp(i))*prev(kk(sel));
   end
   c{r} = Ar/r;
end
end

function He0 = he0list(I)
%  He_nu(0) = prod_j He_{nu_j}(0) on the list of I (He_n(0) = (-1)^(n/2) (n-1)!! for even n)
h = hcoef(floor(I.N/2)+1);
He0 = ones(size(I.list,1),1);
for j = 1:size(I.list,2)
   nj = I.list(:,j); od = mod(nj,2) == 1;
   He0(od) = 0;
   ev = ~od; He0(ev) = He0(ev).*h(nj(ev)/2+1);
end
end

function h = hcoef(n)
%  h(i+1) = (-1)^i (2i-1)!! = He_{2i}(0), i = 0..n
h = ones(n+1,1);
for i = 1:n, h(i+1) = -h(i)*(2*i-1); end
end

% ----- the lattice amplitude g(s) = s/(2 sinh(s/2))

function co = ginvseries(n)
%  coefficients of 1/Q(u), Q(u) = 2 sinh(u/2)/u = sum_k u^(2k)/(4^k (2k+1)!), degrees 0..n
Q = zeros(n+1,1);
for k = 0:floor(n/2), Q(2*k+1) = 1/(4^k*factorial(2*k+1)); end
co = zeros(n+1,1); co(1) = 1;
for m = 1:n, co(m+1) = -sum(Q(2:m+1).*co(m:-1:1)); end
end

function g = shiftpoly(co,s0,L)
%  Taylor coefficients at s0, degrees 0..L, of the polynomial sum_k co(k+1) s^k
n = numel(co)-1;
Pt = zeros(n+1); Pt(:,1) = 1;                     % Pascal triangle, Pt(k+1,j+1) = binomial(k,j)
for k = 1:n, for j = 1:k, Pt(k+1,j+1) = Pt(k,j)+Pt(k,j+1); end, end
g = zeros(L+1,1);
for k = 0:n
   for j = 0:min(k,L)
      g(j+1) = g(j+1)+co(k+1)*Pt(k+1,j+1)*s0^(k-j);
   end
end
end

function g = sinhcoef(s0,L,mode,deg)
%  Taylor coefficients at s0, degrees 0..L, of g(s) = s/(2 sinh(s/2)) (mode 'saddle'), or of its
%  expansion about the origin truncated after the term s^deg (mode 'origin', Kolassa 2003)
if strcmp(mode,'origin'), g = shiftpoly(ginvseries(deg),s0,L); return, end
if abs(s0) >= 1
   % g = s/q with q = 2 sinh(s/2), by series division
   q = zeros(L+1,1);
   for k = 0:L
      if mod(k,2) == 0, q(k+1) = 2*sinh(s0/2)/(2^k*factorial(k)); else, q(k+1) = 2*cosh(s0/2)/(2^k*factorial(k)); end
   end
   pp = zeros(L+1,1); pp(1) = s0; if L >= 1, pp(2) = 1; end
   g = zeros(L+1,1); g(1) = pp(1)/q(1);
   for m = 1:L, g(m+1) = (pp(m+1)-sum(q(2:m+1).*g(m:-1:1)))/q(1); end
   return
end
% the series at the origin (radius 2 pi), shifted to s0
n = L+max(40,ceil(-log(eps)/log(2*pi))+10);
g = shiftpoly(ginvseries(n),s0,L);
end

% ----- the integrals over the orthant of the tail components

function [Iv,Ih] = condIA(IA,S,LA,tA,logE0,G)
%  the integrals over the orthant z_A >= t_A of the tail components (t_A = S tA, S = K''_{AA.B},
%  z_A = t_A + w) in the canonical coordinates, the conditioning components fixed at t_B, in the form
%  of Appendix B of the first paper: IA(k) = exp(logE0) int_{w >= 0} He_{mu_k}(v)
%  exp(-tA'w - w'S^{-1}w/2) dw/((2 pi)^(a/2) |S|^(1/2)), mu_k = IA.list(k,:), with
%  He_mu(v) = prod_i He_{mu_i}(v_i) in the whitened variable v = LA^{-1} w, LA LA' = S; G and G/2
%  nodes per panel (the second for the error estimate)
a = numel(tA); L = IA.N; c = 40;
if a == 1
   u = LA(1,1)*tA(1);
   Iv = run1(G); Ih = run1(max(8,floor(G/2)));
   return
end
Si = inv(S); Si = (Si+Si.')/2;
ldS = 2*sum(log(diag(LA)));
Iv = runa(G); Ih = runa(max(8,floor(G/2)));
   function out = run1(G)
      [vv,ww] = panels(u,G,c,L);
      W = ww.*(exp(logE0-u*vv-vv.^2/2)/sqrt(2*pi));
      out = zeros(L+1,1);
      h0 = ones(numel(vv),1); h1 = vv;
      out(1) = sum(W);
      if L >= 1, out(2) = W.'*h1; end
      for k = 1:L-1
         h2 = vv.*h1-k*h0; h0 = h1; h1 = h2;
         out(k+2) = W.'*h1;
      end
   end
   function out = runa(G)
      [W,lq] = cgrid(S,tA,G,c,L);
      n = size(W,2);
      lw = lq+logE0-tA(:).'*W-sum(W.*(Si*W),1)/2-(a/2)*log(2*pi)-ldS/2;
      wt = exp(lw).';
      V = LA\W;
      out = zeros(size(IA.list,1),1);
      chunk = max(64,floor(4e6/size(IA.list,1)));
      for c0 = 1:chunk:n
         c1 = min(n,c0+chunk-1);
         out = out+hermprod(IA,V(:,c0:c1)).'*wt(c0:c1);
      end
   end
end

function He = hermprod(I,V)
%  He(j,k) = prod_i He_{nu_i}(V(i,j)), nu = I.list(k,:): the Hermite polynomials of the whitened
%  variables (identity covariance)
[a,n] = size(V); L = I.N;
H = cell(1,a);
for i = 1:a
   Hi = zeros(n,L+1); Hi(:,1) = 1; v = V(i,:).';
   if L >= 1, Hi(:,2) = v; end
   for k = 1:L-1, Hi(:,k+2) = v.*Hi(:,k+1)-k*Hi(:,k); end
   H{i} = Hi;
end
He = H{1}(:,I.list(:,1)+1);
for i = 2:a, He = He.*H{i}(:,I.list(:,i)+1); end
end

function [v,w] = panels(u,G,c,L)
%  nodes v >= 0 and weights of three Gauss-Legendre panels for the integrands
%  He_k(v) exp(-(u + v)^2/2), k <= L, on [0, inf), truncated where L log(max(v, 1)) - (u + v)^2/2
%  falls c below its maximum (for L = 0 and u >= 0 this is the truncation u vmax + vmax^2/2 = c of
%  the paper; the factor v^L, which the paper's rule leaves out, moves the end point out for the
%  integrals of high degree); for u >= 0 the panels are split at 1/(1 + u) and 4/(1 + u), where
%  exp(-u v) decays, and otherwise around the mode v = -u
[xg,wg] = gauleg(G);
h = @(v) L*log(max(v,1))-(u+v)^2/2;
vq = max(0,-u);                                      % maximizer of the Gaussian factor
if L > 0, vs = (-u+sqrt(u^2+4*L))/2; else, vs = 0; end   % stationary point where v >= 1
if vs > 1, vst = max(vq,vs); hm = max(h(vq),h(vs)); else, vst = vq; hm = h(vq); end
lo = vst; hi = vst+1;
while h(hi) > hm-c, lo = hi; hi = 2*hi; end
for it = 1:200                                       % bisection for the end point
   mid = (lo+hi)/2;
   if h(mid) > hm-c, lo = mid; else, hi = mid; end
   if hi-lo <= 8*eps*hi, break, end
end
vmax = hi;
if u >= 0
   br = [0 min(1/(1+u),vmax/3) min(4/(1+u),2*vmax/3) vmax];
else
   v0 = max(0,-u-sqrt(2*c));
   b1 = min(max(-u-2,v0),vmax); b2 = min(max(-u+2,b1),vmax);
   br = [v0 b1 b2 vmax];
end
v = zeros(0,1); w = zeros(0,1);
for k = 1:3
   hw = (br(k+1)-br(k))/2;
   if ~(hw > 0), continue, end
   m = (br(k)+br(k+1))/2;
   v = [v; m+hw*xg]; w = [w; hw*wg]; %#ok<AGROW>
end
end

function [W,lq] = cgrid(S,tA,G,c,L)
%  nodes w = z_A - t_A >= 0 of the a-dimensional orthant in sequentially conditioned coordinates of
%  N(0, S), t_A = S tA, with the logarithms of the products of the panel weights (dz).  The first
%  coordinate has the panels of the case a = 1 with G nodes each; each later one, given the
%  earlier, three panels placed by its own standardized truncation point (t_i - mu_i)/s_i, with G
%  nodes each for a = 2 and max(8, G/2) for more tail components
a = numel(tA); t = S*tA(:);
s1 = sqrt(S(1,1));
[v,w] = panels(t(1)/s1,G,c,L);
Z = (t(1)+s1*v).'; lq = log(s1*w).';
for i = 2:a
   Sp = S(1:i-1,1:i-1);
   beta = S(i,1:i-1)/Sp;
   si = sqrt(S(i,i)-beta*S(1:i-1,i));
   if a == 2, Gi = G; else, Gi = max(8,floor(G/2)); end
   nq = size(Z,2); Zc = cell(1,nq); lc = cell(1,nq);
   for qq = 1:nq
      mu = beta*Z(:,qq);
      [vi,wi] = panels((t(i)-mu)/si,Gi,c,L);
      Zc{qq} = [repmat(Z(:,qq),1,numel(vi)); (t(i)+si*vi).'];
      lc{qq} = lq(qq)+log(si*wi).';
   end
   Z = [Zc{:}]; lq = [lc{:}];
end
W = Z-t;
end

function [x,w] = gauleg(n)
%  Gauss-Legendre nodes and weights on [-1,1] (Golub-Welsch), cached
persistent cache
if isempty(cache), cache = containers.Map('KeyType','double','ValueType','any'); end
if isKey(cache,n), xw = cache(n); x = xw{1}; w = xw{2}; return, end
k = 1:n-1; bb = k./sqrt(4*k.^2-1);
[V,D] = eig(diag(bb,1)+diag(bb,-1));
[x,i] = sort(diag(D)); w = 2*V(1,i)'.^2;
cache(n) = {x, w};
end

% ----- multi-indices (graded order), as in spa_mv

function I = mindex(d,N)
%  the multi-indices of degree <= N in d variables in graded order: I.list, I.deg, I.cnt(k+1) (the
%  number of degree <= k), I.B (binomial table for the ranks); cached per (d, N)
persistent cache
if isempty(cache), cache = containers.Map(); end
key = sprintf('%d_%d',d,N);
if isKey(cache,key), I = cache(key); return, end
nb = N+d+1; Bn = zeros(nb+1); Bn(:,1) = 1;
for n = 1:nb, for k = 1:n, Bn(n+1,k+1) = Bn(n,k)+Bn(n,k+1); end, end
AL = cell(N+1,1);
for k = 0:N, AL{k+1} = comps(k,d); end
AL = vertcat(AL{:});
cnt = Bn((0:N)+d+1,d+1);
I = struct('d',d,'N',N,'list',AL,'deg',sum(AL,2),'cnt',cnt(:),'B',Bn);
cache(key) = I;
end

function k = degrange(I,m)
%  indices of the multi-indices of degree m in the graded list of I
if m == 0, k = 1; else, k = (I.cnt(m)+1:I.cnt(m+1)).'; end
end

function C = comps(m,p)
if p == 1, C = m; return, end
C = cell(m+1,1);
for k = m:-1:0
   R = comps(m-k,p-1);
   C{m-k+1} = [k*ones(size(R,1),1) R];
end
C = vertcat(C{:});
end

function r = mi_rank(M,d,Bn)
%  positions (1-based) of the rows of M in the graded list
k = sum(M,2);
r = zeros(size(M,1),1);
pos = k > 0; r(pos) = bin(Bn,k(pos)-1+d,d);
ki = k;
for i = 1:d-1
   r = r+bin(Bn,ki-M(:,i)-1+d-i,d-i);
   ki = ki-M(:,i);
end
r = r+1;
end

function v = bin(Bn,n,k)
v = zeros(size(n));
ok = n >= 0 & k >= 0 & n >= k;
nn = n(ok); v(ok) = Bn(sub2ind(size(Bn),nn(:)+1,(k+1)*ones(numel(nn),1)));
end

% ----- small helpers

function A = condset(A,d)
%  A as a sorted row of distinct indices in 1:d
A = reshape(A,1,[]);
if isempty(A) || ~isnumeric(A) || any(A ~= round(A)) || any(A < 1 | A > d) || numel(unique(A)) ~= numel(A)
   error('spa_cond:A','spa_cond: A must be a nonempty set of distinct indices in 1:%d',d);
end
A = sort(double(A));
end

function y = embed(v,idx,d)
y = zeros(1,d); y(idx) = v;
end

function tf = canc(v)
tf = ~all(isfinite(v)) || max(v)*eps > 1e-8;
end

function s = addstatus(s,msg)
if strcmp(s,'ok'), s = msg; else, s = [s '; ' msg]; end
end
