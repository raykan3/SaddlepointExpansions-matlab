function [pdf,Fup,tau,kappa,err,info] = spa_mv(Kfun,x,order,opts)
%SPA_MV Saddlepoint approximations of arbitrary order in any dimension d (Broda and Kan,
%   "Saddlepoint expansions of arbitrary order", Theorems 1 and 2), with n = 1.
%   [pdf,Fup,tau,kappa,err,info] = spa_mv(Kfun,x,order,opts)
%   Kfun(nu,t): derivative of order nu (1 x d nonnegative integers) of the cgf K at t (1 x d);
%               nu = zeros(1,d) gives K(t); NaN outside the domain of K.  For the variable-
%               precision computations below Kfun must also accept vpa arguments.
%   x:          the point (1 x d).
%   pdf(M):     density expansion with the terms m <= M-1 (relative error O(n^-M)); a finite
%               sum evaluated without integrals (not the exact density).
%   Fup(M):     P[X >= x] with the terms m <= 2M-1, when the saddlepoint tau has no negative
%               component (NaN otherwise; use spa_orthant).
%   kappa(M):   cancellation indicator of the final sum Fup(M), (sum of |terms|)/|sum|; it does
%               not measure errors in the Taylor coefficients (see info.coef_err_cdf).
%   err(M):     error estimate of Fup(M) from the integrals I(nu) (see opts.method), for every
%               order M: the change with half the quadrature nodes ('quad'), the standard error
%               over random shifts ('qmc'), or the change when the base normal orthant
%               probabilities of the recursion are computed with half the nodes
%               ('vpa-recursion').  These compare two quadratures, are not bounds, and concern the
%               evaluation of the retained terms only, not the asymptotic error.
%   info:       struct with kappa_pdf (cancellation indicator of pdf); method_cdf, the method
%               actually used for Fup; refined_pdf, refined_cdf, digits and precision ('double'
%               or 'variable (n digits)', the working precision of the returned values, also
%               when x is given in variable precision); tau_refined (the saddlepoint used by a
%               refinement); saddle (iterations, Newton decrement and residual of the
%               saddlepoint, see spa_saddle); nodes (the quadrature or QMC settings used);
%               coef_err_pdf(M), coef_err_cdf(M): estimates, in double precision, of the
%               relative errors of pdf(M) and Fup(M) caused by rounding in the formation of the
%               whitened Taylor coefficients (first-order propagation of random perturbations of
%               the size of the rounding errors of the substitution; with opts.Kwhite only the
%               rounding of the supplied derivatives is included; NaN in variable precision);
%               and status, 'ok' or one or more of (joined by '; '): 'negative saddlepoint
%               component (use spa_orthant)', 'leading integral not resolved' (Fup and err are
%               then NaN), 'integration not resolved' (err(order) > qtol |Fup(order)| after the
%               node escalation), 'coefficients inaccurate' (a coefficient-error estimate above
%               1e-6 that no refinement removed), 'refinement failed', 'no orthant probability
%               requested', 'probability below the double-precision range' (the Chernoff bound
%               exp(K(tau) - tau'x) is below realmin; Fup is then returned as zero).
%               info.refined is true if either refinement was made.
%   opts (optional struct):
%     tau0      starting value of the saddlepoint (default zeros)
%     method    'auto' (default), 'quad' (d = 3, default there), 'qmc' (default for d >= 4) or
%               'vpa-recursion' (d = 3, variable precision; the older name 'exact' is accepted)
%               for the integrals I(nu):
%               'quad':  for u_j = tau_j/sd(W_j) > 2.5 for all j, W ~ N(0,Sigma^{-1}), the
%                        representation of Appendix B by nGH^3 Gauss-Hermite nodes; otherwise
%                        Gauss-Legendre quadrature of the orthant-integral form with nG nodes per
%                        coordinate in conditioned coordinates, the ranges split at the boundary
%                        layers that the truncation creates for correlations near +-1 and at
%                        far-tail decays (nG nodes on each piece); err is the change with half
%                        the nodes,
%                        and while err(order) > qtol |Fup(order)| the number of nodes is doubled,
%                        up to nGmax;
%               'qmc':   randomized quasi-Monte Carlo (Richtmyer lattice with antithetic points,
%                        nshift random shifts of nqmc points) after Genz's sequential conditioning
%                        of the normal vector truncated to the orthant; err is the standard error
%                        over the shifts;
%               'vpa-recursion': binomial sum (14) and recursion of Lemma 1 in variable
%                        precision (current digits; x and the derivatives must be vpa); the
%                        bivariate and trivariate normal orthant probabilities that start the
%                        recursion are computed by Gauss-Legendre quadrature (4 digits nodes,
%                        ranges split as for 'quad'), so more digits do not by
%                        themselves make them more accurate; err compares with half the nodes.
%     nG (64), nGH (48), nqmc (2^14), nshift (8), seed (1); qtol (1e-8, 1e-3 for 'qmc'): the
%               relative integration tolerance for the status; nGmax (2 nG) for 'quad'.
%     Kwhite    optional function handle @(nu,u,T): derivative of order nu at u of the cgf
%               K_Y(u) = K(T u) of Y = T' X for a d x d matrix T (with T = L^{-T}, Y = L^{-1} X).
%               The expansions are evaluated in whitened coordinates (see below); when Kwhite is
%               absent the Taylor coefficients of K_Y are obtained from those of K by polynomial
%               substitution, which loses digits when K'' is ill conditioned
%               (info.coef_err_pdf and info.coef_err_cdf estimate the loss).
%     refine    (default true) in more than two dimensions the terms of both expansions can cancel
%               by many orders of magnitude at high orders, and the substitution can lose
%               digits; when a double-precision cancellation indicator exceeds 1e6, or a
%               coefficient-error estimate exceeds 1e-10, the computation is repeated with about
%               log10(max(kappa, coef_err/eps))+26 digits (Symbolic Math Toolbox), with the
%               derivatives of K, and hence the coefficients, computed in that precision: the
%               density always, and for d = 3 with the quadrature method also the orthant
%               probability ('vpa-recursion').  If Kfun fails with vpa arguments, or the Symbolic
%               Math Toolbox is not available, the double-precision values are kept, with a
%               warning and info.status = 'refinement failed'.
%   The density prefactor and, in the double-precision methods, the exponential factor E_n of
%   (12) are formed in the log domain (from the Cholesky factor of K'', and combined with the
%   normal density or the probabilities of the sequential conditioning), so that the products
%   are formed without overflow or underflow of their factors.
%   Whitened evaluation.  With K'' = L L' (Cholesky) and T = L^{-T}, the terms of each order are
%   evaluated for Y = L^{-1} X: the Taylor coefficients of K_Y at L' tau replace those of K at tau,
%   the Hermite polynomials He_nu(w,K'') are replaced by products of univariate ones in L^{-1} w,
%   and the integrals I(nu) by integrals of these products over the same orthant, at the same
%   nodes (methods 'quad' and 'qmc'; in the frequency-domain representation (iW)^nu becomes
%   (iV)^mu with W = T V).  The sums of the terms of a given order are unchanged in exact
%   arithmetic, and the terms cancel much less when the components are strongly correlated under
%   the tilted distribution, provided the whitened coefficients are formed accurately (see
%   coef_err).  The method 'vpa-recursion' (Lemma 1) keeps the original coordinates, since the
%   recursion is tied to the orthant.
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Saddlepoint expansions of arbitrary order for multivariate distribution functions"; free to use and modify with a citation of the paper.
if nargin < 4, opts = struct(); end
def = struct('tau0',[],'method','auto','nG',64,'nGH',48,'nqmc',2^14,'nshift',8,'seed',1, ...
   'refine',true,'Kwhite',[],'qtol',[],'nGmax',[]);
fn = fieldnames(def);
for i = 1:numel(fn), if ~isfield(opts,fn{i}), opts.(fn{i}) = def.(fn{i}); end, end
if ~isa(Kfun,'function_handle'), error('spa_mv: Kfun must be a function handle'); end
if ~(isnumeric(x) || isa(x,'sym')) || isempty(x), error('spa_mv: x must be a numeric vector'); end
if isnumeric(x) && ~all(isfinite(x(:))), error('spa_mv: x must be finite'); end
if ~isscalar(order) || double(order) < 1 || order ~= round(order), error('spa_mv: order must be a positive integer'); end
if strcmp(opts.method,'exact'), opts.method = 'vpa-recursion'; end
if ~any(strcmp(opts.method,{'auto','quad','qmc','vpa-recursion','none'}))
   error('spa_mv: unknown method %s',opts.method);
end
d = numel(x); x = reshape(x,1,d);
[tau,sinfo] = spa_saddle(Kfun,x,opts.tau0);
[pdf,Fup,kappa,err,kpdf,meth,qinfo] = spa_mv_at(Kfun,tau,x,order,opts);
if isa(x,'sym'), prec = sprintf('variable (%d digits)',digits); dg0 = digits; else, prec = 'double'; dg0 = NaN; end
info = struct('kappa_pdf',kpdf,'method_cdf',meth,'refined_pdf',false,'refined_cdf',false, ...
   'refined',false,'digits',dg0,'precision',prec,'tau_refined',[],'saddle',sinfo,'nodes',qinfo, ...
   'coef_err_pdf',qinfo.coef_err_pdf,'coef_err_cdf',qinfo.coef_err_cdf,'status',qinfo.status);
cflag = @(v) any(v > 1e-6);                              % coefficient-error flag
if ~opts.refine || isa(x,'sym')
   info.status = coefstatus(info.status,cflag(info.coef_err_pdf),cflag(info.coef_err_cdf));
   return
end
%  refinement: cancellation in the sums (kappa) or loss of accuracy in the formation of the
%  whitened coefficients (coef_err) beyond about 1e-10 relative
cp = info.coef_err_pdf; cc = info.coef_err_cdf; cp(isnan(cp)) = 0; cc(isnan(cc)) = 0;
needpdf = ~all(kpdf <= 1e6) || any(cp > 1e-10);         % NaN or Inf counts as needing refinement
needcdf = d == 3 && any(strcmp(opts.method,{'auto','quad'})) && all(tau >= 0) && ...
   ~strcmp(qinfo.status,'leading integral not resolved') && (~all(kappa <= 1e6) || any(cc > 1e-10));
if ~(needpdf || needcdf)
   info.status = coefstatus(info.status,cflag(info.coef_err_pdf),cflag(info.coef_err_cdf));
   return
end
kap = max([kpdf; cp/eps]); if needcdf, kap = max([kap; kappa; cc/eps]); end
if ~(kap < 1e30), kap = 1e30; end                        % cap the precision for infinite indicators
try
   dig = ceil(log10(kap)+26);
   olddig = digits(dig); cleanup = onCleanup(@() digits(olddig));
   xv = vpa(x);
   tv = spa_saddle(Kfun,xv,vpa(tau));
   o2 = opts; if needcdf, o2.method = 'vpa-recursion'; else, o2.method = 'none'; end
   [pv,Fv,kv,ev,kpv,~,q2] = spa_mv_at(Kfun,tv,xv,order,o2);
   pdf = double(pv); info.kappa_pdf = double(kpv); info.refined_pdf = true;
   if needcdf
      Fup = double(Fv); kappa = double(kv); err = double(ev);
      info.refined_cdf = true; info.method_cdf = 'vpa-recursion'; info.status = q2.status;
   end
   info.refined = true; info.digits = dig; info.precision = sprintf('variable (%d digits)',dig);
   info.tau_refined = double(tv);
catch e
   info.status = 'refinement failed';
   warning('spa_mv:refine','refinement in variable precision failed (%s); double-precision values returned',e.message);
end
info.status = coefstatus(info.status,cflag(info.coef_err_pdf) && ~info.refined_pdf, ...
   cflag(info.coef_err_cdf) && ~info.refined_cdf && ~any(isnan(Fup)));
end

function s = coefstatus(s,fp,fc)
%  appends the coefficient-error flag to the status
if ~(fp || fc), return, end
if strcmp(s,'ok'), s = 'coefficients inaccurate'; else, s = [s '; coefficients inaccurate']; end
end

function [pdf,Fup,kappa,err,kpdf,meth,qinfo] = spa_mv_at(Kfun,tau,x,order,opts)
%  the expansions at the saddlepoint tau (double or vpa)
d = numel(x);
p = 2*order-1;
Nn = max(3*p,6*(order-1));
[A,B] = mi_list(d,Nn); nN = size(A,1); deg = sum(A,2);
rk = @(M) mi_rank(M,d,B);
n2 = mi_count(B,d,p+2);
Kd = zeros(n2,1,'like',x);
for k = 1:n2, Kd(k) = Kfun(A(k,:),tau); end
E = eye(d);
K0 = Kd(1);
K2 = zeros(d,d,'like',x);
for i = 1:d, for j = 1:d, K2(i,j) = Kd(rk(E(i,:)+E(j,:))); end, end
%  theta(k) = K_nu/nu!, 3 <= |nu| <= p+2, of K at tau (theta0) and of K_Y(u) = K(T u) at
%  uh = L' tau (theta), K2 = L L', T = L^{-T};  a{r}(k) = [u^nu] G^r/r! for K_Y
meth = opts.method;
if strcmp(meth,'auto')
   if isa(x,'sym'), meth = 'vpa-recursion'; elseif d == 3, meth = 'quad'; else, meth = 'qmc'; end
end
Lc = chol(K2,'lower'); T = inv(Lc.'); uh = (Lc.'*tau.').';
logdet2 = sum(log(diag(Lc)));                           % log det(K'')^(1/2)
sel = find(deg >= 3 & deg <= p+2);
if isa(x,'sym'), fsel = prod(factorial(sym(A(sel,:))),2); else, fsel = prod(factorial(A(sel,:)),2); end
theta0 = zeros(nN,1,'like',x); theta0(sel) = Kd(sel)./fsel;
isd = ~isa(x,'sym');                                    % coefficient-error estimate in double only
if strcmp(meth,'vpa-recursion')
   theta = theta0;                                      % Lemma 1 works in the original coordinates
   if isd, dth = 2*eps*abs(theta0); end
elseif ~isempty(opts.Kwhite)
   theta = zeros(nN,1,'like',x);
   for k = sel.', theta(k) = opts.Kwhite(A(k,:),uh,T); end
   theta(sel) = theta(sel)./fsel;
   if isd, dth = 2*eps*abs(theta); end
else
   theta = polysub_mv(theta0,T,A,deg,rk,d);
   if isd, dth = (deg+2)*eps.*polysub_mv(abs(theta0),abs(T),A,deg,rk,d); end
end
supp = sel(nonzero(theta(sel)));
R = max(p,2*(order-1));
a = cell(max(R,1),1); a{1} = theta;
for r = 2:R
   prev = a{r-1}; Ar = zeros(nN,1,'like',x);
   kidx = find(nonzero(prev) & deg >= 3*(r-1)); Ak = A(kidx,:); dk = deg(kidx);
   for q = supp.'
      s2 = dk+deg(q) <= Nn;
      tgt = rk(Ak(s2,:)+A(q,:));
      Ar(tgt) = Ar(tgt)+theta(q)*prev(kidx(s2));
   end
   a{r} = Ar/r;
end
%  first-order propagation of random perturbations of the whitened coefficients of the size of
%  their rounding errors (double precision only): da{r} = a{r-1} dtheta, the derivative of G^r/r!
ndraw = 2;
if isd
   st = RandStream('mt19937ar','Seed',1); da = cell(ndraw,1);
   for qd = 1:ndraw
      dt = dth.*sign(rand(st,nN,1)-0.5); da{qd} = cell(max(R,1),1); da{qd}{1} = dt;
      sdt = find(dt ~= 0 & deg <= Nn).';
      for r = 2:R
         prev = a{r-1}; Ar = zeros(nN,1);
         kidx = find(prev ~= 0 & deg >= 3*(r-1)); Ak = A(kidx,:); dk = deg(kidx);
         for q = sdt
            s2 = dk+deg(q) <= Nn;
            tgt = rk(Ak(s2,:)+A(q,:));
            Ar(tgt) = Ar(tgt)+dt(q)*prev(kidx(s2));
         end
         da{qd}{r} = Ar;
      end
   end
end
rec = mi_rec(A,d,B);
%  density: Hermite polynomials at 0, in the whitened coordinates (products of univariate values)
Dk = diag(1./sqrt(diag(K2))); Si = Dk*inv(Dk*K2*Dk)*Dk;   % inverse insensitive to the units
if strcmp(meth,'vpa-recursion'), SiH = Si; else, SiH = sym_or_double(eye(d),x); end
He0 = hermite_mv(SiH,zeros(d,1,'like',x),rec).';
f0 = exp(K0-tau*x.'-(d/2)*log(2*sym_or_double(pi,x))-logdet2);
pdf = zeros(order,1,'like',x); kpdf = ones(order,1); s = 1; sa = 1; pdf(1) = f0;
cpdf = nan(order,1); if isd, cpdf(1) = 0; dsp = zeros(ndraw,1); end
for M = 2:order
   m = M-1;
   for r = 1:2*m
      idx = find(deg == 2*m+2*r);
      terms = a{r}(idx).*He0(idx);
      s = s+sum(terms); sa = sa+sum(abs(terms));
      if isd, for qd = 1:ndraw, dsp(qd) = dsp(qd)+sum(da{qd}{r}(idx).*He0(idx)); end, end
   end
   pdf(M) = f0*s; kpdf(M) = double(sa/abs(s));
   if isd, cpdf(M) = max(abs(dsp))/abs(s); end
end
%  orthant probability
Fup = nan(order,1); kappa = nan(order,1); err = nan(order,1);
qinfo = struct('method',meth,'nG',NaN,'nGH',NaN,'nqmc',NaN,'nshift',NaN,'status','ok', ...
   'coef_err_pdf',cpdf,'coef_err_cdf',nan(order,1));
if strcmp(meth,'none'), qinfo.status = 'no orthant probability requested'; return, end
if any(double(tau) < 0), qinfo.status = 'negative saddlepoint component (use spa_orthant)'; return, end
if ~isa(x,'sym') && K0-tau*x.' < log(realmin)
   %  Chernoff bound P[X >= x] <= exp(K(tau) - tau'x): the probability is below the double-
   %  precision range, and is returned as zero, as an underflow
   Fup = zeros(order,1); err = zeros(order,1); kappa = ones(order,1);
   qinfo.status = 'probability below the double-precision range'; return
end
L = 3*p; nL = mi_count(B,d,L); degL = deg(1:nL); recL = mi_rec(A(1:nL,:),d,B);
t = (K2*tau.').';
uu = double(tau)./sqrt(diag(double(Si))).';
idxk = cell(L,1);
for k = 1:L, idxk{k} = find(degL == k); end
qtol = opts.qtol; if isempty(qtol), if strcmp(meth,'qmc'), qtol = 1e-3; else, qtol = 1e-8; end, end
cF = nan(order,1);
switch meth
   case 'quad'
      if d ~= 3 || ~isa(x,'double'), error('spa_mv: method ''quad'' needs d = 3 and double precision'); end
      if all(uu > 2.5)
         [Inu,Ih] = Ifreq(T,tau,recL,opts.nGH);
         Ee = exp(K0-tau*x.'-(d/2)*log(2*pi)-logdet2);
         qinfo.nGH = opts.nGH;
         if resolved(Inu), [Fup,kappa,cF] = sums(Inu); err = abs(sums(Ih)-Fup); end
      else
         Ee = 1;                                  % the exponential factor is in the weights
         logE = K0-tau*x.'+tau*K2*tau.'/2;
         nGmax = opts.nGmax; if isempty(nGmax), nGmax = 2*opts.nG; end
         G = opts.nG; Ih = [];
         while true
            Inu = Iquad3(K2,Lc,t,recL,G,logE);
            if ~resolved(Inu), break, end
            if isempty(Ih), Ih = Iquad3(K2,Lc,t,recL,floor(G/2),logE); end
            [Fup,kappa,cF] = sums(Inu); err = abs(sums(Ih)-Fup);
            if ~(err(order) > qtol*abs(Fup(order))) || 2*G > nGmax, break, end
            G = 2*G; Ih = Inu;                    % escalate; the previous grid is the half grid
         end
         qinfo.nG = G;
      end
      Isets = {Inu};
   case 'qmc'
      if ~isa(x,'double'), error('spa_mv: method ''qmc'' needs double precision'); end
      Isets = Iqmc(K2,Lc,t,recL,opts.nqmc,opts.nshift,opts.seed,K0-tau*x.'+tau*K2*tau.'/2);
      Ee = 1;
      qinfo.nqmc = opts.nqmc; qinfo.nshift = opts.nshift;
      if all(cellfun(@resolved,Isets))
         Fall = zeros(order,numel(Isets));
         for i = 1:numel(Isets), Fall(:,i) = sums(Isets{i}); end
         Fup = mean(Fall,2); err = std(Fall,0,2)/sqrt(numel(Isets));
         Im = mean(cell2mat(Isets),2); [~,kappa,cF] = sums(Im);
      end
   case 'vpa-recursion'
      if d ~= 3, error('spa_mv: method ''vpa-recursion'' is implemented for d = 3'); end
      Ee = exp(K0-tau*x.'+tau*K2*tau.'/2);
      Inu = Iexact3(vpa(K2),vpa(tau),A(1:nL,:),L,1);
      if resolved(Inu)                            % the base probabilities with half the nodes
         [Fup,kappa] = sums(Inu); err = double(abs(sums(Iexact3(vpa(K2),vpa(tau),A(1:nL,:),L,1/2))-Fup));
      end
      Isets = {Inu};
   otherwise
      error('spa_mv: unknown method %s',meth);
end
qinfo.coef_err_cdf = cF;
if ~all(cellfun(@resolved,Isets))                % the leading integral is not resolved
   Fup = nan(order,1); kappa = nan(order,1); err = nan(order,1);
   qinfo.status = 'leading integral not resolved'; return
end
if ~(double(err(order)) <= qtol*double(abs(Fup(order)))), qinfo.status = 'integration not resolved'; end
   function [F,ka,cf] = sums(Inu)
      %  Fup(M), M = 1..order, from the integrals Inu, with the cancellation indicators and (in
      %  double precision) the relative coefficient-error estimates
      F = zeros(order,1,'like',Inu); ka = zeros(order,1); cf = nan(order,1);
      ss = Inu(1); sb = abs(ss); mm = 0; dd = zeros(ndraw,1);
      for MM = 1:order
         while mm < 2*MM-1
            mm = mm+1;
            for rr = 1:mm
               ii = idxk{mm+2*rr};
               tt = a{rr}(ii).*Inu(ii);
               ss = ss+sum(tt); sb = sb+sum(abs(tt));
               if isd, for qq = 1:ndraw, dd(qq) = dd(qq)+sum(da{qq}{rr}(ii).*Inu(ii)); end, end
            end
         end
         F(MM) = Ee*ss; ka(MM) = double(sb/abs(ss));
         if isd, cf(MM) = max(abs(dd))/abs(ss); end
      end
   end
end

function ok = resolved(v)
%  the leading integral v(1) is positive and finite, tested in the working arithmetic
if isa(v,'sym')
   ok = logical(isAlways(v(1) > 0,'Unknown','false')) && logical(isfinite(v(1)));
else
   ok = isfinite(v(1)) && v(1) > 0;
end
end

function c = sym_or_double(v,x)
if isa(x,'sym'), c = vpa(sym(v)); else, c = v; end
end

function nz = nonzero(v)
%  logical index of the nonzero entries, without conversion to double (a nonzero vpa coefficient
%  can underflow in double)
if isa(v,'sym'), nz = logical(isAlways(v ~= 0,'Unknown','true')); else, nz = v ~= 0; end
end

function th = polysub_mv(th0,T,A,deg,rk,d)
%  coefficients of P(T u) for the polynomial P with coefficients th0 on the multi-index list A,
%  T upper triangular: the variables are substituted slot by slot, s_j -> sum_{k>=j} T(j,k) u_k,
%  for j = d, d-1, ..., 1, so that the u_k already introduced (k > j) are not transformed again
th = th0;
for j = d:-1:1
   new = zeros(size(th),'like',th);
   idx = find(nonzero(th));
   for q = idx.'
      nu = A(q,:); m = nu(j);
      if m == 0, new(q) = new(q)+th(q); continue, end
      Cm = comps(m,d-j+1);                                % compositions of m into the slots j..d
      for c = 1:size(Cm,1)
         mu = nu; mu(j:d) = mu(j:d)+Cm(c,:); mu(j) = mu(j)-m;
         if isa(th,'sym')                                 % exact multinomial coefficient
            coef = factorial(sym(m))/prod(factorial(sym(Cm(c,:))))*prod(T(j,j:d).^Cm(c,:));
         else
            coef = factorial(m)/prod(factorial(Cm(c,:)))*prod(T(j,j:d).^Cm(c,:));
         end
         new(rk(mu)) = new(rk(mu))+th(q)*coef;
      end
   end
   th = new;
end
end

% ----- multi-indices: graded order, within a degree the first component descending -----

function [A,B] = mi_list(d,N)
nb = N+d+1; B = zeros(nb+1);
for n = 0:nb, for k = 0:n, B(n+1,k+1) = nchoosek(n,k); end, end
A = zeros(0,d);
for k = 0:N, A = [A; comps(k,d)]; end %#ok<AGROW>
end

function C = comps(m,p)
if p == 1, C = m; return; end
C = zeros(0,p);
for k = m:-1:0
   R = comps(m-k,p-1);
   C = [C; k*ones(size(R,1),1) R]; %#ok<AGROW>
end
end

function n = mi_count(B,d,N)
n = B(N+d+1,d+1);                                % number of multi-indices with |nu| <= N
end

function r = mi_rank(M,d,B)
%  positions (1-based) of the rows of M in the graded list
k = sum(M,2);
r = zeros(size(M,1),1);
pos = k > 0; r(pos) = bin(B,k(pos)-1+d,d);
ki = k;
for i = 1:d-1
   r = r+bin(B,ki-M(:,i)-1+d-i,d-i);
   ki = ki-M(:,i);
end
r = r+1;
end

function v = bin(B,n,k)
v = zeros(size(n));
ok = n >= 0 & k >= 0 & n >= k;
nn = n(ok); v(ok) = B(sub2ind(size(B),nn(:)+1,(k+1)*ones(numel(nn),1)));
end

function rec = mi_rec(A,d,B)
%  recursion indices for He: first nonzero component, nu-e_i, nu-e_i-e_l and (nu-e_i)_l
n = size(A,1);
rec.first = zeros(n,1); rec.kp = zeros(n,1); rec.kpp = zeros(n,d); rec.cnt = zeros(n,d);
for k = 2:n
   nu = A(k,:); i = find(nu > 0,1); pr = nu; pr(i) = pr(i)-1;
   rec.first(k) = i; rec.kp(k) = mi_rank(pr,d,B);
   for l = 1:d
      if pr(l) > 0
         pp = pr; pp(l) = pp(l)-1;
         rec.kpp(k,l) = mi_rank(pp,d,B); rec.cnt(k,l) = pr(l);
      end
   end
end
end

function He = hermite_mv(Si,W,rec)
%  He(j,k) = He_{nu_k}(W(:,j),Sigma), by the recursion of Takemura and Takeuchi (1988)
n = size(W,2); nN = numel(rec.first); d = size(W,1);
Y = (Si*W).';
He = zeros(n,nN,'like',W); He(:,1) = 1;
for k = 2:nN
   i = rec.first(k);
   v = Y(:,i).*He(:,rec.kp(k));
   for l = 1:d
      if rec.cnt(k,l) > 0
         v = v-(Si(i,l)*rec.cnt(k,l))*He(:,rec.kpp(k,l));
      end
   end
   He(:,k) = v;
end
end

function Inu = Isum(Si,W,w,rec)
%  Inu(k) = sum_n w(n) He_{nu_k}(W(:,n)), over chunks of nodes
nN = numel(rec.first); n = size(W,2);
chunk = max(256,min(n,floor(1e7/nN)));
Inu = zeros(nN,1);
for c0 = 1:chunk:n
   c1 = min(n,c0+chunk-1);
   He = hermite_mv(Si,W(:,c0:c1),rec);
   Inu = Inu+He.'*w(c0:c1);
end
end

% ----- the integrals I(nu) -----

function Inu = Iquad3(S,Lc,t,rec,G,logE)
%  d = 3: Gauss-Legendre quadrature of the orthant-integral form in conditioned coordinates
%  (grid3), G nodes per coordinate and piece; the Hermite polynomials are the products of
%  univariate ones in the whitened variable v = L^{-1}(z - t).  The weights include the
%  exponential factor exp(logE), combined with the normal density in the log domain.
[W,w] = grid3(S,t,G,logE); Inu = Isum(eye(3),Lc\W,w,rec);
end

function [W,w] = grid3(S,t,G,logE)
%  nodes z1 in [t1, t1 + 16 sd1], z2 | z1 and z3 | z1, z2 over the parts above t2 and t3 of
%  intervals of 16 conditional standard deviations around the conditional means; the ranges of
%  z1 and z2 are split at narrow boundary layers created by the truncation of the later
%  coordinates (layerbp), which resolves correlations near +-1, and where their densities have
%  decayed when the truncation point is far in the tail (tailbp), with G nodes on each piece
[xg,wg] = gauleg(G); xg = xg.'; wg = wg.';
sd1 = sqrt(S(1,1)); b21 = S(2,1)/S(1,1); s2 = sqrt(S(2,2)-S(2,1)^2/S(1,1));
c31 = S(3,1)/S(1,1); s31 = sqrt(S(3,3)-S(3,1)^2/S(1,1));
beta = S(3,1:2)/S(1:2,1:2); s3 = sqrt(S(3,3)-beta*S(1:2,3));
lo1 = t(1); hi1 = t(1)+16*sd1;
[z1,w1] = glpieces(lo1,hi1,[layerbp(lo1,hi1,b21,0,s2,t(2),16) layerbp(lo1,hi1,c31,0,s31,t(3),16) ...
   tailbp(lo1,hi1,0,sd1,t(1),16)],xg,wg);
lw1 = log(w1)-z1.^2/(2*S(1,1))-log(sqrt(2*pi)*sd1);
n1 = numel(z1); Z1c = cell(1,n1); Z2c = Z1c; L12 = Z1c;
for i = 1:n1
   mu2 = b21*z1(i); lo2 = max(t(2),mu2-16*s2); hi2 = max(mu2,t(2))+16*s2;
   [z2,w2] = glpieces(lo2,hi2,[layerbp(lo2,hi2,beta(2),beta(1)*z1(i),s3,t(3),16) ...
      tailbp(lo2,hi2,mu2,s2,t(2),16)],xg,wg);
   Z1c{i} = z1(i)*ones(size(z2)); Z2c{i} = z2;
   L12{i} = lw1(i)+log(w2)-(z2-mu2).^2/(2*s2^2)-log(sqrt(2*pi)*s2);
end
z1v = [Z1c{:}].'; z2v = [Z2c{:}].'; l12 = [L12{:}].';
if isempty(z1v), W = zeros(3,0); w = zeros(0,1); return, end
mu3 = beta(1)*z1v+beta(2)*z2v;
lo3 = max(t(3),mu3-16*s3); hi3 = max(mu3,t(3))+16*s3;
Z3 = (lo3+hi3)/2+(hi3-lo3)/2*xg;                                  % n12 x G
lW3 = log((hi3-lo3)/2*wg)-(Z3-mu3).^2/(2*s3^2)-log(sqrt(2*pi)*s3);
Z1 = repmat(z1v,1,G); Z2 = repmat(z2v,1,G);
w = reshape(exp(logE+l12+lW3),[],1);
W = [reshape(Z1,1,[])-t(1); reshape(Z2,1,[])-t(2); reshape(Z3,1,[])-t(3)];
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

function [Inu,Ih] = Ifreq(T,tau,rec,G)
%  whitened form of I(nu)/phi(t,Sigma) = Re E[(iW)^nu/prod_j(tau_j+iW_j)], W ~ N(0,Si):
%  Re E[(iV)^mu/prod_j(tau_j+iW_j)] with W = T V, V standard normal, by G^d Gauss-Hermite
%  quadrature, and with G/2 nodes
Inu = run(G); Ih = run(floor(G/2));
   function I = run(G)
      d = numel(tau); C = T;
      [xg,wg] = gauher(G);
      nN = numel(rec.first);
      I = zeros(nN,1);
      ntot = G^d; chunk = max(256,floor(5e6/nN));
      sub = cell(1,d);
      for c0 = 1:chunk:ntot
         c1 = min(ntot,c0+chunk-1); id = (c0:c1).';
         [sub{:}] = ind2sub(G*ones(1,d),id);
         Z = zeros(d,numel(id)); q = ones(numel(id),1);
         for j = 1:d, Z(j,:) = xg(sub{j}).'; q = q.*wg(sub{j}); end
         Wc = C*Z;
         q = q./prod(tau(:)+1i*Wc,1).';
         iW = (1i*Z).';
         V = zeros(numel(id),nN); V(:,1) = 1;
         for k = 2:nN, V(:,k) = V(:,rec.kp(k)).*iW(:,rec.first(k)); end
         I = I+real(V.'*q);
      end
   end
end

function Isets = Iqmc(S,Lc,t,rec,nqmc,nshift,seed,logE)
%  randomized QMC (Richtmyer lattice, antithetic points, nshift random shifts), Genz's sequential
%  conditioning of the normal vector truncated to z >= t; one vector of I(nu) per shift
d = numel(t);
perm = genz_order(S,t);                       % integration order (most restrictive first)
Sp = S(perm,perm); tp = t(perm); Lp = chol(Sp,'lower');
pr = primes(1000); alpha = sqrt(pr(1:d));
st = RandStream('mt19937ar','Seed',seed);
Isets = cell(1,nshift);
k = (1:nqmc).';
for sh = 1:nshift
   shift = rand(st,1,d);
   U = mod(k*alpha+shift,1); U = [U; 1-U];                         % 2 nqmc x d
   n = size(U,1);
   Y = zeros(n,d); lw = zeros(n,1);
   for j = 1:d
      aj = (tp(j)-Y(:,1:j-1)*Lp(j,1:j-1).')/Lp(j,j);
      lpa = logPhibar(aj);                                       % log P[Z >= aj]
      lw = lw+lpa;
      Y(:,j) = tailinv(aj,lpa+log(U(:,j)));                      % P[Z >= Y] = U P[Z >= aj]
   end
   Z = Y*Lp.';
   W = zeros(d,n); W(perm,:) = (Z-tp).';
   Isets{sh} = Isum(eye(d),Lc\W,exp(logE+lw)/n,rec);
end
end

function v = logPhibar(a)
%  log P[Z >= a], Z standard normal, without underflow (erfcx for a > 0)
v = zeros(size(a));
p = a > 0;
v(p) = log(erfcx(a(p)/sqrt(2))/2)-a(p).^2/2;
v(~p) = log(erfc(a(~p)/sqrt(2))/2);
end

function y = tailinv(a,lq)
%  y >= a with log P[Z >= y] = lq: erfcinv where the tail probability is representable, and
%  otherwise Newton's method on log P[Z >= y], started from the exponential approximation
y = sqrt(2)*erfcinv(2*exp(lq));
bad = ~(isfinite(y)) | lq < -600;
if any(bad)
   ab = a(bad); lb = lq(bad);
   yb = ab+(logPhibar(ab)-lb)./max(ab,1);                         % P[Z >= a + e/a] ~ e^{-e} P[Z >= a]
   for it = 1:20
      f = logPhibar(yb)-lb;
      h = -sqrt(2/pi)./erfcx(yb/sqrt(2));                         % d/dy log P[Z >= y] (y > 0)
      yb = max(ab,yb-f./h);
   end
   y(bad) = yb;
end
y = max(y,a);
end

function perm = genz_order(S,t)
%  variable prioritization of Genz and Bretz: at each step the remaining variable with the smallest
%  conditional probability of exceeding its limit, the earlier variables set to their conditional
%  expectations given the truncation
d = numel(t); perm = 1:d; Sp = S; tp = t(:).'; L = zeros(d); y = zeros(1,d);
for j = 1:d
   best = j; bestp = Inf;
   for i = j:d
      v = Sp(i,i)-sum(L(i,1:j-1).^2);
      a = (tp(i)-L(i,1:j-1)*y(1:j-1).')/sqrt(max(v,eps));
      pa = erfc(a/sqrt(2))/2;
      if pa < bestp, best = i; bestp = pa; end
   end
   if best ~= j
      perm([j best]) = perm([best j]);
      Sp([j best],:) = Sp([best j],:); Sp(:,[j best]) = Sp(:,[best j]);
      tp([j best]) = tp([best j]); L([j best],:) = L([best j],:);
   end
   L(j,j) = sqrt(Sp(j,j)-sum(L(j,1:j-1).^2));
   for i = j+1:d
      L(i,j) = (Sp(i,j)-L(i,1:j-1)*L(j,1:j-1).')/L(j,j);
   end
   a = (tp(j)-L(j,1:j-1)*y(1:j-1).')/L(j,j);
   pa = erfc(a/sqrt(2))/2;
   if pa > 0, y(j) = exp(-a^2/2)/sqrt(2*pi)/pa; else, y(j) = a; end
end
end

function Inu = Iexact3(S,tau,AL,L,fac)
%  d = 3, variable precision: I(nu) = sum_{rho <= nu} binom(nu,rho)(-tau)^(nu-rho) J(rho-1),
%  with J from Lemma 1 (Jtable3), as three one-dimensional transforms; in the order of AL.
%  fac scales the number of quadrature nodes of the base orthant probabilities (1, or 1/2 for
%  the error estimate)
Si = inv(S); t = (S*tau.').';
Jm = Jtable3(L,S,Si,t,fac);
n = L+1;
X = Jm;
for j = 1:3
   T = vpa(zeros(n,n));
   for nn = 0:L
      for rr = 0:nn
         T(nn+1,rr+1) = nchoosek(sym(nn),rr)*(-tau(j))^(nn-rr);
      end
   end
   perm = [j setdiff(1:3,j)];
   Y = permute(X,perm);
   Y = reshape(T*reshape(Y,n,[]),n,n,n);
   X = ipermute(Y,perm);
end
Inu = X(sub2ind([n n n],AL(:,1)+1,AL(:,2)+1,AL(:,3)+1));
end

function He = hermite3(N,Si,x)
%  He(nu+1) = He_nu(x,Sigma), |nu| <= N (3-d array), for Jtable3
y = Si*x(:);
He = zeros(N+1,N+1,N+1,'like',y); He(1,1,1) = 1;
for k = 1:N
   for a1 = k:-1:0
      for a2 = k-a1:-1:0
         a3 = k-a1-a2; nu = [a1 a2 a3];
         j = find(nu > 0,1); pr = nu; pr(j) = pr(j)-1;
         v = y(j)*He(pr(1)+1,pr(2)+1,pr(3)+1);
         for kk = 1:3
            if pr(kk) > 0
               pp = pr; pp(kk) = pp(kk)-1;
               v = v-Si(j,kk)*pr(kk)*He(pp(1)+1,pp(2)+1,pp(3)+1);
            end
         end
         He(a1+1,a2+1,a3+1) = v;
      end
   end
end
end

function Jm = Jtable3(L,S,Si,t,fac)
%  Jm(r+2) = J(r) = prod_j (-D_j)^{r_j+1} Pbar(t,S), r_j >= -1, |r+1| <= L (Lemma 1)
Jm = vpa(zeros(L+1,L+1,L+1));
He = hermite3(L,Si,t);
phi3 = exp(-t*Si*t.'/2)/((2*vpa(pi))^(3/2)*sqrt(det(S)));
Jm(2:L+1,2:L+1,2:L+1) = phi3*He(1:L,1:L,1:L);                    % r >= 0
for i = 1:3                                                       % I = {i}
   Jx = setdiff(1:3,i);
   SJ = S(Jx,Jx); SJi = inv(SJ); tJ = t(Jx);
   bJ = SJi*tJ.'; cJ = S(i,Jx)*SJi;
   cv = S(i,i)-S(i,Jx)*SJi*S(Jx,i);
   g = exp(-tJ*SJi*tJ.'/2)/(2*vpa(pi)*sqrt(det(SJ)))*erfc((t(i)-cJ*tJ.')/sqrt(2*cv))/2;
   Jr = vpa(zeros(L,L)); Jr(1,1) = g;
   for k = 1:L-2
      for q1 = k:-1:0
         q2 = k-q1; q = [q1 q2];
         jj = find(q > 0,1); r = q; r(jj) = r(jj)-1;
         v = bJ(jj)*Jr(r(1)+1,r(2)+1);
         for kk = 1:2
            if r(kk) > 0
               rr = r; rr(kk) = rr(kk)-1;
               v = v-r(kk)*SJi(jj,kk)*Jr(rr(1)+1,rr(2)+1);
            end
         end
         full = zeros(1,3); full(Jx) = r;
         v = v-cJ(jj)*Jm(full(1)+2,full(2)+2,full(3)+2);
         Jr(q1+1,q2+1) = v;
      end
   end
   for q1 = 0:L-2
      for q2 = 0:L-2-q1
         idx = zeros(1,3); idx(i) = 1; idx(Jx) = [q1 q2]+2;
         Jm(idx(1),idx(2),idx(3)) = Jr(q1+1,q2+1);
      end
   end
end
for j = 1:3                                                       % I = the other two
   Ix = setdiff(1:3,j);
   Sjj = S(j,j);
   SI = S(Ix,Ix)-S(Ix,j)*S(j,Ix)/Sjj;
   g = exp(-t(j)^2/(2*Sjj))/sqrt(2*vpa(pi)*Sjj)*orthant2v(t(Ix)-S(Ix,j).'*t(j)/Sjj,SI,fac);
   Jr = vpa(zeros(L,1)); Jr(1) = g;
   for q = 0:L-2
      v = t(j)/Sjj*Jr(q+1);
      if q > 0, v = v-q/Sjj*Jr(q); end
      for ii = 1:2
         i = Ix(ii);
         idx = zeros(1,3); idx(Ix) = 1; idx(i) = 2; idx(j) = q+2;
         v = v-S(i,j)/Sjj*Jm(idx(1),idx(2),idx(3));
      end
      Jr(q+2) = v;
   end
   for q = 0:L-1
      idx = zeros(1,3); idx(Ix) = 1; idx(j) = q+2;
      Jm(idx(1),idx(2),idx(3)) = Jr(q+1);
   end
end
Jm(1,1,1) = orthant3v(t,S,fac);                                       % r = (-1,-1,-1)
end

function P = orthant2v(t,S,fac)
%  P[Z1 >= t1, Z2 >= t2], Z ~ N(0,S), Gauss-Legendre in variable precision, over a range of
%  2R standard deviations, R = sqrt(2 digits log 10) + 2, so that the truncation is negligible,
%  split at the boundary layer of the truncation of z2 and at the tail of z1 (glpieces); closed
%  form in z2
[R,G] = vrange; G = max(8,round(fac*G));
[xg,wg] = gaulegv(G);
sd1 = sqrt(S(1,1)); b = S(1,2)/S(1,1); s2 = sqrt(S(2,2)-S(1,2)^2/S(1,1));
lo = t(1); hi = t(1)+2*R*sd1;
[z1,w1] = glpieces(lo,hi,[layerbp(lo,hi,b,0,s2,t(2),R) tailbp(lo,hi,0,sd1,t(1),R)],xg,wg);
P = sum(w1.*exp(-z1.^2/(2*S(1,1)))/(sqrt(2*vpa(pi))*sd1).*erfc((t(2)-b*z1)/(sqrt(vpa(2))*s2))/2);
end

function P = orthant3v(t,S,fac)
%  P[Z >= t], Z ~ N(0,S) trivariate: Gauss-Legendre in z1 and z2 | z1, split at the boundary
%  layers of the truncations and at the tails (glpieces), closed form in z3
[R,G] = vrange; G = max(8,round(fac*G));
[xg,wg] = gaulegv(G);
sd1 = sqrt(S(1,1)); b21 = S(1,2)/S(1,1); s2 = sqrt(S(2,2)-S(1,2)^2/S(1,1));
c31 = S(1,3)/S(1,1); s31 = sqrt(S(3,3)-S(1,3)^2/S(1,1));
S12 = S(1:2,1:2); beta = S(3,1:2)/S12; s3 = sqrt(S(3,3)-beta*S(1:2,3));
lo = t(1); hi = t(1)+2*R*sd1;
[z1,w1] = glpieces(lo,hi,[layerbp(lo,hi,b21,0,s2,t(2),R) layerbp(lo,hi,c31,0,s31,t(3),R) ...
   tailbp(lo,hi,0,sd1,t(1),R)],xg,wg);
P = vpa(0);
for i = 1:numel(z1)
   mu = b21*z1(i);
   lo = max(t(2),mu-R*s2); hi = max(mu,t(2))+R*s2;
   [z2,w2] = glpieces(lo,hi,[layerbp(lo,hi,beta(2),beta(1)*z1(i),s3,t(3),R) ...
      tailbp(lo,hi,mu,s2,t(2),R)],xg,wg);
   f2 = exp(-(z2-mu).^2/(2*s2^2))/(sqrt(2*vpa(pi))*s2).*erfc((t(3)-beta(1)*z1(i)-beta(2)*z2)/(sqrt(vpa(2))*s3))/2;
   P = P+w1(i)*exp(-z1(i)^2/(2*S(1,1)))/(sqrt(2*vpa(pi))*sd1)*sum(w2.*f2);
end
end

function [R,G] = vrange
%  half-range (in standard deviations) and number of Gauss-Legendre nodes for the current digits
R = sqrt(2*digits*log(10))+2; G = 4*digits;
end

function [x,w] = gaulegv(n)
%  Gauss-Legendre nodes and weights on [-1,1] in variable precision (Newton on P_n)
persistent cache
key = sprintf('n%d_d%d',n,digits);
if isempty(cache), cache = struct(); end
if isfield(cache,key), x = cache.(key){1}; w = cache.(key){2}; return; end
x = vpa(zeros(n,1)); w = x;
for i = 1:n
   z = vpa(cos(pi*(i-0.25)/(n+0.5)));
   for it = 1:100
      p1 = vpa(1); p2 = vpa(0);
      for j = 1:n
         p3 = p2; p2 = p1; p1 = ((2*j-1)*z*p2-(j-1)*p3)/j;
      end
      pp = n*(z*p1-p2)/(z^2-1);
      dz = p1/pp; z = z-dz;
      if abs(double(dz)) < 10^(-digits+5), break; end
   end
   x(i) = -z; w(i) = 2/((1-z^2)*pp^2);
end
cache.(key) = {x,w};
end

function [x,w] = gauher(n)
%  Gauss-Hermite nodes and weights for the weight exp(-x^2/2)/sqrt(2 pi) (Golub-Welsch)
b = sqrt(1:n-1);
[V,D] = eig(diag(b,1)+diag(b,-1));
[x,i] = sort(diag(D)); w = V(1,i)'.^2;
end

function [x,w] = gauleg(n)
%  Gauss-Legendre nodes and weights on [-1,1] (Golub-Welsch)
k = 1:n-1; b = k./sqrt(4*k.^2-1);
[V,D] = eig(diag(b,1)+diag(b,-1));
[x,i] = sort(diag(D)); w = 2*V(1,i)'.^2;
end
