function [Fup,tau,err] = spa_orthant(Kfun,x,order,opts)
%SPA_ORTHANT P[X >= x] in any dimension d (orders 1..order), with the inclusion-exclusion (17) of
%   the paper when the saddlepoint has negative components: every term is the orthant probability
%   of a marginal with some components negated, computed by spa_uni (one dimension), spa_bi (two),
%   spa_tri (three) or spa_mv (four or more), recursively when its own saddlepoint has negative
%   components.  Kfun(nu,t) as in spa_mv; opts are passed to spa_bi, spa_tri and spa_mv.  err(M)
%   adds the absolute error estimates of the terms of order M (quadrature: the change with half
%   the nodes, |F| qerr for the bivariate terms; simulation: the standard error), so that the
%   amplification of these errors by cancellation among the terms is visible relative to Fup; it
%   is NaN when a term has no estimate, and it does not include the asymptotic error of the
%   expansions.
%   opts.Kwhite (see spa_mv) is transformed for every term: the term with the kept components K
%   and the components in T negated has the cgf K_B(u) = K(B u), where B (d x k) has the column
%   flip_j e_j for each kept j, and is passed the callback @(nu,u,M) Kwhite(nu,u,B*M).  Kwhite must
%   therefore accept a rectangular d x k matrix in place of T (the cgf of Y = M' X, of dimension
%   k), as it does when it is written in terms of K(M u) for a general matrix M.
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Uniform saddlepoint expansions for multivariate tail probabilities"; free to use and modify with a citation of the paper.
if nargin < 4, opts = struct(); end
if ~isa(Kfun,'function_handle'), error('spa_orthant: Kfun must be a function handle'); end
if ~isnumeric(x) || isempty(x) || ~all(isfinite(x(:))), error('spa_orthant: x must be a finite numeric vector'); end
if ~isscalar(order) || order < 1 || order ~= round(order), error('spa_orthant: order must be a positive integer'); end
d = numel(x); x = reshape(x,1,d);
if d == 1
   Fup = uni_upper(@(k,s) Kfun(k,s),x,order); tau = NaN; err = zeros(order,1);
   return
end
tau = spa_saddle(Kfun,x,zeros(1,d));
if all(tau >= 0)
   [Fup,err] = core(Kfun,x,order,tau,opts);
   return
end
N = find(tau < 0);
Fup = zeros(order,1); err = zeros(order,1);
for mask = 0:2^numel(N)-1
   T = N(logical(bitget(mask,1:numel(N))));
   sgn = (-1)^numel(T);
   keep = setdiff(1:d,setdiff(N,T));
   flip = ones(1,d); flip(T) = -1;
   if isempty(keep), Fup = Fup+sgn; continue, end
   Km = @(nu,t) marg(Kfun,nu,t,keep,flip,d);
   xk = flip(keep).*x(keep);
   ok = opts;
   if isfield(opts,'Kwhite') && ~isempty(opts.Kwhite)
      B = zeros(d,numel(keep)); B(sub2ind([d numel(keep)],keep,1:numel(keep))) = flip(keep);
      ok.Kwhite = @(nu,u,M) opts.Kwhite(nu,u,B*M);          % K_B(M u) = K(B M u)
   end
   if numel(keep) == d
      [F,e] = core(Km,xk,order,abs(tau),ok);
   else
      [F,~,e] = spa_orthant(Km,xk,order,ok);
   end
   Fup = Fup+sgn*F; err = err+e;
end
end

function v = marg(Kfun,nu,t,keep,flip,d)
%  derivative of the cgf of the kept components, with the components in flip negated
nf = zeros(1,d); tf = zeros(1,d,'like',t);
nf(keep) = nu; tf(keep) = t;
v = prod(flip(keep).^nu)*Kfun(nf,flip.*tf);
end

function [F,e] = core(Kfun,x,order,tau0,opts)
d = numel(x);
o = opts; o.tau0 = tau0;
switch d
   case 2
      [~,F,~,~,~,qe] = spa_bi(Kfun,x,order,tau0,[],[],o); e = abs(F).*qe;
   case 3
      [~,F,~,~,e] = spa_tri(Kfun,x,order,o);
   otherwise
      [~,F,~,~,e] = spa_mv(Kfun,x,order,o);
end
end

function F = uni_upper(K1,xj,order)
% P[X >= xj] for the univariate cgf K1(k,s) (NaN outside the domain), by spa_uni; the
% saddlepoint by spa_saddle in one dimension
N = 2*order+1;
s = spa_saddle(K1,xj,0);
Kd = arrayfun(@(k) K1(k,s),0:N)';
[~,~,~,F] = spa_uni(Kd,s,xj,order);
end
