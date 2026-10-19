function [Fup,tau,canc,kappa,info] = spa_orthant2(Kfun,x,order,opts)
%SPA_ORTHANT2 P[X1 >= x1, X2 >= x2] by the bivariate expansion of Theorem 1, with the
%   inclusion-exclusion of Section 3.3 when the saddlepoint has negative components.
%   Kfun(nu,t) as in spa_bi; Fup(M), M = 1..order.  Marginal terms use spa_uni.
%   opts (optional): as in spa_bi (opts.Kwhite is transformed for the negated components).
%   With opts.logscale = c every term of the inclusion-exclusion is scaled by exp(-c), the
%   marginal terms and the constant term included; the marginal terms are computed in double
%   precision before scaling, so the scaling extends the range of the bivariate term only.
%   canc, kappa: cancellation indicators of the bivariate term of the highest order (see spa_bi);
%   they do not measure cancellation between the terms of the inclusion-exclusion.
%   info: the info struct of spa_bi for the bivariate term (status, quadrature nodes,
%   coefficient-error estimates).
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Saddlepoint expansions of arbitrary order for multivariate distribution functions"; free to use and modify with a citation of the paper.
if nargin < 4, opts = struct(); end
if isfield(opts,'logscale'), c = opts.logscale; else, c = 0; end
[~,Fup,tau,canc,kappa,~,info] = spa_bi(Kfun,x,order,[],[],[],opts); canc = canc(end); kappa = kappa(end);
if all(tau >= 0), return, end
N = find(tau < 0);
Fup = zeros(order,1);
for T = subsets(N)
   T = T{1}; sgn = (-1)^numel(T);
   keep = setdiff(1:2,setdiff(N,T));               % P cup T
   flip = ones(1,2); flip(T) = -1;
   if numel(keep) == 2
      Kf = @(nu,t) prod(flip.^nu)*Kfun(nu,flip.*t);
      of = opts;
      if isfield(opts,'Kwhite') && ~isempty(opts.Kwhite)
         F = diag(flip); of.Kwhite = @(nu,u,M) opts.Kwhite(nu,u,F*M);   % K_f(M u) = K(F M u)
      end
      [~,F2,~,c2,k2,~,info] = spa_bi(Kf,flip.*x,order,abs(tau),[],[],of); canc = c2(end); kappa = k2(end);
      Fup = Fup+sgn*F2;
   elseif numel(keep) == 1
      j = keep; f = flip(j); e = zeros(1,2); e(j) = 1;
      K1 = @(k,s) f^k*Kfun(k*e,f*s*e);           % marginal cgf of f*X_j
      F1 = uni_upper(K1,f*x(j),order);
      Fup = Fup+sgn*sign(F1).*exp(log(abs(F1))-c);
   else
      Fup = Fup+sgn*exp(-c);
   end
end
end

function S = subsets(N)
S = cell(1,2^numel(N));
for mask = 0:2^numel(N)-1
   S{mask+1} = N(logical(bitget(mask,1:numel(N))));
end
end

function F = uni_upper(K1,xj,order)
% P[X >= xj] for the univariate cgf K1(k,s) (NaN outside the domain), by spa_uni; the
% saddlepoint by spa_saddle in one dimension
N = 2*order+1;                                  % derivatives needed by spa_uni
s = spa_saddle(K1,xj,0);
Kd = arrayfun(@(k) K1(k,s),0:N)';
[~,~,~,F] = spa_uni(Kd,s,xj,order);             % upper tail, computed directly
end
