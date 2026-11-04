function [F,part,info] = wang_bivar(Kfun,x,ord)
%WANG_BIVAR First-order bivariate approximation of Wang (1990, Theorem 1) to the distribution
%   function F(x1,x2) = P[X1 <= x1, X2 <= x2], with n = 1 (pass the cgf of the vector itself).
%   Kfun(nu,t): derivative of order nu = [nu1 nu2] of the cgf at t, NaN outside the domain.
%   The inner inversion integral, over the first coordinate, is approximated by the
%   Lugannani-Rice formula and the outer one by Watson's lemma, so the formula depends on the
%   order of the coordinates: ord = [1 2] (default) or [2 1].  Wang recommends the order with the
%   smaller signed root info.wu0.  F = sum(part), the four terms of his equation (27): the
%   bivariate normal distribution function (17) and the corrections (18), (25) and (26).
%   The formula needs the cgf at (0,tau2), the saddlepoint with its first component replaced by
%   zero; F is NaN when that point is outside the domain or when a component of the saddlepoint
%   is zero up to rounding, |tau_j| K_jj^(1/2) < 1e-8 (where Wang takes limits).  The term (18) contains phi(vbar0) with
%   vbar0^2 = -2{K(0,tau2) - tau2 x2} (his Lemma 4), which can be negative; it is evaluated
%   through exp{K(0,tau2) - tau2 x2}/sqrt(2 pi), which is what his derivation produces and equals
%   phi(vbar0) when vbar0^2 >= 0, and info.vbar2 reports vbar0^2 (Table 3 of the paper).
%   info: tau, th1 (marginal saddlepoint of X1), x1, wu0, v0, b, y1, rho1, G, vbar2.
%   Checked against the values printed in Wang (1990, Table 1) by wang_check.m.  MATLAB port of
%   wang_bivar of the Julia programs.  Requires the Statistics and Machine Learning Toolbox
%   (mvncdf, normcdf, normpdf).  Used by lap_table and nsweep_table.
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Uniform saddlepoint expansions for multivariate tail probabilities"; free to use and modify with a citation of the paper.
if nargin < 3, ord = [1 2]; end
x = reshape(x,1,2);
if isequal(ord,[2 1])
   K = @(nu,t) Kfun(nu([2 1]),t([2 1])); x = x([2 1]);
else
   K = Kfun;
end
part = nan(1,4); F = NaN;
info = struct('tau',[NaN NaN],'wu0',NaN,'v0',NaN,'vbar2',NaN);
tau = saddle2(K,x,[0 0]); t0 = tau(1); u0 = tau(2);
if ~all(isfinite(tau)), return, end
% a saddlepoint component that is zero up to rounding (where Wang takes limits): the formula is 0/0
if any(abs(tau).*sqrt([K([2 0],tau) K([0 2],tau)]) < 1e-8), info.tau = tau; return, end
Ksw = @(nu,t) K(nu([2 1]),t([2 1]));
th1 = inner(Ksw,0,x(1),t0);                        % marginal saddlepoint of X1: K_1(th1,0) = x1
if isnan(th1), return, end
l0 = K([0 0],[th1 0])-th1*x(1);                    % l(0)
lu0 = K([0 0],tau)-tau*x';                         % l(u0) = K(tau) - tau'x
K0u0 = K([0 0],[0 u0]);                            % K(0,u0)
if ~isfinite(K0u0), return, end
x1 = sign(th1)*sqrt(max(-2*l0,0));                 % signed root of the marginal of X1
wu0 = sign(t0)*sqrt(max(2*(K0u0-lu0-u0*x(2)),0));  % signed root of the inner integral at u0
v0 = sign(u0)*sqrt(max(2*(l0-lu0),0));             % signed root of the outer integral
b = (wu0-x1)/v0;
y1 = (v0-b*x1)/sqrt(1+b^2); rho1 = -b/sqrt(1+b^2);
Ktt = K([2 0],tau); Ktu = K([1 1],tau); Kuu = K([0 2],tau);
G = sqrt(Kuu-Ktu^2/Ktt);
gx = 1/wu0-1/(t0*sqrt(Ktt)); gy = 1/v0-1/(u0*G);
vbar2 = -2*(K0u0-u0*x(2));                         % vbar0^2 of Wang's Lemma 4
info = struct('tau',tau,'th1',th1,'x1',x1,'wu0',wu0,'v0',v0,'b',b,'y1',y1,'rho1',rho1,'G',G,'vbar2',vbar2);
if ~(all(isfinite([x1 wu0 v0 b y1 rho1 G gx gy])) && abs(rho1) < 1), return, end
part(1) = mvncdf([x1 y1],[0 0],[1 rho1; rho1 1]);  % (17): P[Z1 <= x1, Z2 <= y1], correlation rho1
part(2) = normcdf(wu0)*exp(K0u0-u0*x(2))/sqrt(2*pi)*gy;   % (18): phi(vbar0) as above
part(3) = normcdf(v0)*normpdf(x1)*gx;              % (25)
part(4) = normpdf(x1)*normpdf(v0)*gx*gy;           % (26)
F = sum(part);
end

function g = inner(K,s,x2,g0)
%  solve K_2(s,g) = x2 by Newton with step halving (NaN if it fails)
g = g0; if ~isfinite(K([0 0],[s g])), g = 0; end
for it = 1:200
   f1 = K([0 1],[s g])-x2; f2 = K([0 2],[s g]);
   st = -f1/f2; lam = 1;
   while ~isfinite(K([0 0],[s g+lam*st])) && lam > 1e-12, lam = lam/2; end
   g = g+lam*st;
   if abs(lam*st) < 1e-14*(1+abs(g)), break; end
end
if ~isfinite(K([0 0],[s g])) || abs(K([0 1],[s g])-x2) > 1e-8*(1+abs(x2)), g = NaN; end
end

function tau = saddle2(K,x,tau0)
%  saddlepoint by Newton with backtracking, polished by full Newton steps (as in kl_bivar)
tau = tau0;
obj = @(t) K([0 0],t)-t*x';
f = obj(tau);
for it = 1:200
   g = [K([1 0],tau) K([0 1],tau)]-x;
   H = [K([2 0],tau) K([1 1],tau); K([1 1],tau) K([0 2],tau)];
   st = -(H\g')';
   if norm(st) < 1e-14*(1+norm(tau)), break; end
   lam = 1;
   while true
      tn = tau+lam*st; fn = obj(tn);
      if isfinite(fn) && fn <= f+1e-4*lam*(g*st'), break; end
      lam = lam/2; if lam < 1e-14, break; end
   end
   if norm(lam*st) < 1e-13*(1+norm(tau)), tau = tn; break; end
   tau = tn; f = fn;
end
for it = 1:4
   g = [K([1 0],tau) K([0 1],tau)]-x;
   H = [K([2 0],tau) K([1 1],tau); K([1 1],tau) K([0 2],tau)];
   tn = tau-(H\g')';
   if ~isfinite(K([0 0],tn)), break; end
   tau = tn;
end
end
