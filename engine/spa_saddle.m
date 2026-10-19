function [tau,info] = spa_saddle(Kfun,x,tau0)
%SPA_SADDLE Saddlepoint K'(tau) = x of a cumulant generating function in any dimension d.
%   [tau,info] = spa_saddle(Kfun,x,tau0)
%   Kfun(nu,t): derivative of order nu (1 x d integers) of K at t (1 x d), NaN outside the domain;
%   x: the point (1 x d); tau0: starting value (default zeros).  Newton's method with backtracking
%   on the convex objective K(tau) - tau'x (a step is accepted only if it stays in the domain and
%   decreases the objective).  Convergence is measured by the Newton decrement
%   delta = {g' K''(tau)^{-1} g}^{1/2}, g = K'(tau) - x, which is invariant under linear changes of
%   units of the components of X.  Once delta <= 1e-6 the iteration is in its quadratic phase:
%   full Newton steps are taken without the objective test, which rounding errors in K would
%   otherwise make fail, until delta <= tol, tol = 1e-12 in double precision and 10^(-0.7 digits)
%   in variable precision, or for at most three steps (twelve in variable precision).  Before
%   that phase the iteration is declared stalled, and an error is raised, only when a step no
%   longer changes the iterate in the working arithmetic, or the backtracking cannot find a
%   decrease of the objective; there is no threshold on the size of the step in the units of x.
%   Components of tau that are zero up to the accuracy of the iteration (|tau_j| K''_jj^{1/2}
%   below tol) are set to zero, so that a saddlepoint on a coordinate hyperplane is not reported
%   with a spurious sign, but only if the decrement at the rounded point is still at most
%   max(tol, 2 delta); otherwise tau is returned unrounded.  The gradient, Hessian and decrement are
%   recomputed at the point actually returned, and the result is accepted if that decrement is at
%   most 1e-6 (an error is raised otherwise).
%   info: struct with the number of iterations, the decrement and the residual K'(tau) - x at the
%   returned tau (converted to double), the tolerance tol, and snapped (logical, the
%   components set to zero).
%   Works in double precision and, with vpa x and tau0, in variable precision; in variable precision
%   all tests are made in the working precision (the Hessian is tested for positive definiteness
%   after scaling to unit diagonal).
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Saddlepoint expansions of arbitrary order for multivariate distribution functions"; free to use and modify with a citation of the paper.
d = numel(x); x = reshape(x,1,d);
if nargin < 3 || isempty(tau0), tau0 = zeros(1,d,'like',x); end
tau = reshape(tau0,1,d);
if isa(x,'sym') && ~isa(tau,'sym'), tau = vpa(tau); end
if isa(tau,'sym'), tol = 10^(-0.7*digits); nqmax = 12; else, tol = 1e-12; nqmax = 3; end
obj = @(t) Kfun(zeros(1,d),t)-t*x.';
f = obj(tau);
if ~isfinite(double(f)), error('spa_saddle: the starting value is outside the domain of K'); end
nq = 0;
for it = 1:200
   [g,~,step,delta] = newton(Kfun,tau,x,d);
   if delta <= 1e-6                                      % quadratic phase: full steps, no test
      tn = tau+step;
      if isfinite(double(obj(tn))), tau = tn; f = obj(tau); end
      nq = nq+1;
      if delta <= tol || nq >= nqmax, break, end
      continue
   end
   if isequal(tau+step,tau)                              % the step is below rounding level
      error('spa_saddle: the iteration stalled at rounding level with decrement %.2g',delta);
   end
   lam = 1;
   while true
      tn = tau+lam*step; fn = obj(tn);
      if isfinite(double(fn)) && double(fn-f) <= double(1e-4*lam*(g*step.')), break; end
      lam = lam/2;
      if lam < 1e-14 || isequal(tn,tau)
         error('spa_saddle: saddlepoint iteration failed to find a descent step (decrement %.2g)',delta);
      end
   end
   tau = tn; f = fn;
end
%  diagnostics at the returned point, after the last step and the optional rounding of components
[g,H,~,delta] = newton(Kfun,tau,x,d);
sc = abs(tau.*sqrt(abs(diag(H))).');
if isa(sc,'sym')
   snap = logical(isAlways(sc < tol,'Unknown','false')) & logical(isAlways(tau ~= 0,'Unknown','true'));
else
   snap = sc < tol & tau ~= 0;
end
if any(snap)
   ts = tau; ts(snap) = 0;
   [gs,~,~,ds] = newton(Kfun,ts,x,d);
   if ds <= max(double(tol),2*delta), tau = ts; g = gs; delta = ds; else, snap(:) = false; end
end
if ~(delta <= 1e-6)
   error('spa_saddle: saddlepoint iteration did not converge (Newton decrement %.2g)',delta);
end
info = struct('iterations',it,'decrement',delta,'residual',double(g),'tol',double(tol),'snapped',snap);
end

function [g,H,step,delta] = newton(Kfun,tau,x,d)
%  gradient, Hessian, Newton step and Newton decrement at tau; the Hessian is tested for positive
%  definiteness after scaling to unit diagonal, in the working arithmetic
E = eye(d);
g = zeros(1,d,'like',x); H = zeros(d,d,'like',x);
for i = 1:d
   g(i) = Kfun(E(i,:),tau)-x(i);
   for j = i:d
      H(i,j) = Kfun(E(i,:)+E(j,:),tau); H(j,i) = H(i,j);
   end
end
dg = diag(H);
if isa(dg,'sym'), okd = all(isAlways(dg > 0,'Unknown','false')); else, okd = all(isfinite(dg) & dg > 0); end
if ~okd, error('spa_saddle: the Hessian of K is not positive definite at the current iterate'); end
Ds = diag(1./sqrt(dg));                                  % diagonal scaling: the Newton step does
Hs = Ds*H*Ds;                                            % not depend on the units of the components
Hsd = double(Hs);
if ~all(isfinite(Hsd(:))) || min(eig((Hsd+Hsd.')/2)) <= 0
   if ~isa(Hs,'sym') || any(double(eig(Hs)) <= 0)
      error('spa_saddle: the Hessian of K is not positive definite at the current iterate');
   end
end
step = -(Ds*(Hs\(Ds*g.'))).';
delta = sqrt(max(double(-g*step.'),0));                  % Newton decrement
end
