function sd = stud_saddle(K, q)
%STUD_SADDLE The saddlepoint (s, t, v) of the expansions for Q = X/sqrt(Y) at q: the root of
%   K_s = q v, K_t = v^2, q s + 2 t v = 0 with v > 0, by Newton's method with continuation in q
%   from q0 = E(X)/E(Y)^(1/2), where s = t = 0.  Returns a struct with s, t, v, q0.
P0 = stud_jet(K, 0, 0, 2);
EU = coef(P0, [1 0]); EV = coef(P0, [0 1]);
if ~(EV > 0), error('CGF:studentized', 'studentized_saddlepoint: E(Y) must be positive'); end
q0 = EU/sqrt(EV);
s = 0; t = 0; v = sqrt(EV);
steps = max(4, floor(abs(q - q0)/0.25) + 1);
for k = 1:steps
   [s, t, v] = continuation(K, q0 + (q - q0)*(k - 1)/steps, q0 + (q - q0)*k/steps, s, t, v, 0);
end
sd = struct('s', s, 't', t, 'v', v, 'q0', q0);
end

function [s, t, v] = continuation(K, qa, qb, s, t, v, depth)
% from the solution at qa to qb, halving the step on failure
try
   [s, t, v] = newton3(K, qb, s, t, v);
catch err
   if depth >= 12 || strcmp(err.identifier, 'CGF:studentized'), rethrow(err); end
   qm = (qa + qb)/2;
   [s, t, v] = continuation(K, qa, qm, s, t, v, depth + 1);
   [s, t, v] = continuation(K, qm, qb, s, t, v, depth + 1);
end
end

function [s, t, v] = newton3(K, q, s, t, v)
% Newton's method on the saddlepoint equations, Jacobian from the quadratic Taylor polynomial
tol = 50*eps; dprev = Inf;
for it = 1:60
   P = taylor(K, [s; t], 2);
   if isempty(P), error('CGF:domain', 'outside the domain'); end
   Ks = coef(P, [1 0]); Kt = coef(P, [0 1]);
   Kss = coef(P, [2 0]); Kst = coef(P, [1 1]); Ktt = coef(P, [0 2]);
   F = [Ks - q*v; Kt - v*v; q*s + 2*t*v];
   A = [Kss Kst -q; Kst Ktt -2*v; q 2*v 2*t];
   ws = warning('off', 'MATLAB:singularMatrix'); wn = warning('off', 'MATLAB:nearlySingularMatrix');
   d = A\F;
   warning(ws); warning(wn);
   if ~all(isfinite(d)), error('CGF:newton', 'singular Jacobian'); end
   s = s - d(1); t = t - d(2); v = v - d(3);
   dn = sum(abs(d)); scale = 1 + abs(s) + abs(t) + abs(v);
   if dn < tol*scale, return; end
   if dn < 1e-7*scale && dn >= dprev/2, return; end   % stagnation at the rounding level of K
   dprev = dn;
end
error('CGF:newton', 'the saddlepoint iteration did not converge at q = %g', q);
end
