function K = quadratic_forms_cgf(Mats, varargin)
%QUADRATIC_FORMS_CGF Joint cgf of the quadratic forms Q_i = z'A_i z, i = 1..d, Mats = {A_1,...,A_d}
%   (symmetric p x p), z ~ N(mu, Sigma) (options 'Sigma' and 'mu'; default standard normal):
%      K(t) = -1/2 log det(I - 2C) + nu'C(I - 2C)^{-1}nu,  C = sum_i t_i Sigma^{1/2}A_i Sigma^{1/2},
%      nu = Sigma^{-1/2}mu,
%   multiplied by 'scale' (default 1; the sum of scale independent copies for an integer scale).
%   The domain is the set where I - 2C is positive definite.  The derivatives are computed in
%   closed form by the trace recursion (qf_taylor), not by automatic differentiation, so that
%   large p is fast; K itself is evaluated through a Cholesky factorization.
p = inputParser; p.addParameter('Sigma', []); p.addParameter('mu', []); p.addParameter('scale', 1);
p.parse(varargin{:});
S = p.Results.Sigma; mu = p.Results.mu; sc = p.Results.scale;
d = numel(Mats); n = size(Mats{1}, 1);
if isempty(S), Sh = eye(n); else, [V, E] = eig((S + S.')/2); Sh = V*diag(sqrt(diag(E)))*V.'; end
B = cell(1, d);
for i = 1:d, A = (Mats{i} + Mats{i}.')/2; Bi = Sh*A*Sh; B{i} = (Bi + Bi.')/2; end
if isempty(mu), m = zeros(n, 1); else, m = Sh \ reshape(mu, [], 1); end
fun = @(t) qfval(B, m, sc, t);
dom = @(t) qfdomain(B, t);
tf = @(t, N, M) qf_taylor(B, m, sc, t, N, M);
K = CGF(fun, d, 'domain', dom, 'Taylor', tf);
end
function v = qfval(B, m, sc, t)
if isa(t, 'tps'), error('quadratic_forms_cgf:tps', 'quadratic_forms_cgf: the derivatives are computed in closed form; use taylor or derivatives'); end
C = zeros(size(B{1})); for j = 1:numel(B), C = C + t(j)*B{j}; end
[R, flag] = chol(eye(size(C)) - 2*C);
if flag, v = NaN; return; end
w = R \ (R.' \ m);
v = sc*(-sum(log(diag(R))) + 0.5*(m.'*w - m.'*m));
end
function tf = qfdomain(B, t)
C = zeros(size(B{1})); for j = 1:numel(B), C = C + t(j)*B{j}; end
[~, flag] = chol(eye(size(C)) - 2*C); tf = flag == 0;
end
