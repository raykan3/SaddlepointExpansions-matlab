function ok = test_spx(verbose)
%TEST_SPX Tests of the MATLAB package against SaddlepointExpansions.jl (julia_reference.csv, made by
%   julia_reference.jl) and against closed forms.  Returns true if every check passes; prints the
%   largest relative difference of each case.
if nargin < 1, verbose = true; end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..'), fullfile(here, '..', 'engine'));
ok = true;
%% derivatives of the Taylor class against closed forms
v = tps.vars(2, 6);
p = exp(v(1) + 2*v(2)).*log(1 + v(1)) + sqrt(4 + v(2)).*atan(0.3 + v(1).*v(2)) + sinh(v(1))./cosh(v(2) + 1);
f = @(a, b) exp(a + 2*b).*log(1 + a) + sqrt(4 + b).*atan(0.3 + a.*b) + sinh(a)./cosh(b + 1);
h = 1e-2; % finite differences of order (1,1) with Richardson extrapolation
d11 = @(h) (f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h))/(4*h^2);
fd = (4*d11(h/2) - d11(h))/3;
ok = check('tps (1,1) derivative', coef(p, [1 1]), fd, 1e-7, verbose) && ok;
K = gamma_cgf(3, 2); D = derivatives(K, 0.1, 6);
ex = arrayfun(@(k) 3*factorial(max(k-1, 0))*2^k/(1 - 0.2)^k, 1:6); ex = [-3*log(0.8) ex];
ok = check('gamma derivatives', D.value(:)', ex, 1e-13, verbose) && ok;
%% comparison with the Julia package
fn = fullfile(here, 'julia_reference.csv');
if ~isfile(fn), fprintf('julia_reference.csv not found: run julia_reference.jl\n'); ok = false; return; end
R = readtable(fn, 'TextType', 'string', 'VariableNamingRule', 'preserve');
get = @(c, q) R.value(R.("case") == c & R.quantity == q);
toep = @(r, n) r.^abs((1:n)' - (1:n));
% 1
K = CGF(@(t) -log(1 - t(1)) - log(1 - t(2)) - log(1 - t(1) - t(2)), 2, 'domain', @(t) t(1) < 1 && t(2) < 1 && t(1) + t(2) < 1);
Kn = sample_mean(K, 5);
ok = cmp('expsum2 F', tailprob(Kn, [3 3], 'order', 4).F, get("expsum2", "F"), verbose) && ok;
ok = cmp('expsum2 f', density(Kn, [3 3], 'order', 4).f, get("expsum2", "f"), verbose) && ok;
ok = cmp('expsum2 lower', tailprob(Kn, [1.5 1.8], 'order', 4, 'lower', true).F, get("expsum2", "Flow"), verbose) && ok;
ok = cmp('expsum2 mixed', tailprob(Kn, [2.5 1.5], 'order', 4, 'signs', [1 -1]).F, get("expsum2", "Fmix"), verbose) && ok;
% 2
ok = cmp('gamma F', tailprob(gamma_cgf(5), 8, 'order', 5).F, get("gamma", "F"), verbose) && ok;
ok = cmp('gamma lower', tailprob(gamma_cgf(5), 3, 'order', 5, 'lower', true).F, get("gamma", "Flow"), verbose) && ok;
ok = cmp('gamma f', density(gamma_cgf(5), 8, 'order', 5).f, get("gamma", "f"), verbose) && ok;
ok = cmp('ncx2 F', tailprob(ncx2_cgf(10, 5), 25, 'order', 5).F, get("ncx2", "F"), verbose) && ok;
ok = cmp('ncx2 f', density(ncx2_cgf(10, 5), 25, 'order', 5).f, get("ncx2", "f"), verbose) && ok;
% 3
Ku = sample_mean(uniform_cgf(), 4);
ok = cmp('uniform F', tailprob(Ku, 0.4, 'order', 4).F, get("uniform", "F"), verbose) && ok;
ok = cmp('uniform f', density(Ku, 0.4, 'order', 4).f, get("uniform", "f"), verbose) && ok;
% 4
Kc = common_component_cgf([15 20 10], [5 10 8]);
x = [48.09950493836207795336339, 65.23368793961408597955183];
ok = cmp('cc2 F', tailprob(Kc, x, 'order', 5).F, get("cc2", "F"), verbose) && ok;
ok = cmp('cc2 f', density(Kc, x, 'order', 5).f, get("cc2", "f"), verbose) && ok;
Kc3 = common_component_cgf([4 5 6 7]);
ok = cmp('cc3 F', tailprob(Kc3, [14 16 17], 'order', 3).F, get("cc3", "F"), verbose, 1e-8) && ok;
ok = cmp('cc3 f', density(Kc3, [14 16 17], 'order', 3).f, get("cc3", "f"), verbose) && ok;
% 5
n = 8;
Kq = quadratic_forms_cgf({toep(0.5, n), eye(n)}, 'Sigma', toep(0.3, n), 'mu', 0.5*ones(n, 1));
ok = cmp('qf2 F', tailprob(Kq, [14 13], 'order', 4).F, get("qf2", "F"), verbose) && ok;
ok = cmp('qf2 f', density(Kq, [14 13], 'order', 4).f, get("qf2", "f"), verbose) && ok;
D = derivatives(Kq, [0.05 -0.02], 5);
nus = [0 0; 1 0; 0 1; 2 1; 3 2; 1 4]; dv = zeros(6, 1); jv = dv;
for i = 1:6, dv(i) = D.value(ismember(D.nu, nus(i, :), 'rows')); jv(i) = get("qf2", sprintf("deriv_%d_%d", nus(i, 1), nus(i, 2))); end
ok = cmp('qf2 derivatives', dv, jv, verbose) && ok;
% the same derivatives by automatic differentiation of a generic formula
B1 = sqrtm(toep(0.3, n))*toep(0.5, n)*sqrtm(toep(0.3, n)); B2 = toep(0.3, n); nu0 = sqrtm(toep(0.3, n)) \ (0.5*ones(n, 1));
Kg = CGF(@(t) qfgeneric(t, B1, B2, nu0), 2);
Dg = derivatives(Kg, [0.05 -0.02], 5); gv = zeros(6, 1);
for i = 1:6, gv(i) = Dg.value(ismember(Dg.nu, nus(i, :), 'rows')); end
ok = cmp('qf2 derivatives by AD', gv, jv, verbose) && ok;
% 6
ok = cmp('ratio density', ratio_density(Kq, 1.2, 'order', 4).f, get("ratio_qf", "f"), verbose) && ok;
ok = cmp('ratio tail', ratio_tailprob(Kq, 1.2, 'order', 4).F, get("ratio_qf", "F"), verbose) && ok;
ok = cmp('ratio lower', ratio_tailprob(Kq, 0.9, 'order', 4, 'lower', true).F, get("ratio_qf", "Flow"), verbose) && ok;
% 7
Kw = wishart_diag_cgf([1 0.4; 0.4 2], 10);
ok = cmp('wishart F', tailprob(Kw, [12 25], 'order', 4).F, get("wishart", "F"), verbose) && ok;
Kz = normal_cgf([0; 1], [1 0.5; 0.5 2]);
ok = cmp('normal F', tailprob(Kz, [0.5 1.5], 'order', 2).F, get("normal", "F"), verbose) && ok;
% 8
A = [1 0 0; 0 1 0; 0 0 1; 1 1 0; 0 1 1];
Ke = exponential_sums_cgf(A, 'n', 6);
ok = cmp('expsum3 F', tailprob(Ke, [2.2 2.6 1.9], 'order', 3).F, get("expsum3", "F"), verbose, 1e-8) && ok;
ok = cmp('expsum3 f', density(Ke, [2.2 2.6 1.9], 'order', 3).f, get("expsum3", "f"), verbose) && ok;
if ok, fprintf('all tests passed\n'); else, fprintf('SOME TESTS FAILED\n'); end
end

function v = qfgeneric(t, B1, B2, nu)
% -1/2 log det(I - 2C) + nu'C(I - 2C)^{-1}nu by a Cholesky factorization written in plain
% arithmetic, so that it can be differentiated by the Taylor class
n = size(B1, 1); C = t(1)*B1 + t(2)*B2; Q = eye(n) - 2*C;
L = Q(1, 1)*zeros(n);
for j = 1:n
   s = Q(j, j); for k = 1:j-1, s = s - L(j, k)^2; end
   L(j, j) = sqrt(s);
   for i = j+1:n, w = Q(i, j); for k = 1:j-1, w = w - L(i, k)*L(j, k); end, L(i, j) = w/L(j, j); end
end
v = -sum(log(diag(L)));
b = C*nu; y = Q(1, 1)*zeros(n, 1);                 % solve L L' w = nu
for i = 1:n, s = nu(i); for k = 1:i-1, s = s - L(i, k)*y(k); end, y(i) = s/L(i, i); end
w = y; for i = n:-1:1, s = y(i); for k = i+1:n, s = s - L(k, i)*w(k); end, w(i) = s/L(i, i); end
v = v + sum(b.*w);
end

function ok = cmp(name, a, b, verbose, tol)
if nargin < 5, tol = 1e-10; end
a = a(:); b = b(:);
if numel(a) ~= numel(b), fprintf('%-24s size mismatch\n', name); ok = false; return; end
e = max(abs(a - b)./max(abs(b), realmin));
ok = e <= tol;
if verbose || ~ok, fprintf('%-24s max rel. diff %8.1e  %s\n', name, e, ifelse(ok, 'ok', 'FAILED')); end
end
function ok = check(name, a, b, tol, verbose), ok = cmp(name, a, b, verbose, tol); end
function s = ifelse(c, a, b), if c, s = a; else, s = b; end, end
