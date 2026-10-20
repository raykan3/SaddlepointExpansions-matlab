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
% several ratios (Theorem 4): the d-ratio code at d = 1 against Rice's expansion, and the
% inverted Dirichlet R_j = X_j/Y (X_j ~ Gamma(a_j), Y ~ Gamma(b)) against its closed form
ok = cmp('ratio d=1 general code', ratio_density(Kq, 1.2, 'order', 4, 'general', true).f, get("ratio_qf", "f"), verbose) && ok;
aa = [3 4 5]; b = 6; rr = [0.6 0.5 0.9];
idir = @(a, b, r) exp(gammaln(sum(a)+b) - sum(gammaln(a)) - gammaln(b))*prod(r.^(a-1))*(1+sum(r))^(-(sum(a)+b));
idirF = @(a, b, r) integral(@(y) prod(gammainc(r(:)*y, repmat(a(:), 1, numel(y))), 1).*y.^(b-1).*exp(-y)/gamma(b), 0, Inf, 'RelTol', 1e-13, 'AbsTol', 0);
for d = 2:3
   a = aa(1:d); r = rr(1:d);
   Kd = CGF(@(t) -sum(a(:).*log(1 - t(1:d))) - b*log(1 - t(d+1)), d+1, 'domain', @(t) all(t < 1));
   f = ratio_density(Kd, r, 'order', 5).f;
   ok = check(sprintf('%d ratios f, order 5', d), f(5), idir(a, b, r), 1e-5, verbose) && ok;
   M = 6 - d;  F = ratio_tailprob(Kd, r, 'order', M, 'lower', true).F;
   ok = check(sprintf('%d ratios F, order %d', d, M), F(M), idirF(a, b, r), 2e-5, verbose) && ok;
end
e = zeros(2, 1);                                  % rate O(n^-4) of the order-4 density
for i = 1:2
   n = 4*i; a = n*aa(1:2); bn = n*b;
   Kd = CGF(@(t) -sum(a(:).*log(1 - t(1:2))) - bn*log(1 - t(3)), 3, 'domain', @(t) all(t < 1));
   f = ratio_density(Kd, rr(1:2), 'order', 4).f; e(i) = abs(f(4)/idir(a, bn, rr(1:2)) - 1);
end
ok = check('2 ratios f, rate n^-4', min(e(1)/e(2), 16), 16, 0.25, verbose) && ok;
a = [4 5 6 7]; r = [0.9 0.5];                     % independent ratios X_j/Y_j, K of dimension 2d
K4 = CGF(@(t) -sum(a(:).*log(1 - t)), 4, 'domain', @(t) all(t < 1));
bp = @(x, p, q) x^(p-1)*(1+x)^(-p-q)/beta(p, q);
f = ratio_density(K4, r, 'order', 5).f;
ok = check('2 indep. ratios f', f(5), bp(r(1), a(1), a(3))*bp(r(2), a(2), a(4)), 1e-5, verbose) && ok;
F = ratio_tailprob(K4, r, 'order', 4, 'lower', true).F;
ok = check('2 indep. ratios F', F(4), betainc(r(1)/(1+r(1)), a(1), a(3))*betainc(r(2)/(1+r(2)), a(2), a(4)), 1e-5, verbose) && ok;
%% the shared cases of the Julia and Python packages (second part of julia_reference.csv)
sv = @(c) R.("case") == c;
dcmp = @(name, c, K, t, N) cmp(name, derivs(K, t, N, R(sv(c) & startsWith(R.quantity, "deriv_"), :)), ...
   R.value(sv(c) & startsWith(R.quantity, "deriv_")), verbose);
Aw = [1 0; 1 1; 0 1];
Kwg = exponential_sums_cgf(Aw, 'n', 10, 'shift', [-2 -2]);         % Wang's example
ok = dcmp('wang derivatives', "wang", Kwg, [0.3; -0.2], 7) && ok;
ok = cmp('wang tau', saddlepoint(Kwg, [-0.5 -0.5]), get("wang", "tau"), verbose) && ok;
r = tailprob(Kwg, [-0.5 -0.5], 'order', 5, 'lower', true);
ok = cmp('wang Flow', [r.F; r.tau(:)], [get("wang", "Flow"); get("wang", "Flow_tau")], verbose) && ok;
r = tailprob(Kwg, [-1 2], 'order', 5, 'lower', true);
ok = cmp('wang Flow2', [r.F; r.tau(:)], [get("wang", "Flow2"); get("wang", "Flow2_tau")], verbose) && ok;
r = tailprob(Kwg, [-0.8 0.5], 'order', 3);
ok = cmp('wang Fie', [r.F; r.tau(:)], [get("wang", "Fie"); get("wang", "Fie_tau")], verbose) && ok;
ok = cmp('wang f', density(Kwg, [-0.5 -0.5], 'order', 4).f, get("wang", "f"), verbose) && ok;
K5 = exponential_sums_cgf(Aw, 'n', 5);                              % Kolassa and Li
pts = {"F33", [3 3], 5; "Fbdry", [2.5 3], 5; "Ffar", [6 7], 4};
for i = 1:3
   r = tailprob(K5, pts{i, 2}, 'order', pts{i, 3});
   ok = cmp(sprintf('kl %s', pts{i, 1}), [r.F; r.tau(:)], [get("kl", pts{i, 1}); get("kl", pts{i, 1} + "_tau")], verbose) && ok;
end
mm = [15 20 10]; ll = [5 10 8]; Kcc = common_component_cgf(mm, ll);  % pairs of quadratic forms
ok = cmp('bqf F', tailprob(Kcc, [55 70], 'order', 4).F, get("bqf", "F"), verbose) && ok;
ok = cmp('bqf f', density(Kcc, [55 70], 'order', 4).f, get("bqf", "f"), verbose) && ok;
ok = cmp('bqf Flow', tailprob(Kcc, [40 60], 'order', 3, 'lower', true).F, get("bqf", "Flow"), verbose) && ok;
[Fx, fx] = bqfcc_exact(55, 70, mm, ll); [~, ~, Fl] = bqfcc_exact(40, 60, mm, ll);
ok = cmp('bqf exact', [Fx; fx; Fl], [get("bqf", "F_exact"); get("bqf", "f_exact"); get("bqf", "Flow_exact")], verbose, 1e-8) && ok;
m3 = [15 20 25 10]; l3 = [5 10 0 8]; K3 = common_component_cgf(m3, l3);   % trivariate
ok = cmp('tri F', tailprob(K3, [80 95 100], 'order', 4).F, get("tri", "F"), verbose, 1e-8) && ok;
ok = cmp('tri f', density(K3, [80 95 100], 'order', 4).f, get("tri", "f"), verbose) && ok;
r = tailprob(K3, [55 70 75], 'order', 3);
ok = cmp('tri Fie', [r.F; r.tau(:)], [get("tri", "Fie"); get("tri", "Fie_tau")], verbose, 1e-8) && ok;
Kwi = wishart_diag_cgf(0.75*eye(3) + 0.25, 5);                       % Wishart diagonal
ok = dcmp('wishart3 derivatives', "wishart3", Kwi, [0.05; -0.02; 0.03], 6) && ok;
ok = cmp('wishart3 F', tailprob(Kwi, [6.5 6.5 7.5], 'order', 3).F, get("wishart3", "F"), verbose, 1e-8) && ok;
K4 = common_component_cgf([15 20 25 30 10], [5 10 0 3 8]); x4 = [70 85 95 105];   % d = 4
ok = cmp('d4 f', density(K4, x4, 'order', 3).f, get("d4", "f"), verbose) && ok;
r = tailprob(K4, x4, 'order', 3);                                     % quasi-Monte Carlo: within the errors
e = abs(r.F - get("d4", "F"))./(3*(abs(r.info.err(:)) + abs(get("d4", "F_err"))));
if verbose || max(e) > 1, fprintf('%-24s max |diff|/(3 s.e.) %5.2f  %s\n', 'd4 F (quasi-MC)', max(e), ifelse(max(e) <= 1, 'ok', 'FAILED')); end
ok = max(e) <= 1 && ok;
Kg = gamma_cgf(3);                                                    % univariate
r = tailprob(Kg, 6, 'order', 4);
ok = cmp('gamma3 F', [r.F; r.tau], [get("gamma3", "F"); get("gamma3", "tau")], verbose) && ok;
ok = cmp('gamma3 Flow', tailprob(Kg, 6, 'order', 4, 'lower', true).F, get("gamma3", "Flow"), verbose) && ok;
ok = cmp('gamma3 f', density(Kg, 6, 'order', 4).f, get("gamma3", "f"), verbose) && ok;
ok = cmp('gamma3 Ffar', tailprob(Kg, 20, 'order', 5).F, get("gamma3", "Ffar"), verbose) && ok;
Kx = ncx2_cgf(5, 10);
ok = cmp('ncx2b F', tailprob(Kx, 30, 'order', 6).F, get("ncx2b", "F"), verbose) && ok;
ok = cmp('ncx2b f', density(Kx, 30, 'order', 6).f, get("ncx2b", "f"), verbose) && ok;
Ku10 = sample_mean(uniform_cgf(), 10);                                % uniform
ok = cmp('uniform10 F', tailprob(Ku10, 0.3, 'order', 4).F, get("uniform10", "F"), verbose) && ok;
ok = cmp('uniform10 f', density(Ku10, 0.3, 'order', 4).f, get("uniform10", "f"), verbose) && ok;
ok = dcmp('uniform derivatives', "uniform10", uniform_cgf(), 0, 6) && ok;
Kb = CGF(@(t) log(sinhc(t(1))) + log(sinhc(t(1) + t(2))) + log(sinhc(t(2))), 2);
r = tailprob(sample_mean(Kb, 8), [0.4 0.3], 'order', 3);
ok = cmp('uniform bivariate', [r.F; r.tau(:)], [get("uniform10", "Fbi"); get("uniform10", "Fbi_tau")], verbose) && ok;
m_ = 4; n_ = 12; lam = 3; c = 2.5*m_/n_;                               % noncentral F as a ratio
K1 = ncx2_cgf(m_, lam); K2 = ncx2_cgf(n_);
KF = CGF(@(t) K1.fun(t(1)) + K2.fun(t(2)), 2, 'domain', @(t) t(1) < 0.5 && t(2) < 0.5);
q = ratio_density(KF, c, 'order', 4);
ok = cmp('ncf f', [q.f; q.shat], [get("ncf", "f"); get("ncf", "shat")], verbose) && ok;
ok = cmp('ncf Flow', ratio_tailprob(KF, c, 'order', 4, 'lower', true).F, get("ncf", "Flow"), verbose) && ok;
ok = cmp('ncf F', ratio_tailprob(KF, c, 'order', 4).F, get("ncf", "F"), verbose) && ok;
S = [1 0.6; 0.6 2];                                                   % mixed corner
ok = cmp('normal mixed', tailprob(normal_cgf([0; 0], S), [0.5 -0.3], 'order', 2, 'signs', [1 -1]).F, get("normal_mixed", "F"), verbose) && ok;
a = [3 4 5 6];                                                        % several ratios
Kg4 = CGF(@(t) -sum(a(:).*log(1 - t)), 4, 'domain', @(t) all(t < 1));
ok = cmp('ratio2 f', ratio_density(Kg4, [0.7 0.9], 'order', 5).f, get("ratio2", "f"), verbose) && ok;
ok = cmp('ratio2 Flow', ratio_tailprob(Kg4, [0.7 0.9], 'order', 5, 'lower', true).F, get("ratio2", "Flow"), verbose) && ok;
ok = cmp('ratio2 F', ratio_tailprob(Kg4, [0.7 0.9], 'order', 5).F, get("ratio2", "F"), verbose) && ok;
Kg3 = CGF(@(t) -3*log(1 - t(1)) - 4*log(1 - t(2)) - 5*log(1 - t(3)), 3, 'domain', @(t) all(t < 1));
ok = cmp('ratio2 common', ratio_density(Kg3, [0.5 0.8], 'order', 4).f, get("ratio2", "f_common"), verbose) && ok;
ok = cmp('ratio2 general d=1', ratio_density(Kq, 1.2, 'order', 4, 'general', true).f, get("ratio2", "f_qf_general"), verbose) && ok;
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

function v = derivs(K, t, N, Rd)
% the derivatives of K at t named in the rows Rd ("deriv_i_j...") of the reference table
D = derivatives(K, t, N); v = zeros(height(Rd), 1);
for i = 1:height(Rd)
   nu = sscanf(strrep(extractAfter(Rd.quantity(i), "deriv_"), "_", " "), '%d')';
   v(i) = D.value(all(D.nu == nu, 2));
end
end

function ok = cmp(name, a, b, verbose, tol)
if nargin < 5, tol = 1e-10; end
a = a(:); b = b(:);
if numel(a) ~= numel(b), fprintf('%-24s size mismatch\n', name); ok = false; return; end
e = max(abs(a - b)./max(abs(b), max(1e-12*max(abs(b)), realmin)));   % relative, with a floor for components near zero
ok = e <= tol;
if verbose || ~ok, fprintf('%-24s max rel. diff %8.1e  %s\n', name, e, ifelse(ok, 'ok', 'FAILED')); end
end
function ok = check(name, a, b, tol, verbose), ok = cmp(name, a, b, verbose, tol); end
function s = ifelse(c, a, b), if c, s = a; else, s = b; end, end
