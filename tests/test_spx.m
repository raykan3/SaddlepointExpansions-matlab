function ok = test_spx(verbose)
%TEST_SPX Tests of the MATLAB package against SaddlepointExpansions.jl (julia_reference.csv, made by
%   julia_reference.jl) and against closed forms.  Returns true if every check passes; prints the
%   largest relative difference of each case.
if nargin < 1, verbose = true; end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..'), fullfile(here, '..', 'engine'), here);
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
%% Studentized means Q = X/sqrt(Y) (Bao, Broda and Kan): the tests of the Julia and Python packages
ok = test_studentized(verbose) && ok;
%% double-double arithmetic (dd, tpsdd) and the double-double Studentized expansions
ok = test_dd(verbose, here) && ok;
%% conditional expansions, lattice sums, quantiles and normalized densities (version 0.3.0):
%% the third part of julia_reference.csv and conditional_reference.csv (julia_conditional_reference.jl)
ok = test_conditional_reference(verbose, here, R) && ok;
%% the tables of Broda and Kan, "Conditional and lattice saddlepoint expansions of arbitrary order"
%% (conditional_rows.txt, as in the Julia and Python tests) recomputed against exact values
ok = test_conditional_tables(verbose, here) && ok;
%% conditional expansions: B empty, Gaussian vectors, the Dirichlet distribution, the derivative
%% interface, the leading term, the arguments
ok = test_conditional(verbose) && ok;
%% lattice variables, quantiles and normalized densities against exact values
ok = test_lattice_quantiles(verbose) && ok;
if ok, fprintf('all tests passed\n'); else, fprintf('SOME TESTS FAILED\n'); end
end

function ok = test_studentized(verbose)
ok = true;
% noncentral t: X ~ N(mu, 1), Y ~ chi2_nu/nu independent, n pairs; Xbar/sqrt(Ybar) = T/sqrt(n)
mu = 1; nu = 3; n = 16;
K1 = CGF(@(t) mu*t(1) + t(1)^2/2 - nu/2*log(1 - 2*t(2)/nu), 2, 'domain', @(t) t(2) < nu/2);
Kn = sample_mean(K1, n);
k = n*nu/2; th = 2/(n*nu);
logg = @(y) (k - 1)*log(y) - y/th - gammaln(k) - k*log(th);
exactF = @(q) integral(@(y) exp(logg(y)).*normcdf(sqrt(n)*(q*sqrt(y) - mu)), 0, Inf, 'RelTol', 1e-12, 'AbsTol', 0);
exactf = @(q) integral(@(y) exp(logg(y)).*sqrt(n*y).*exp(-n*(q*sqrt(y) - mu).^2/2)/sqrt(2*pi), 0, Inf, 'RelTol', 1e-12, 'AbsTol', 0);
for q = [0.5 1.5 2.5]
   sd = studentized_saddlepoint(Kn, q);
   ok = check(sprintf('stud. saddle q=%.1f', q), double(sd.v > 0 && (q > sd.q0) == (sd.s > 0)), 1, 0, verbose) && ok;
   r = studentized_tailprob(Kn, q, 'order', 3, 'lower', true); ex = exactF(q);
   ok = check(sprintf('stud. F q=%.1f', q), r.F(end), ex, 2e-9/ex, verbose) && strcmp(r.status, 'ok') && ok;
   ok = check(sprintf('stud. F gain q=%.1f', q), double(abs(r.info.Fp(end) - ex) < abs(r.info.Fp(1) - ex)/100), 1, 0, verbose) && ok;
   ru = studentized_tailprob(Kn, q, 'order', 3);
   ok = check(sprintf('stud. upper q=%.1f', q), ru.F(end), 1 - r.F(end), 1e-12, verbose) && ok;
   d = studentized_density(Kn, q, 'order', 3); fex = exactf(q);
   ok = check(sprintf('stud. f order 1 q=%.1f', q), d.f(1), fex, 1e-2, verbose) && ok;
   ok = check(sprintf('stud. f order 3 q=%.1f', q), d.f(3), fex, 1e-6, verbose) && ok;
end
Q = studentized_quantile(Kn, 0.05, 'order', 3);
ok = check('stud. quantile', studentized_tailprob(Kn, Q.q, 'order', 3, 'lower', true).F(end), 0.05, 1e-9/0.05, verbose) && ok;
ok = check('stud. quantile exact', exactF(Q.q), 0.05, 1e-8/0.05, verbose) && ok;
sd = studentized_saddlepoint(Kn, 1.5); loc = studentized_locus(Kn, sd.s, 0.9*sd.t);
ok = check('stud. locus', [loc.t loc.v loc.q], [sd.t sd.v 1.5], 1e-12, verbose) && ok;
% the Dickey-Fuller t-ratio: the reference values of the paper (the Julia and Python programs agree
% on them to 1e-10) at the 5% asymptotic critical values (c = 0): distribution function with the
% terms up to n^(-j/2), j = 0..5, and density with M = 1..4 terms
ref = {{-1.941, [0.04131561, 0.04932910, 0.05021031, 0.05004336, 0.04996497, 0.04997437], [1.137468e-01, 1.145105e-01, 1.142462e-01, 1.143557e-01]}, ...
       {-2.862, [0.04748127, 0.05196366, 0.04837446, 0.04492099, 0.04534934, 0.04962684], [1.288676e-01, 1.235403e-01, 1.262806e-01, 1.274765e-01]}, ...
       {-3.411, [0.04466519, 0.04979224, 0.04990885, 0.04967211, 0.04978553, 0.04999147], [1.307302e-01, 1.305121e-01, 1.304833e-01, 1.306511e-01]}};
for m = 1:3
   q = ref{m}{1}; K = dickey_fuller_cgf(m);
   r = studentized_tailprob(K, q, 'order', 3, 'lower', true);
   ok = check(sprintf('Dickey-Fuller M%d F', m), r.info.Fp, ref{m}{2}, 1e-8/0.04, verbose) && ok;
   d = studentized_density(K, q, 'order', 4, 'refine', false);
   ok = check(sprintf('Dickey-Fuller M%d f', m), d.f(1:3), ref{m}{3}(1:3), 2e-6, verbose) && ok;
   ok = check(sprintf('Dickey-Fuller M%d f(4)', m), d.f(4), ref{m}{3}(4), 1e-3, verbose) && ok;
   % the cancellation indicator reports where double precision runs out (M2 at order 4; without refinement)
   ok = check(sprintf('Dickey-Fuller M%d status', m), double(strcmp(d.status, 'coefficients inaccurate')), double(m == 2), 0, verbose) && ok;
end
K = dickey_fuller_cgf(3, 'c', -5);                     % the 0.75 quantile of M3 under c = -5
r = studentized_tailprob(K, -2.025180818400757, 'order', 3, 'lower', true);
ok = check('Dickey-Fuller c=-5 saddle', [r.saddle.s r.saddle.t], [2.391156286 10.320589682], 1e-8, verbose) && ok;
ok = check('Dickey-Fuller c=-5 F', r.info.Fp(end), 0.739908646, 1e-8/0.74, verbose) && ok;
Q = studentized_quantile(dickey_fuller_cgf(2), 0.05, 'order', 3);
ok = check('Dickey-Fuller M2 quantile', Q.q, -2.859305, 1e-5/2.86, verbose) && ok;
ok = check('Dickey-Fuller window', eval(dickey_fuller_cgf(2, 'c', -5), [0.3 11]), eval(dickey_fuller_cgf(2, 'c', -5, 'window', 1e-3), [0.3 11]), 1e-10, verbose) && ok;
end

function ok = test_dd(verbose, here)
ok = true;
rel = @(a, X) double(max(abs((vpa(a, 40) - X)./X)));      % a: dd, X: sym
% the arithmetic and the elementary functions against vpa
A = sym(1)/3; B = sqrt(sym(2)); a = dd(A); b = dd(B);
ok = echeck('dd arithmetic', rel([a + b, a - b, a*b, a/b, sqrt(b), b^5], vpa([A + B, A - B, A*B, A/B, sqrt(B), B^5], 40)), 1e-30, verbose) && ok;
ok = echeck('dd functions', rel([exp(a), log(b), erfc(b)], vpa([exp(A), log(B), erfc(B)], 40)), 1e-30, verbose) && ok;
x = dd((1:1000)')/7;
ok = echeck('dd sum', rel(sum(x), vpa(sym(500500)/7, 40)), 1e-30, verbose) && ok;
% derivatives of the double-double Taylor class against symbolic differentiation
syms y1 y2
v = tpsdd.vars(2, 6);
p = exp(v(1) + 2*v(2)).*log(1 + v(1)) + sqrt(4 + v(2)).*atan(0.25 + v(1).*v(2)) + sinh(v(1))./cosh(v(2) + 1);
g = exp(y1 + 2*y2)*log(1 + y1) + sqrt(4 + y2)*atan(sym(1)/4 + y1*y2) + sinh(y1)/cosh(y2 + 1);
nus = [0 0; 1 1; 3 2; 0 6; 2 4; 5 1];
e = 0;
for i = 1:size(nus, 1)
   ex = subs(diff(diff(g, y1, nus(i, 1)), y2, nus(i, 2)), [y1 y2], [0 0]);
   e = max(e, rel(coef(p, nus(i, :)), vpa(ex, 40)));
end
ok = echeck('tpsdd derivatives', e, 1e-28, verbose) && ok;
% the Studentized expansions against SaddlepointExpansions.jl in BigFloat (256 bits),
% studentized_reference.csv made by julia_studentized_reference.jl
fn = fullfile(here, 'studentized_reference.csv');
if ~isfile(fn), fprintf('studentized_reference.csv not found: run julia_studentized_reference.jl\n'); ok = false; return; end
R = readtable(fn, 'TextType', 'string', 'VariableNamingRule', 'preserve');
get = @(c, q) R.value(R.("case") == c & R.quantity == q);
tol = 3e-16;                                        % about one unit in the last place
mu = 1; nu = 3; n = 16;
K1 = CGF(@(t) mu*t(1) + t(1)^2/2 - nu/2*log(1 - 2*t(2)/nu), 2, 'domain', @(t) t(2) < nu/2);
Kn = sample_mean(K1, n);
d = studentized_density(Kn, 0.5, 'order', 6, 'precision', 'dd');
ok = check('dd stud. f nct q=0.5', d.f, get("nct_0.5", "f"), tol, verbose) && strcmp(d.status, 'ok') && ok;
r = studentized_tailprob(Kn, 0.5, 'order', 4, 'lower', true, 'precision', 'dd');
ok = check('dd stud. Fp nct q=0.5', r.info.Fp, get("nct_0.5", "Fp"), tol, verbose) && ok;
% the default refinement: the Dickey-Fuller M2 density at order 5 (kappa about 1e17 in double)
d = studentized_density(dickey_fuller_cgf(2), -2.862, 'order', 5);
fr = get("df2", "f");
ok = check('dd stud. f DF M2 refined', d.f, fr(1:5), 1e-14, verbose) && strcmp(d.status, 'ok') ...
   && strcmp(d.info.refined, 'double-double') && ok;
r = studentized_tailprob(dickey_fuller_cgf(3, 'c', -5), -2.025180818400757, 'order', 3, 'lower', true, 'precision', 'dd');
ok = check('dd stud. Fp DF c=-5', r.info.Fp, get("df3_c-5", "Fp"), tol, verbose) && ok;
end

function ok = test_conditional_reference(verbose, here, R)
ok = true;
get = @(c, q) R.value(R.("case") == c & R.quantity == q);
% the gamma sum given its total: d = 2 with a positive and a negative saddlepoint component, d = 3
Kb2 = CGF(@(t) -16*log(1 - t(1) - t(2)) - 16*log(1 - t(2)), 2, 'domain', @(t) t(1) + t(2) < 1 && t(2) < 1);
nm = ["pos", "neg"]; xs = {[19 32], [12 32]};
for i = 1:2
   r = conditional_tailprob(Kb2, xs{i}, 1, 'order', 5);
   ok = cmp(sprintf('cond beta2 F %s', nm(i)), [r.F; r.info.Fp; r.tau(:); r.taut(:)], [get("cond_beta2", "F_" + nm(i)); ...
      get("cond_beta2", "Fp_" + nm(i)); get("cond_beta2", "tau_" + nm(i)); get("cond_beta2", "taut_" + nm(i))], verbose) && ok;
   ok = cmp(sprintf('cond beta2 f %s', nm(i)), conditional_density(Kb2, xs{i}, 1, 'order', 5).f, get("cond_beta2", "f_" + nm(i)), verbose) && ok;
end
ok = cmp('cond beta2 lower', conditional_tailprob(Kb2, [19 32], 1, 'order', 5, 'lower', true).F, get("cond_beta2", "Flow_pos"), verbose) && ok;
Kb3 = CGF(@(t) -16*log(1 - t(1) - t(2)) - 32*log(1 - t(2) - t(3)) - 8*log(1 - t(2)), 3, ...
   'domain', @(t) t(1) + t(2) < 1 && t(2) + t(3) < 1 && t(2) < 1);
ok = cmp('cond beta3 F', conditional_tailprob(Kb3, [18 56 32], 1, 'order', 4).F, get("cond_beta3", "F"), verbose) && ok;
ok = cmp('cond beta3 f', conditional_density(Kb3, [18 56 32], 1, 'order', 4).f, get("cond_beta3", "f"), verbose) && ok;
% a quadratic form given another
Kq2 = common_component_cgf([15 20 10], [5 10 8]);
ok = cmp('cond bqf F', conditional_tailprob(Kq2, [55 50], 1, 'order', 5).F, get("cond_bqf", "F"), verbose) && ok;
ok = cmp('cond bqf f', conditional_density(Kq2, [55 50], 1, 'order', 5).f, get("cond_bqf", "f"), verbose) && ok;
% two tail components given the total (Dirichlet): a mixed-sign saddlepoint, a mixed corner
Kdir = CGF(@(t) -5*log(1 - t(1) - t(3)) - 7.5*log(1 - t(2) - t(3)) - 10*log(1 - t(3)), 3, ...
   'domain', @(t) t(1) + t(3) < 1 && t(2) + t(3) < 1 && t(3) < 1);
ok = cmp('cond dir F', conditional_tailprob(Kdir, [0.35 0.45 1]*22.5, [1 2], 'order', 3).F, get("cond_dir", "F"), verbose) && ok;
ok = cmp('cond dir mixed', conditional_tailprob(Kdir, [0.30 0.25 1]*22.5, [1 2], 'order', 3).F, get("cond_dir", "F_mixed"), verbose) && ok;
ok = cmp('cond dir signs', conditional_tailprob(Kdir, [0.30 0.25 1]*22.5, [1 2], 'order', 3, 'signs', [1 -1]).F, get("cond_dir", "F_signs"), verbose) && ok;
ok = cmp('cond dir f', conditional_density(Kdir, [0.35 0.45 1]*22.5, [1 2], 'order', 3).f, get("cond_dir", "f"), verbose) && ok;
% logistic regression, lattice, both expansions of the amplitude
Klg = logistic_regression_cgf([ones(30, 1) mod((0:29)', 5) mod((0:29)', 3)]);
ok = cmp('cond logit saddle', conditional_tailprob(Klg, [15 37 15], 2, 'order', 4, 'lattice', true).F, get("cond_logit", "F_saddle"), verbose) && ok;
ok = cmp('cond logit origin', conditional_tailprob(Klg, [15 37 15], 2, 'order', 4, 'lattice', true, 'sinh', 'origin').F, get("cond_logit", "F_origin"), verbose) && ok;
ok = cmp('cond logit neg', conditional_tailprob(Klg, [10 14 12], 2, 'order', 4, 'lattice', true).F, get("cond_logit", "F_neg"), verbose) && ok;
ok = cmp('cond logit pmf', conditional_density(Klg, [15 37 15], 2, 'order', 4, 'lattice', true).f, get("cond_logit", "pmf"), verbose) && ok;
% a sum of two ratios of quadratic forms (d = 3)
l1 = [-0.6 -0.1 0.3 0.7 0.9]; l2 = [-0.8 -0.2 0.0 0.4 0.5 0.8];
Ksr = CGF(@(t) -(sum(log(1 - 2*(t(1)*l1 + t(2)))) + sum(log(1 - 2*(t(1)*l2 + t(3)))))/2, 3);
ok = cmp('cond ratios F', conditional_tailprob(Ksr, [0.5 1 1], 1, 'order', 4).F, get("cond_ratios", "F"), verbose) && ok;
% lattice sums without conditioning
ok = cmp('lattice pois F', tailprob(poisson_cgf(5), 11, 'order', 4, 'lattice', true).F, get("lattice", "pois_F"), verbose) && ok;
ok = cmp('lattice pois Fneg', tailprob(poisson_cgf(5), 3, 'order', 4, 'lattice', true).F, get("lattice", "pois_Fneg"), verbose) && ok;
ok = cmp('lattice pois f', density(poisson_cgf(5), 11, 'order', 4, 'lattice', true).f, get("lattice", "pois_f"), verbose) && ok;
ok = cmp('lattice binom F', tailprob(binomial_cgf(40, 0.3), 18, 'order', 4, 'lattice', true).F, get("lattice", "binom_F"), verbose) && ok;
ok = cmp('lattice binom lower', tailprob(binomial_cgf(40, 0.3), 9, 'order', 4, 'lattice', true, 'lower', true).F, get("lattice", "binom_Flow"), verbose) && ok;
ok = cmp('lattice negbin F', tailprob(negative_binomial_cgf(5, 0.4), 15, 'order', 4, 'lattice', true).F, get("lattice", "negbin_F"), verbose) && ok;
Kbp = CGF(@(t) 2*(exp(t(1)) - 1) + 2*(exp(t(2)) - 1) + (exp(t(1) + t(2)) - 1), 2);
ok = cmp('lattice bipois F', tailprob(Kbp, [6 2], 'order', 4, 'lattice', true).F, get("lattice", "bipois_F"), verbose) && ok;
ok = cmp('lattice bipois f', density(Kbp, [6 2], 'order', 4, 'lattice', true).f, get("lattice", "bipois_f"), verbose) && ok;
% quantiles and normalized densities
q = quantile(gamma_cgf(3), 0.05, 'order', 4);
ok = cmp('quantile gamma', [q.q; q.tau], [get("quantile", "gamma_q"); get("quantile", "gamma_tau")], verbose) && ok;
ok = cmp('quantile gamma upper', quantile(gamma_cgf(3), 1e-6, 'order', 4, 'lower', false).q, get("quantile", "gamma_q_upper"), verbose) && ok;
K1 = ncx2_cgf(4, 3); K2 = ncx2_cgf(12);
KF = CGF(@(t) K1.fun(t(1)) + K2.fun(t(2)), 2, 'domain', @(t) t(1) < 0.5 && t(2) < 0.5);
q = ratio_quantile(KF, 0.99, 'order', 4);
ok = cmp('ratio quantile ncf', [q.r; q.shat], [get("quantile", "ncf_r"); get("quantile", "ncf_shat")], verbose) && ok;
ok = cmp('normalize gamma', density(gamma_cgf(3), 2, 'order', 4, 'normalize', true).info.mass, get("normalize", "gamma_mass"), verbose) && ok;
ok = cmp('normalize ncx2', density(ncx2_cgf(4, 3), 7, 'order', 3, 'normalize', true).info.mass, get("normalize", "ncx2_mass"), verbose) && ok;
ok = cmp('normalize ratio', ratio_density(KF, 1, 'order', 3, 'normalize', true).info.mass, get("normalize", "ratio_mass"), verbose) && ok;
% the further cases of conditional_reference.csv
fn = fullfile(here, 'conditional_reference.csv');
if ~isfile(fn), fprintf('conditional_reference.csv not found: run julia_conditional_reference.jl\n'); ok = false; return; end
C = readtable(fn, 'TextType', 'string', 'VariableNamingRule', 'preserve');
get = @(c, q) C.value(C.("case") == c & C.quantity == q);
Kl = linear_combination(CGF(@(t) -2*log(1 - t(1)) - 3*log(1 - t(2)), 2, 'domain', @(t) t(1) < 1 && t(2) < 1), [1 0.5; 0.3 1]);
r = density(Kl, [1 2], 'order', 2, 'normalize', true);
ok = cmp('normalize 2-d', [r.info.mass; r.f], [get("norm2", "lc_mass"); get("norm2", "lc_f")], verbose) && ok;
mu = 1; nu = 3; n = 16;
Kn = sample_mean(CGF(@(t) mu*t(1) + t(1)^2/2 - nu/2*log(1 - 2*t(2)/nu), 2, 'domain', @(t) t(2) < nu/2), n);
r = studentized_density(Kn, 1.5, 'order', 2, 'normalize', true);
ok = cmp('normalize studentized', [r.info.mass; r.f], [get("normstud", "mass"); get("normstud", "f")], verbose) && ok;
ok = cmp('normalize gamma(0.5)', density(gamma_cgf(0.5), 2, 'order', 4, 'normalize', true).info.mass, get("norm1", "gamma05_mass"), verbose) && ok;
ok = cmp('normalize normal', density(normal_cgf(1, 2), 0.3, 'order', 3, 'normalize', true).info.mass, get("norm1", "normal_mass"), verbose) && ok;
Sig = [2.0 0.6 0.5 0.3; 0.6 1.5 -0.4 0.2; 0.5 -0.4 1.0 0.1; 0.3 0.2 0.1 0.8]; m4 = [0.5; -1.0; 0.2; 0.3];
Kg = normal_cgf(m4, Sig);
xs = {[1.0 0.0 0.5 0.0], [-1.0 0.0 0.5 0.0], [1.5 0.0 0.5 0.3], [0.2 -2.5 0.5 0.3], [-1.0 -2.5 0.5 0.3], [1.2 -0.8 0.9 0.3]};
As = {1, 1, [1 2], [1 2], [2 1], [1 2 3]};
for i = 1:6
   r = conditional_tailprob(Kg, xs{i}, As{i}, 'order', 2 + (numel(As{i}) < 3));
   f = conditional_density(Kg, xs{i}, As{i}, 'order', 3);
   ok = cmp(sprintf('cond gauss %d (|A| = %d)', i, numel(As{i})), [r.F; r.tau(:); f.f], [get("gauss", "F" + i); get("gauss", "tau" + i); get("gauss", "f" + i)], verbose) && ok;
end
ok = cmp('cond gauss signs', conditional_tailprob(Kg, xs{4}, [1 2], 'order', 2, 'signs', [-1 1]).F, get("gauss", "Fsigns"), verbose) && ok;
ok = cmp('cond gauss lower', conditional_tailprob(Kg, xs{4}, [1 2], 'order', 2, 'lower', true).F, get("gauss", "Flower"), verbose) && ok;
[P, f, info] = spa_cond(@kf3, [19 56 32], 1, 4);                % closed-form derivatives, substitution
ok = cmp('spa_cond derivatives', [P; f; info.Pp; info.tau(:); info.taut(:)], [get("kf3", "P"); get("kf3", "f"); get("kf3", "Pp"); get("kf3", "tau"); get("kf3", "taut")], verbose) && ok;
w = [0.35 0.45; 0.15 0.40; 0.30 0.25; 0.15 0.25];
for i = 1:4
   r = conditional_tailprob(Kdir, [w(i, :) 1]*22.5, [1 2], 'order', 4);
   f = conditional_density(Kdir, [w(i, :) 1]*22.5, [1 2], 'order', 4);
   ok = cmp(sprintf('cond dirichlet %d', i), [r.F; f.f], [get("dir", "F" + i); get("dir", "f" + i)], verbose) && ok;
end
K1 = poisson_cgf(5);
ok = cmp('lattice pois origin', tailprob(K1, 7, 'order', 3, 'lattice', true, 'sinh', 'origin').F, get("lat", "pois7_origin"), verbose) && ok;
ok = cmp('lattice pois saddle', tailprob(K1, 7, 'order', 3, 'lattice', true).F, get("lat", "pois7_saddle"), verbose) && ok;
ok = cmp('lattice pois k = 1', tailprob(K1, 1, 'order', 4, 'lattice', true).F, get("lat", "pois1"), verbose) && ok;
for k = [6 18 25]
   ok = cmp(sprintf('lattice binom k = %d', k), [tailprob(binomial_cgf(40, 0.3), k, 'order', 4, 'lattice', true).F; density(binomial_cgf(40, 0.3), k, 'order', 4, 'lattice', true).f], ...
      [get("lat", "binom_F" + k); get("lat", "binom_f" + k)], verbose) && ok;
end
for k = [2 7 20]
   ok = cmp(sprintf('lattice negbin k = %d', k), [tailprob(negative_binomial_cgf(5, 0.4), k, 'order', 4, 'lattice', true).F; density(negative_binomial_cgf(5, 0.4), k, 'order', 4, 'lattice', true).f], ...
      [get("lat", "negbin_F" + k); get("lat", "negbin_f" + k)], verbose) && ok;
end
ok = cmp('lattice bipois lower', tailprob(Kbp, [2 1], 'order', 3, 'lattice', true, 'lower', true).F, get("lat", "bipois_Flow"), verbose) && ok;
ok = cmp('lattice bipois mixed', tailprob(Kbp, [6 1], 'order', 3, 'lattice', true, 'signs', [1 -1]).F, get("lat", "bipois_Fmix"), verbose) && ok;
Kga = gamma_cgf(3); e = 0;
for p = [1e-6 0.05 0.5 0.95]
   for lower = [true false]
      q = quantile(Kga, p, 'order', 4, 'lower', lower);
      ref = get("quant", sprintf('gamma_%s_%s', jlnum(p), lower2str(lower)));
      e = max(e, max(abs([q.q; q.tau] - ref)./abs(ref)));
   end
end
ok = echeck('quantile gamma, 8 cases', e, 1e-10, verbose) && ok;
ok = cmp('quantile normal', quantile(normal_cgf(1, 4), 0.025, 'order', 2).q, get("quant", "normal"), verbose) && ok;
for p = [0.01 0.5]
   q = ratio_quantile(KF, p, 'order', 4);
   ok = cmp(sprintf('ratio quantile p = %g', p), [q.r; q.shat], get("quant", "ncf_" + jlnum(p)), verbose) && ok;
end
q = ratio_quantile(KF, 1e-4, 'order', 4, 'lower', false);
ok = cmp('ratio quantile upper', [q.r; q.shat], get("quant", "ncf_upper"), verbose) && ok;
% against the Julia package in BigFloat (256 bits): the double-precision values to the accuracy
% that the Julia tests require of its Float64 values (1e-12 absolute for the gamma sum, 1e-13
% relative otherwise)
x = get("big", "beta_x");
Kbb = CGF(@(t) -4*log(1 - t(1) - t(2)) - 8*log(1 - t(2) - t(3)) - 2*log(1 - t(2)), 3, 'domain', @(t) t(1) + t(2) < 1 && t(2) + t(3) < 1 && t(2) < 1);
r = conditional_tailprob(Kbb, x, 1, 'order', 5);
ok = echeck('cond beta3 vs BigFloat', max(abs(r.F - get("big", "beta_F"))), 1e-12, verbose) && ok;
ok = cmp('cond beta3 f vs BigFloat', conditional_density(Kbb, x, 1, 'order', 5).f, get("big", "beta_f"), verbose, 1e-12) && ok;
ok = cmp('cond dir vs BigFloat', conditional_tailprob(Kdir, [0.30 0.45 1]*22.5, [1 2], 'order', 3).F, get("big", "dir_F"), verbose, 1e-13) && ok;
ok = cmp('cond logit vs BigFloat', conditional_tailprob(Klg, [15 37 15], 2, 'order', 3, 'lattice', true).F, get("big", "logit_F"), verbose, 1e-13) && ok;
ok = echeck('lattice pois vs BigFloat', max(abs(tailprob(K1, 11, 'order', 4, 'lattice', true).F - get("big", "pois_F"))), 1e-15, verbose) && ok;
end

function ok = test_conditional_tables(verbose, here)
% the printed rows of the tables (two significant digits) of the conditional paper, transcribed in
% conditional_rows.txt (shared with the Julia and Python tests): every cell must be reproduced
ok = true;
rows = readrows(fullfile(here, 'conditional_rows.txt'));
bpdf = @(w, p, q) exp((p - 1)*log(w) + (q - 1)*log1p(-w) + gammaln(p + q) - gammaln(p) - gammaln(q));
% the gamma sum given its total (cond_beta_rows.tex): exact beta
nok = 0; ncell = 0; good = true;
for i = find(rows.table == 1)'
   cells = rows.cells{i}; N = str2double(cells{1}); q = str2double(cells{2});
   if rows.block(i) <= 2
      if rows.block(i) == 1, a = 1; b = 1; else, a = 0.5; b = 3; end
      K = CGF(@(t) -N*a*log(1 - t(1) - t(2)) - N*b*log(1 - t(2)), 2, 'domain', @(t) t(1) + t(2) < 1 && t(2) < 1);
      s2 = N*(a + b); w = betaincinv(q, N*a, N*b); x = [w*s2 s2];
      Pl = betainc(w, N*a, N*b); Pu = betainc(w, N*a, N*b, 'upper'); fex = bpdf(w, N*a, N*b)/s2;
   else
      a = 1; b = 2; c = 0.5;
      K = CGF(@(t) -N*a*log(1 - t(1) - t(2)) - N*b*log(1 - t(2) - t(3)) - N*c*log(1 - t(2)), 3, ...
         'domain', @(t) t(1) + t(2) < 1 && t(2) + t(3) < 1 && t(2) < 1);
      s2 = N*(a + b + c); s3 = N*b; w = betaincinv(q, N*a, N*c); x = [w*(s2 - s3) s2 s3];
      Pl = betainc(w, N*a, N*c); Pu = betainc(w, N*a, N*c, 'upper'); fex = bpdf(w, N*a, N*c)/(s2 - s3);
   end
   r = conditional_tailprob(K, x, 1, 'order', 5); f = conditional_density(K, x, 1, 'order', 5);
   good = good && strcmp(r.status, 'ok') && strcmp(f.status, 'ok') && printed(Pu, str2double(cells{3})) && round(r.tau(1), 2) == str2double(cells{4});
   [nk, nc] = rowcheck([(r.F - Pu)/min(Pu, Pl); f.f/fex - 1], cells(5:14)); nok = nok + nk; ncell = ncell + nc;
end
ok = tablecheck('table gamma sum | total', nok, ncell, 450, good, verbose) && ok;
% the symmetric case a = b: the expansions at q and 1 - q add up to one, at every order
K = CGF(@(t) -4*log(1 - t(1) - t(2)) - 4*log(1 - t(2)), 2, 'domain', @(t) t(1) + t(2) < 1 && t(2) < 1);
w = betaincinv(0.01, 4, 4);
F1 = conditional_tailprob(K, [w*8 8], 1, 'order', 5).F; F2 = conditional_tailprob(K, [(1 - w)*8 8], 1, 'order', 5).F;
ok = echeck('cond symmetric beta', max(abs(F1 + F2 - 1)), 1e-14, verbose) && ok;
% a quadratic form given another (cond_bqf_rows.tex): exact by quadrature over the common component
cases = {[15 20 10], [5 10 8]; [40 60 30], [20 10 30]};
opt = {'RelTol', 1e-13, 'AbsTol', 0};
nok = 0; ncell = 0; good = true;
for i = find(rows.table == 2)'
   cells = rows.cells{i}; m = cases{rows.block(i), 1}; lam = cases{rows.block(i), 2};
   c2 = str2double(cells{1}); q = str2double(cells{2});
   mu2 = m(2) + lam(2) + m(3) + lam(3); sd2 = sqrt(2*(m(2) + 2*lam(2)) + 2*(m(3) + 2*lam(3)));
   x2 = mu2 + c2*sd2;
   g02 = @(u) ncx2x(u, m(3), lam(3), 'pdf').*ncx2x(x2 - u, m(2), lam(2), 'pdf');
   den = integral(g02, 0, x2, opt{:});
   ex = @(x1, what) bqfint(@(u) g02(u).*ncx2x(x1 - u, m(1), lam(1), what), x1, x2, opt)/den;
   x1 = illinois(@(x) ex(x, 'cdf') - q, 1e-6, mu2 + m(1) + lam(1) + 20*sd2);
   Pu = ex(x1, 'upp'); Pl = ex(x1, 'cdf');
   fex = integral(@(u) g02(u).*ncx2x(x1 - u, m(1), lam(1), 'pdf'), 0, min(x1, x2), opt{:})/den;
   K = common_component_cgf(m, lam);
   r = conditional_tailprob(K, [x1 x2], 1, 'order', 5); f = conditional_density(K, [x1 x2], 1, 'order', 5);
   good = good && printed(Pu, str2double(cells{3})) && round(r.tau(1), 2) == str2double(cells{4});
   [nk, nc] = rowcheck([(r.F - Pu)/min(Pu, Pl); f.f/fex - 1], cells(5:14)); nok = nok + nk; ncell = ncell + nc;
end
ok = tablecheck('table quad. form | other', nok, ncell, 300, good, verbose) && ok;
% logistic regression, lattice (logit_rows.tex): the exact joint pmf of S under beta = 0
n = 30; X = [ones(n, 1) mod((0:n-1)', 5) mod((0:n-1)', 3)];
mx = sum(X, 1);
pm = zeros(mx + 1); pm(1, 1, 1) = 1;
for j = 1:n, pm = (pm + circshift(pm, X(j, :)))/2; end
K = logistic_regression_cgf(X);
nok = 0; ncell = 0; good = true; it = find(rows.table == 4)'; i = 0;
for kk = [15 15; 10 12; 12 9]'
   k0 = kk(1); k2 = kk(2);
   qv = reshape(pm(k0+1, :, k2+1), [], 1); qv = qv/sum(qv);
   Qu = flipud(cumsum(flipud(qv))); Ql = cumsum(qv);
   k1s = unique(arrayfun(@(tg) find(Ql >= tg, 1) - 1, [0.01 0.05 0.25 0.5 0.75 0.95 0.99]), 'stable');
   for k1 = k1s
      i = i + 1; cells = rows.cells{it(i)};
      Pu = Qu(k1+1); pk = qv(k1+1); x = [k0 k1 k2];
      r = conditional_tailprob(K, x, 2, 'order', 4, 'lattice', true);
      f = conditional_density(K, x, 2, 'order', 4, 'lattice', true);
      ro = conditional_tailprob(K, x, 2, 'order', 4, 'lattice', true, 'sinh', 'origin');
      good = good && str2double(cells{1}) == k1 && printed(Pu, str2double(cells{2})) && round(r.tau(2), 2) == str2double(cells{3}) ...
         && strcmp(r.status, 'ok') && abs(ro.F(2)/r.F(2) - 1) < 1e-3;
      [nk, nc] = rowcheck([(r.F - Pu)/min(Pu, 1 - Pu); f.f/pk - 1; (ro.F(2) - Pu)/min(Pu, 1 - Pu)], cells(4:12)); nok = nok + nk; ncell = ncell + nc;
      % the lower tail P(S_1 <= k_1 - 1) is the complement, at every order
      rl = conditional_tailprob(K, x - [0 1 0], 2, 'order', 4, 'lattice', true, 'lower', true);
      good = good && max(abs(rl.F + r.F - 1)) < 1e-14;
   end
end
ok = tablecheck('table logistic regr.', nok, ncell, 189, good, verbose) && ok;
% lattice sums without conditioning (lattice_rows.tex): Poisson(5) and the bivariate Poisson
% with a common component; the unconditional lattice path is the conditional one with B empty
ppmf = @(k, l) (k >= 0).*exp(k*log(l) - l - gammaln(k + 1));
pupp = @(k, l) gammainc(l, max(k, eps)).*(k > 0) + (k <= 0);       % P(S >= k)
l1 = 2; l2 = 2; l0 = 1;
K1 = poisson_cgf(5); K2 = CGF(@(t) l1*(exp(t(1)) - 1) + l2*(exp(t(2)) - 1) + l0*(exp(t(1) + t(2)) - 1), 2);
nok = 0; ncell = 0; good = true;
for i = find(rows.table == 0)'
   cells = rows.cells{i};
   if rows.block(i) == 1
      k = str2double(cells{1}); K = K1; Pex = pupp(k, 5); pex = ppmf(k, 5);
      rc = conditional_tailprob(K, k, 1, 'order', 4, 'lattice', true); fc = conditional_density(K, k, 1, 'order', 4, 'lattice', true);
   else
      k = sscanf(cells{1}, '%d,%d')'; K = K2; y = 0:60;
      Pex = sum(ppmf(y, l0).*pupp(k(1) - y, l1).*pupp(k(2) - y, l2));
      y = 0:min(k); pex = sum(ppmf(y, l0).*ppmf(k(1) - y, l1).*ppmf(k(2) - y, l2));
      rc = conditional_tailprob(K, k, 1:2, 'order', 4, 'lattice', true); fc = conditional_density(K, k, 1:2, 'order', 4, 'lattice', true);
   end
   r = tailprob(K, k, 'order', 4, 'lattice', true); f = density(K, k, 'order', 4, 'lattice', true);
   good = good && isequal(r.F, rc.F) && isequal(f.f, fc.f) && strcmp(r.status, 'ok') ...
      && isequal(round(r.tau, 2), sscanf(strrep(cells{3}, '+', ''), '%f,%f')') ...
      && round(Pex, 4) == str2double(cells{2}) && round(pex, 4) == str2double(cells{8});
   [nk, nc] = rowcheck([(r.F - Pex)/min(Pex, 1 - Pex); f.f/pex - 1], [cells(4:7) cells(9:12)]); nok = nok + nk; ncell = ncell + nc;
end
ok = tablecheck('table lattice sums', nok, ncell, 88, good, verbose) && ok;
% the mean of N independent ratios of quadratic forms (sph_rows_*.tex): P(Rbar >= x/N) =
% P(X >= x | D_1 = ... = D_N = 1), X = sum_i v_i'A_iv_i, D_i = v_i'v_i, against the exact upper
% quantiles of the paper (Gil-Pelaez inversion, as in the Julia tests); the columns are the normal
% approximation, Skovgaard's approximation (LR), the leading term (HR) and the expansions of orders
% 1..4 (1..2 for N = 10).  Skovgaard's formula at the median, where 1/w - 1/z cancels, is
% evaluated in BigFloat by the Julia tests; in double precision it is not, and those five cells are
% left out (150 of the 155 cells are checked)
qex = {[-0.649984406337777 0.173640308684513 0.404657822568259 0.830136764358454 1.28888641542161], ...
       [-0.28101745873196 0.342338885972127 0.518020283322166 0.844515706483764 1.20359268966255], ...
       [-0.179727297726595 0.335098902241048 0.480370305892609 0.751043128437577 1.05038179960608], ...
       [-1.17188025603077 -0.0622176078581697 0.251186562311314 0.834501869250704 1.47747984786112], ...
       [-0.525777371167559 0.329433323857853 0.571415079075747 1.02356949383106 1.52620380057527]};
design = [5 5; 5 15; 5 25; 10 4; 10 14];                % N and T_0 (T_i = T_0 + i)
ps = [0.5 0.1 0.05 0.01 0.001];
Phi = @(x) erfc(-x/sqrt(2))/2; phi = @(x) exp(-x.^2/2)/sqrt(2*pi);
nok = 0; ncell = 0; good = true; it = find(rows.table == 3)';
for block = 1:5
   N = design(block, 1); d = N + 1; lam = cell(1, N);
   for i = 1:N
      T = design(block, 2) + i; A = diag(0.5*ones(T-1, 1), 1); A = A + A.';
      Q = null(ones(1, T)); lam{i} = sort(eig((Q.'*A*Q + (Q.'*A*Q).')/2)).';
   end
   K = CGF(@(t) sphK(t, lam), d, 'domain', @(t) sphdom(t, lam));
   mu = sum(cellfun(@mean, lam));
   va = sum(cellfun(@(a) (sum(a)^2 + 2*sum(a.^2))/(numel(a)*(numel(a) + 2)) - mean(a)^2, lam));
   ib = it(rows.block(it) == block);
   for k = 1:5
      cells = rows.cells{ib(k)}; p = ps(k); x = qex{block}(k); xv = [x ones(1, N)];
      r = conditional_tailprob(K, xv, 1, 'order', 4 - 2*(N == 10));
      good = good && strcmp(r.status, 'ok') && round(x/N, 4) == str2double(cells{2}) && round(r.tau(1), 3) == str2double(cells{4});
      D = derivatives(K, r.tau, 2); Dt = derivatives(K, r.taut, 2);
      H = hess(D, d); Ht = hess(Dt, d); Ht = Ht(2:d, 2:d);
      w = sign(r.tau(1))*sqrt(2*((Dt.value(1) - r.taut*xv.') - (D.value(1) - r.tau*xv.')));
      z = r.tau(1)*sqrt(det(H)/det(Ht)); u = r.tau(1)*sqrt(H(1, 1) - H(1, 2:d)*(H(2:d, 2:d)\H(2:d, 1)));
      Plr = Phi(-w) - phi(w)*(1/w - 1/z); Phr = u/z*exp((u^2 - w^2)/2)*Phi(-u);
      if p < 0.5, good = good && abs(r.info.Fp(1)/Phr - 1) < 1e-12; end   % (20) of the paper
      comp = [([Phi(-(x - mu)/sqrt(va)) Plr Phr] - p)/p, (r.F.' - p)/p];
      cc = cells(5:4+numel(comp));
      if p == 0.5, comp(2) = []; cc(2) = []; end
      [nk, nc] = rowcheck(comp, cc); nok = nok + nk; ncell = ncell + nc;
   end
end
ok = tablecheck('table sums of ratios', nok, ncell, 150, good, verbose) && ok;
end

function ok = test_conditional(verbose)
ok = true;
rd = @(a, b) max(abs(a(:) - b(:))./abs(b(:)));
% B empty: the expansions of tailprob and density
Kg = gamma_cgf(3);
for x = [6 1.5]
   ok = echeck(sprintf('cond B empty gamma x=%g', x), rd(conditional_tailprob(Kg, x, 1, 'order', 4).F, tailprob(Kg, x, 'order', 4).F), 1e-13, verbose) && ok;
   ok = echeck(sprintf('cond B empty gamma f x=%g', x), rd(conditional_density(Kg, x, 1, 'order', 4).f, density(Kg, x, 'order', 4).f), 1e-14, verbose) && ok;
end
K5 = exponential_sums_cgf([1 0; 1 1; 0 1], 'n', 5);
for x = {[3 3], [2.5 3], [1.2 3.5], [0.8 0.9]}                  % tau >= 0, tau_1 = 0, mixed, negative
   ok = echeck(sprintf('cond B empty %s', mat2str(x{1})), rd(conditional_tailprob(K5, x{1}, [1 2], 'order', 4).F, tailprob(K5, x{1}, 'order', 4).F), 1e-12, verbose) && ok;
   ok = echeck(sprintf('cond B empty f %s', mat2str(x{1})), rd(conditional_density(K5, x{1}, 1:2, 'order', 4).f, density(K5, x{1}, 'order', 4).f), 1e-14, verbose) && ok;
end
% Gaussian vectors: exact at every order, one, two and three tail components, every sign of the
% saddlepoint (the inclusion-exclusion marginalizes the components of N \ T)
Sig = [2.0 0.6 0.5 0.3; 0.6 1.5 -0.4 0.2; 0.5 -0.4 1.0 0.1; 0.3 0.2 0.1 0.8]; mu = [0.5; -1.0; 0.2; 0.3];
Kn = normal_cgf(mu, Sig);
xs = {[1.0 0.0 0.5 0.0], [-1.0 0.0 0.5 0.0], [1.5 0.0 0.5 0.3], [0.2 -2.5 0.5 0.3], [-1.0 -2.5 0.5 0.3], [1.2 -0.8 0.9 0.3]};
As = {1, 1, [1 2], [1 2], [2 1], [1 2 3]};
for i = 1:6
   x = xs{i}; Av = sort(As{i}); [m, V] = condnorm(Sig, mu, x, Av);
   ex = orthantx(x(Av).' - m, V);
   r = conditional_tailprob(Kn, x, As{i}, 'order', 3 - (numel(Av) == 3));
   ok = check(sprintf('cond gauss exact %d', i), r.F, ex*ones(size(r.F)), 1e-13, verbose) && strcmp(r.status, 'ok') && ok;
   z = x(Av).' - m; fex = exp(-z.'*(V\z)/2)/sqrt((2*pi)^numel(Av)*det(V));
   f = conditional_density(Kn, x, As{i}, 'order', 3).f;
   ok = check(sprintf('cond gauss exact f %d', i), f, fex*ones(3, 1), 1e-13, verbose) && ok;
end
r = conditional_tailprob(Kn, xs{4}, [1 2]);
ok = check('cond gauss negative tau', double(any(r.tau(1:2) < 0)), 1, 0, verbose) && ok;
% lower tails and mixed corners, by the reflection
x = xs{4}; [m, V] = condnorm(Sig, mu, x, [1 2]);
rm = conditional_tailprob(Kn, x, [1 2], 'order', 2, 'signs', [-1 1]);
ok = check('cond gauss mixed corner', rm.F(2), orthantx([m(1) - x(1); x(2) - m(2)], [V(1, 1) -V(1, 2); -V(1, 2) V(2, 2)]), 1e-13, verbose) && ok;
rl = conditional_tailprob(Kn, x, [1 2], 'order', 2, 'lower', true);
ok = check('cond gauss lower', rl.F(2), orthantx(m - x(1:2).', V), 1e-13, verbose) && ok;
% two tail components given the total: the Dirichlet distribution, exact by quadrature
al = [5 7.5 10]; s = sum(al); lgD = gammaln(s) - sum(gammaln(al));
bpdf = @(w, p, q) exp((p - 1)*log(w) + (q - 1)*log1p(-w) + gammaln(p + q) - gammaln(p) - gammaln(q));
Kd = CGF(@(t) -al(1)*log(1 - t(1) - t(3)) - al(2)*log(1 - t(2) - t(3)) - al(3)*log(1 - t(3)), 3, ...
   'domain', @(t) t(1) + t(3) < 1 && t(2) + t(3) < 1 && t(3) < 1);
w = [0.35 0.45; 0.15 0.40; 0.30 0.25; 0.15 0.25];
for i = 1:4
   w1 = w(i, 1); w2 = w(i, 2);
   ex = integral(@(v) bpdf(v, al(1), al(2) + al(3)).*betainc(w2./(1 - v), al(2), al(3), 'upper'), w1, 1 - w2, 'RelTol', 1e-13, 'AbsTol', 0);
   fex = exp(lgD + (al(1) - 1)*log(w1) + (al(2) - 1)*log(w2) + (al(3) - 1)*log(1 - w1 - w2))/s^2;
   r = conditional_tailprob(Kd, [w1*s w2*s s], [1 2], 'order', 4);
   e = abs(r.F/ex - 1);
   good = e(1) < 2e-2 && e(2) < 1e-3 && e(3) < 2e-5 && e(4) < 1e-5 && e(4) < e(1) && strcmp(r.status, 'ok') ...
      && max(r.info.kappa) < 1e3 && r.info.err(end) < 1e-9*r.F(end);
   fd = conditional_density(Kd, [w1*s w2*s s], [1 2], 'order', 4).f/fex - 1;
   good = good && abs(fd(1) - 3.3e-2) < 1e-3 && abs(fd(4)) < 2e-6;      % the same at every point
   ok = check(sprintf('cond dirichlet exact %d', i), double(good), 1, 0, verbose) && ok;
end
% the derivative interface: closed-form derivatives with the polynomial substitution into the
% canonical coordinates, and with Kwhite as conditional_tailprob supplies it; the leading term
N = 16;
K3 = CGF(@(t) -N*log(1 - t(1) - t(2)) - 2*N*log(1 - t(2) - t(3)) - 0.5*N*log(1 - t(2)), 3);
x3 = [19 14*N/4 8*N/4];                                             % above the conditional mean 16
[P1, f1, i1] = spa_cond(@kf3, x3, 1, 4);
r2 = conditional_tailprob(K3, x3, 1, 'order', 4); f2 = conditional_density(K3, x3, 1, 'order', 4);
ok = echeck('spa_cond closed form', max(rd(P1, r2.F), rd(f1, f2.f)), 1e-10, verbose) && ok;
[P3, ~, i3] = spa_cond(kfunof(K3, 9), x3, 1, 4, struct('density', false, 'Kwhite', kwhiteof(K3, 9)));
ok = check('spa_cond with Kwhite', double(isequal(P3, r2.F) && isequal(i3.Pp, r2.info.Fp) && isequal(i1.Pp(2:2:end), P1) && numel(i1.Pp) == 8), 1, 0, verbose) && ok;
ok = check('spa_cond saddlepoints', [i1.tau i1.taut], [r2.tau r2.taut], 1e-12, verbose) && i1.taut(1) == 0 && ok;
D = derivatives(K3, r2.tau, 2); Dt = derivatives(K3, r2.taut, 2);
H = hess(D, 3); Ht = hess(Dt, 3); Ht = Ht(2:3, 2:3);
wh = sqrt(2*((Dt.value(1) - r2.taut*x3.') - (D.value(1) - r2.tau*x3.')));
u = r2.tau(1)*sqrt(H(1, 1) - H(1, 2:3)*(H(2:3, 2:3)\H(2:3, 1))); z = r2.tau(1)*sqrt(det(H)/det(Ht));
ok = check('cond leading term', r2.info.Fp(1), u/z*exp((u^2 - wh^2)/2)*erfc(u/sqrt(2))/2, 1e-13, verbose) && ok;
ok = check('cond tau, err, nG', double(r2.tau(1) > 0 && r2.info.err(end) < 1e-12*r2.F(end) && r2.info.nG == 64), 1, 0, verbose) && ok;
% the arguments
bad = {@() conditional_tailprob(K3, x3, 4), @() conditional_tailprob(K3, x3, [1 1]), @() conditional_tailprob(K3, x3, 1, 'order', 0), ...
       @() conditional_tailprob(K3, x3, [1 2], 'signs', 1), @() conditional_tailprob(K3, x3, 1, 'sinh', 'other'), @() conditional_density(K3, x3(1:2), 1)};
ok = check('cond argument errors', sum(cellfun(@raises, bad)), numel(bad), 0, verbose) && ok;
end

function ok = test_lattice_quantiles(verbose)
ok = true;
ppmf = @(k, l) exp(k*log(l) - l - gammaln(k + 1));
pupp = @(k, l) gammainc(l, k);                                       % P(S >= k), k >= 1
K1 = poisson_cgf(5);
% the complement: P(S <= k - 1) + P(S >= k) = 1 at every order, through the reflection
for k = [3 8]
   ok = echeck(sprintf('lattice complement k=%d', k), max(abs(tailprob(K1, k, 'order', 4, 'lattice', true).F + tailprob(K1, k - 1, 'order', 4, 'lattice', true, 'lower', true).F - 1)), 1e-14, verbose) && ok;
end
% no saddlepoint at the continuity-corrected point below the support
ok = check('lattice P(S >= 0) error', double(raises(@() tailprob(K1, 0, 'lattice', true), 'spa_cond:support')), 1, 0, verbose) && ok;
r = tailprob(K1, 1, 'order', 4, 'lattice', true);                    % P(S >= 1) = 1 - P(S = 0)
ok = check('lattice P(S >= 1)', double(abs(r.F(4) - pupp(1, 5)) < 2e-2*ppmf(0, 5)), 1, 0, verbose) && ok;
ok = check('lattice normalize error', double(raises(@() density(K1, 3, 'lattice', true, 'normalize', true), 'CGF:normalize')), 1, 0, verbose) && ok;
% binomial and negative binomial against exact values: the errors fall with the order
Kb = binomial_cgf(40, 0.3); good = true;
for k = [6 18 25]
   Pu = betainc(0.3, k, 41 - k);                                     % P(S >= k)
   pk = exp(gammaln(41) - gammaln(k + 1) - gammaln(41 - k) + k*log(0.3) + (40 - k)*log(0.7));
   e = abs(tailprob(Kb, k, 'order', 4, 'lattice', true).F/Pu - 1); ef = abs(density(Kb, k, 'order', 4, 'lattice', true).f/pk - 1);
   good = good && e(4) < 5e-4 && e(4) < e(1) && ef(4) < 1e-4 && ef(4) < ef(1)/10;
end
ok = check('lattice binomial exact', double(good), 1, 0, verbose) && ok;
Knb = negative_binomial_cgf(5, 0.4); good = true;                   % failures before the 5th success
for k = [2 7 20]
   Pu = betainc(0.6, k, 5);                                          % P(S >= k)
   pk = exp(gammaln(k + 5) - gammaln(5) - gammaln(k + 1) + 5*log(0.4) + k*log(0.6));
   e = abs(tailprob(Knb, k, 'order', 4, 'lattice', true).F/Pu - 1); ef = abs(density(Knb, k, 'order', 4, 'lattice', true).f/pk - 1);
   good = good && e(4) < 1e-3 && ef(4) < 1e-4 && ef(4) < ef(1)/10;
end
ok = check('lattice neg. binomial exact', double(good), 1, 0, verbose) && ok;
% the expansion of the amplitude about the origin (Kolassa) agrees closely for small saddlepoints
ro = tailprob(K1, 7, 'order', 3, 'lattice', true, 'sinh', 'origin').F; rs = tailprob(K1, 7, 'order', 3, 'lattice', true).F;
ok = check('lattice origin vs saddle', ro(3), rs(3), 1e-4, verbose) && ok;
% the first-order error of the Poisson tail falls at least like 1/lambda
e1 = @(lam) abs(tailprob(poisson_cgf(lam), round(lam + 2*sqrt(lam)), 'order', 1, 'lattice', true).F(1)/pupp(round(lam + 2*sqrt(lam)), lam) - 1);
ok = check('lattice Poisson rate', double(e1(100) < e1(25)/4), 1, 0, verbose) && ok;
% quantiles: the distribution function at the quantile, the exact gamma, the saddlepoint
Kg = gamma_cgf(3); e = 0; ee = 0; et = 0;
for p = [1e-6 0.05 0.5 0.95]
   for lower = [true false]
      q = quantile(Kg, p, 'order', 4, 'lower', lower);
      if lower, F = tailprob(Kg, q.q, 'order', 4, 'lower', true).F(end); ex = gammainc(q.q, 3);
      else, F = tailprob(Kg, q.q, 'order', 4).F(end); ex = gammainc(q.q, 3, 'upper'); end
      e = max([e abs(F - p)/min(p, 1 - p) abs(q.F - p)/min(p, 1 - p)]);
      ee = max(ee, abs(ex/p - 1)); et = max(et, abs(saddlepoint(Kg, q.q)/q.tau - 1));
      ok = strcmp(q.status, 'ok') && ok;
   end
end
ok = echeck('quantile round trip', e, 1e-12, verbose) && ok;
ok = echeck('quantile gamma exact', ee, 1e-4, verbose) && ok;
ok = echeck('quantile saddlepoint', et, 1e-8, verbose) && ok;
q = quantile(normal_cgf(1, 4), 0.025, 'order', 2);                  % the normal: exact
ok = check('quantile normal exact', q.q, 1 - 2*sqrt(2)*erfcinv(0.05), 1e-12, verbose) && ok;
ok = check('quantile arguments', double(raises(@() quantile(Kg, 1.5)) && raises(@() quantile(normal_cgf([0; 0], eye(2)), 0.5))), 1, 0, verbose) && ok;
ok = check('quantile of data', quantile([1 2 3 4], 0.5), 2.5, 0, verbose) && ok;   % MATLAB's own
% ratios: the noncentral F, against its exact distribution function
m_ = 4; n_ = 12; lam = 3;
Ka = ncx2_cgf(m_, lam); Kc = ncx2_cgf(n_);
KF = CGF(@(t) Ka.fun(t(1)) + Kc.fun(t(2)), 2, 'domain', @(t) t(1) < 0.5 && t(2) < 0.5);
e = 0; ee = 0;
for p = [0.01 0.5 0.99]
   q = ratio_quantile(KF, p, 'order', 4);
   e = max(e, abs(ratio_tailprob(KF, q.r, 'order', 4, 'lower', true).F(end) - p)/min(p, 1 - p));
   ee = max(ee, abs(dncf_exact_dp(q.r*n_/m_, m_, n_, lam, 0) - p)/min(p, 1 - p));
end
ok = echeck('ratio quantile round trip', e, 1e-12, verbose) && ok;
ok = echeck('ratio quantile exact', ee, 3e-4, verbose) && ok;
q = ratio_quantile(KF, 1e-4, 'order', 4, 'lower', false);
[~, upp] = dncf_exact_dp(q.r*n_/m_, m_, n_, lam, 0);
ok = check('ratio quantile upper', ratio_tailprob(KF, q.r, 'order', 4).F(end), 1e-4, 1e-11, verbose) && ok;
ok = check('ratio quantile upper exact', upp, 1e-4, 1e-3, verbose) && ok;
ok = check('ratio quantile arguments', double(raises(@() ratio_quantile(Kg, 0.5))), 1, 0, verbose) && ok;
% normalized densities
cg = @(a) exp(gammaln(a) + a + (1/2 - a)*log(a))/sqrt(2*pi);       % Daniels's factor for the gamma
for a = [0.5 3]
   r = density(gamma_cgf(a), 2, 'order', 4, 'normalize', true);
   tol = 1e-13; if a < 1, tol = 1e-8; end
   ok = check(sprintf('normalize gamma(%g) mass', a), r.info.mass(1), cg(a), tol, verbose) && ok;
   % the bracket of the gamma expansion is constant, so every normalized order is exact
   ok = check(sprintf('normalize gamma(%g) exact', a), r.f, exp((a - 1)*log(2) - 2 - gammaln(a))*ones(4, 1), tol, verbose) && ok;
end
ok = check('normalize normal', density(normal_cgf(1, 2), 0.3, 'order', 3, 'normalize', true).info.mass, ones(3, 1), 1e-13, verbose) && ok;
% the normalized expansion integrates to one, and the masses can be reused
Kx = ncx2_cgf(4, 3);
r = density(Kx, 7, 'order', 3, 'normalize', true);
% (the ranges of the integrals over x, r and q are finite, the masses outside far below the
% tolerances: the saddlepoint iteration fails at the points near 1e10 to which integral maps an
% infinite end, and the Studentized saddlepoint is found by continuation from q0 = 1, slowly
% for |q| large, where the density is below 1e-14 outside (-5, 20))
tot = integral(@(x) density(Kx, x, 'order', 3, 'normalize', r.info.mass).f, 1e-9, 400, 'Waypoints', [10 40], 'ArrayValued', true, 'RelTol', 1e-10, 'AbsTol', 0);
ok = check('normalize ncx2 total', tot, ones(3, 1), 1e-9, verbose) && ok;
ok = check('normalize reuse', double(isequal(density(Kx, 7, 'order', 3, 'normalize', r.info.mass).f, r.f)), 1, 0, verbose) && ok;
% two dimensions: a linear transformation of two independent gamma variables; the mass of the first
% order is the product of the univariate factors, that of the second order c_1(1 - 1/24 - 1/36), the
% brackets of the gamma expansion being 1 - 1/(12a)
Kl = linear_combination(CGF(@(t) -2*log(1 - t(1)) - 3*log(1 - t(2)), 2, 'domain', @(t) t(1) < 1 && t(2) < 1), [1 0.5; 0.3 1]);
m2 = density(Kl, [1 2], 'order', 2, 'normalize', true).info.mass;
ok = check('normalize 2-d exact', m2, cg(2)*cg(3)*[1; 1 - 1/24 - 1/36], 1e-9, verbose) && ok;
% a ratio and a Studentized mean: the normalized expansions integrate to one
rr = ratio_density(KF, 1, 'order', 3, 'normalize', true);
tot = integral(@(r) ratio_density(KF, r, 'order', 3, 'normalize', rr.info.mass).f, 1e-6, 1e6, 'Waypoints', [1 5 50], 'ArrayValued', true, 'RelTol', 1e-9, 'AbsTol', 0);
ok = check('normalize ratio total', tot, ones(3, 1), 1e-8, verbose) && ok;
mu = 1; nu = 3; n = 16;
Kn = sample_mean(CGF(@(t) mu*t(1) + t(1)^2/2 - nu/2*log(1 - 2*t(2)/nu), 2, 'domain', @(t) t(2) < nu/2), n);
rs = studentized_density(Kn, 1.5, 'order', 2, 'normalize', true);
tot = integral(@(q) studentized_density(Kn, q, 'order', 2, 'normalize', rs.info.mass).f, -5, 20, 'Waypoints', [0 1 2.5], 'ArrayValued', true, 'RelTol', 1e-8, 'AbsTol', 0);
ok = check('normalize studentized total', tot, ones(2, 1), 1e-8, verbose) && ok;
ok = check('normalize studentized mass', rs.info.mass(1), 1, 1e-2, verbose) && ok;
end

function rows = readrows(fn)
% the rows of conditional_rows.txt: table id, block, printed cells
L = readlines(fn); L = L(strlength(L) > 0 & ~startsWith(L, "#"));
rows = struct('table', zeros(numel(L), 1), 'block', zeros(numel(L), 1), 'cells', {cell(numel(L), 1)});
for i = 1:numel(L)
   f = split(strtrim(L(i)))';
   rows.table(i) = str2double(f(1)); rows.block(i) = str2double(f(2)); rows.cells{i} = cellstr(f(3:end));
end
end
function tf = printed(c, p)
% a computed value reproduces a printed one (two significant digits) when it rounds to it; values
% below 1e-10, at the rounding level of the exact values, are compared to 2e-12 absolute
if abs(p) < 1e-10, tf = abs(c - p) <= 2e-12; return, end
e = floor(log10(abs(p)) + 1e-9);
tf = abs(c - p) <= 0.05*10^e*(1 + 1e-9);
end
function [nok, ncell] = rowcheck(comp, cells)
nok = 0; ncell = numel(comp);
for k = 1:numel(comp)
   if strcmp(cells{k}, '-'), nok = nok + 1; continue, end
   nok = nok + printed(comp(k), str2double(cells{k}));
end
end
function ok = tablecheck(name, nok, ncell, nexp, good, verbose)
ok = nok == nexp && ncell == nexp && good;
if verbose || ~ok, fprintf('%-24s %3d of %3d cells     %s\n', name, nok, ncell, ifelse(ok, 'ok', 'FAILED')); end
end
function v = ncx2x(x, nu, lam, what)
% distribution function, upper tail or density of chi^2_nu(lam) at x (elementwise), as Poisson
% mixtures of central chi-squared values (both tails directly), as ncx2_exact of the Julia package
mu = lam/2; s = 15*sqrt(mu + 1) + 25;
j = (max(0, floor(mu - s)):ceil(mu + s))';
w = exp(-mu + j*log(mu) - gammaln(j + 1)); keep = w >= 1e-24*max(w); j = j(keep); w = w(keep);
v = zeros(size(x)); pos = x > 0; xp = x(pos)/2; acc = zeros(size(xp));
for i = 1:numel(j)
   a = nu/2 + j(i);
   switch what
      case 'cdf', acc = acc + w(i)*gammainc(xp, a);
      case 'upp', acc = acc + w(i)*gammainc(xp, a, 'upper');
      case 'pdf', acc = acc + w(i)*exp((a - 1)*log(xp) - xp - gammaln(a))/2;
   end
end
v(pos) = acc;
if strcmp(what, 'upp'), v(~pos) = 1; end
end
function v = bqfint(g, x1, x2, opt)
% the integral over the common component on (0, x2), split at x1 < x2 (where the integrand has a kink)
if x1 < x2, v = integral(g, 0, x2, 'Waypoints', x1, opt{:}); else, v = integral(g, 0, x2, opt{:}); end
end
function c = illinois(f, a, b)
fa = f(a); fb = f(b); side = 0;
for it = 1:200
   c = (a*fb - b*fa)/(fb - fa); fc = f(c);
   if abs(fc) < 1e-15 || abs(b - a) < 1e-13*abs(c), return, end
   if sign(fc) == sign(fb)
      b = c; fb = fc; if side == 1, fa = fa/2; end, side = 1;
   else
      a = c; fa = fc; if side == -1, fb = fb/2; end, side = -1;
   end
end
c = (a + b)/2;
end
function [m, V] = condnorm(Sig, mu, x, A)
% the conditional mean and covariance of X_A given X_B = x_B under N(mu, Sig)
B = setdiff(1:numel(mu), A); C = Sig(A, B)/Sig(B, B);
m = mu(A) + C*(x(B).' - mu(B)); V = Sig(A, A) - C*Sig(B, A); V = (V + V.')/2;
end
function P = orthantx(a, V)
% P(Z >= a), Z ~ N(0, V), in one, two or three dimensions, by (nested) adaptive quadrature over
% the first component of the conditional orthant probabilities
k = numel(a); a = a(:);
if k == 1, P = erfc(a/sqrt(2*V))/2; return, end
s1 = sqrt(V(1, 1)); b = V(2:k, 1)/V(1, 1); W = V(2:k, 2:k) - b*V(1, 2:k); W = (W + W.')/2;
g = @(z) arrayfun(@(zz) exp(-zz^2/(2*V(1, 1)))/(sqrt(2*pi)*s1)*orthantx(a(2:k) - b*zz, W), z);
P = integral(g, a(1), Inf, 'RelTol', 1e-14, 'AbsTol', 0);
end
function H = hess(D, d)
% the Hessian from derivatives(K, t, 2)
H = zeros(d); E = eye(d);
for i = 1:d, for j = 1:d, H(i, j) = D.value(all(D.nu == E(i, :) + E(j, :), 2)); end, end
end
function v = sphK(t, lam)
% the joint cgf of X = sum_i v_i'A_iv_i and D_i = v_i'v_i, v_i ~ N(0, I) of the dimension of A_i, from
% the eigenvalues lam{i} of A_i
v = 0;
for i = 1:numel(lam), v = v - sum(log(1 - 2*(t(1)*lam{i} + t(i+1))))/2; end
end
function tf = sphdom(t, lam)
tf = true;
for i = 1:numel(lam), tf = tf && all(1 - 2*(t(1)*lam{i} + t(i+1)) > 0); end
end
function v = kf3(nu, t)
% closed-form derivatives of K(t) = -N(a log(1 - t1 - t2) + b log(1 - t2 - t3) + c log(1 - t2)),
% the gamma sum of the Julia tests (N = 16, a = 1, b = 2, c = 1/2)
N = 16; a = 1; b = 2; c = 0.5;
n = sum(nu); p1 = 1 - t(1) - t(2); p2 = 1 - t(2) - t(3); p3 = 1 - t(2);
if p1 <= 0 || p2 <= 0 || p3 <= 0, v = NaN; return, end
if n == 0, v = -N*(a*log(p1) + b*log(p2) + c*log(p3)); return, end
v = 0;
if nu(3) == 0, v = v + a*factorial(n - 1)/p1^n; end
if nu(1) == 0, v = v + b*factorial(n - 1)/p2^n; end
if nu(1) == 0 && nu(3) == 0, v = v + c*factorial(n - 1)/p3^n; end
v = N*v;
end
function f = kfunof(K, maxdeg)
% the derivative interface of a CGF as CGF/tailprob passes it to the engine (the logic of
% private/cgf_kfun, which the tests cannot call)
lastt = []; lastN = -1; lastp = []; valid = false;
f = @Kfun;
   function v = Kfun(nu, t)
      kk = sum(nu); if kk <= 2, N = 2; else, N = max(maxdeg, kk); end
      t = double(reshape(t, [], 1));
      if ~valid || ~isequal(t, lastt) || N > lastN, lastp = taylor(K, t, N); lastt = t; lastN = N; valid = true; end
      if isempty(lastp), v = NaN; return, end
      v = coef(lastp, reshape(nu, 1, []));
   end
end
function f = kwhiteof(K, maxdeg)
% Kwhite(nu, u, M) as private/cgf_kwhite makes it
lastu = []; lastM = []; lastp = []; lastN = -1; valid = false;
f = @Kwhite;
   function v = Kwhite(nu, u, M)
      u = double(reshape(u, [], 1)); M = double(M); N = max(maxdeg, sum(nu));
      if ~valid || ~isequal(u, lastu) || ~isequal(M, lastM) || N > lastN, lastp = taylor(K, M*u, N, M); lastu = u; lastM = M; lastN = N; valid = true; end
      if isempty(lastp), v = NaN; return, end
      v = coef(lastp, reshape(nu, 1, []));
   end
end
function tf = raises(fh, id)
% true if fh() raises an error (with identifier id when given)
tf = false;
try
   fh();
catch err
   tf = nargin < 2 || strcmp(err.identifier, id);
end
end
function s = jlnum(p)
% p as Julia prints a Float64 in a string ("1.0e-6", "0.05")
if p < 1e-4, s = sprintf('%.1fe%d', p/10^floor(log10(p)), floor(log10(p))); else, s = num2str(p); end
end
function s = lower2str(tf), if tf, s = 'true'; else, s = 'false'; end, end

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
function ok = echeck(name, e, tol, verbose)
% a relative error e computed by the caller (in higher precision)
ok = e <= tol;
if verbose || ~ok, fprintf('%-24s max rel. diff %8.1e  %s\n', name, e, ifelse(ok, 'ok', 'FAILED')); end
end
function s = ifelse(c, a, b), if c, s = a; else, s = b; end, end
