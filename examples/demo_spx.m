%DEMO_SPX Examples of the SaddlepointExpansions package.
run(fullfile(fileparts(mfilename('fullpath')), '..', 'spx_setup'));

%% 1. A bivariate exponential mean (the example of the README)
K = CGF(@(t) -log(1 - t(1)) - log(1 - t(2)) - log(1 - t(1) - t(2)), 2, ...
        'domain', @(t) t(1) < 1 && t(2) < 1 && t(1) + t(2) < 1);
Kn = sample_mean(K, 5);
r = tailprob(Kn, [3 3], 'order', 4);
fprintf('P(X1 >= 3, X2 >= 3), orders 1-4: %s  (exact 0.02215196)\n', sprintf('%.8f ', r.F));
r = density(Kn, [3 3], 'order', 4);
fprintf('density, orders 1-4:             %s\n', sprintf('%.8f ', r.f));

%% 2. Table 6 of the paper, a pair of quadratic forms with a common component
% Q1 = X0 + X1, Q2 = X0 + X2, X_i ~ chi^2_{m_i}(lambda_i), (m0,m1,m2) = (10,15,20), (lambda0,lambda1,lambda2) = (8,5,10)
Kc = common_component_cgf([15 20 10], [5 10 8]);
x = [48.09950493836207795336339, 65.23368793961408597955183];   % E(Q) + (1.0, 1.5) sd(Q)
Fex = 0.0346756412725677484298474; fex = 0.0003658729471762786026251736;   % exact (25 digits)
r = tailprob(Kc, x, 'order', 5); s = density(Kc, x, 'order', 5);
fprintf('Table 6, (c1,c2) = (1.0,1.5): relative errors of the orthant probability %s\n', sprintf('%9.1e', r.F/Fex - 1));
fprintf('                              relative errors of the density             %s\n', sprintf('%9.1e', s.f/fex - 1));

%% 3. A trivariate mean with a saddlepoint with negative components (inclusion-exclusion)
A = [1 0 0; 0 1 0; 0 0 1; 1 1 0; 0 1 1];
Ke = exponential_sums_cgf(A, 'n', 6);
r = tailprob(Ke, [2.2 2.6 1.9], 'order', 3);
fprintf('trivariate: saddlepoint %s, P(X >= x) orders 1-3: %s\n', mat2str(r.tau, 4), sprintf('%.8f ', r.F));

%% 4. A ratio of quadratic forms, R = z''Az/z''z, z ~ N(0.5*1, T(0.3)), A = T(0.5), n = 8
n = 8; toep = @(rho) rho.^abs((1:n)' - (1:n));
Kq = quadratic_forms_cgf({toep(0.5), eye(n)}, 'Sigma', toep(0.3), 'mu', 0.5*ones(n, 1));
fprintf('ratio density at 1.2, orders 1-4: %s\n', sprintf('%.8f ', ratio_density(Kq, 1.2, 'order', 4).f));
fprintf('P(R >= 1.2), orders 1-4:          %s\n', sprintf('%.8f ', ratio_tailprob(Kq, 1.2, 'order', 4).F));

%% 5. Derivatives of any cumulant generating function by the Taylor class
Ku = uniform_cgf();                           % uniform on [-1, 1]: K(t) = log(sinh(t)/t)
D = derivatives(Ku, 0, 8);
fprintf('cumulants of the uniform distribution on [-1,1]: %s\n', mat2str(D.value(2:end)', 6));

%% 6. Two sample autocorrelations (Section S4 of the Supplementary Material): R_j = z'MA_jMz/z'Mz, j = 1, 2,
% z ~ N(0, I_50), M = I - 11'/50, (A_j)_ts = 1/2 for |t - s| = j; a common denominator, so K is the
% joint cgf of the three quadratic forms (dimension d + 1 = 3)
N = 50; Mc = eye(N) - ones(N)/N; Aj = @(j) (diag(ones(N-j, 1), j) + diag(ones(N-j, 1), -j))/2;
Kr = quadratic_forms_cgf({Mc*Aj(1)*Mc, Mc*Aj(2)*Mc, Mc});
r0 = [-1 -1]/sqrt(N);
Fex = 2.373966512166858033207522e-2; fex = 3.935188945615940593490785;   % exact (25 digits)
f = ratio_density(Kr, r0, 'order', 5).f; F = ratio_tailprob(Kr, r0, 'order', 5, 'lower', true).F;
fprintf('autocorrelations at %s: relative errors of the density %s\n', mat2str(r0, 4), sprintf('%9.1e', f/fex - 1));
fprintf('                                  relative errors of P(R <= r) %s\n', sprintf('%9.1e', F/Fex - 1));

%% 7. Three ratios with a common denominator: R_j = X_j/Y, X_j ~ Gamma(a_j), Y ~ Gamma(b)
a = [3 4 5]; b = 6; r0 = [0.6 0.5 0.9];
Kd = CGF(@(t) -sum(a(:).*log(1 - t(1:3))) - b*log(1 - t(4)), 4, 'domain', @(t) all(t < 1));
fex = exp(gammaln(sum(a) + b) - sum(gammaln(a)) - gammaln(b))*prod(r0.^(a - 1))*(1 + sum(r0))^(-(sum(a) + b));
f = ratio_density(Kd, r0, 'order', 5).f;
fprintf('three ratios: relative errors of the density, orders 1-5: %s\n', sprintf('%9.1e', f/fex - 1));

%% 8. A Studentized mean (Bao, Broda and Kan): the noncentral t distribution
% X ~ N(mu, 1), Y ~ chi^2_nu/nu independent, n pairs: Xbar/sqrt(Ybar) = T/sqrt(n), T noncentral t with
% n*nu degrees of freedom and noncentrality sqrt(n)*mu
mu = 1; nu = 3; n = 16;
Kt = sample_mean(CGF(@(t) mu*t(1) + t(1)^2/2 - nu/2*log(1 - 2*t(2)/nu), 2, 'domain', @(t) t(2) < nu/2), n);
x = 6;                                            % P(T <= 6) = P(Xbar/sqrt(Ybar) <= 6/sqrt(n))
F = studentized_tailprob(Kt, x/sqrt(n), 'order', 3, 'lower', true).F;
f = studentized_density(Kt, x/sqrt(n), 'order', 3).f/sqrt(n);
fprintf('noncentral t: P(T <= %g), orders 1-3: %s (exact %.10f)\n', x, sprintf('%.10f ', F), nctcdf(x, n*nu, sqrt(n)*mu));
fprintf('              density, orders 1-3:  %s (exact %.10f)\n', sprintf('%.10f ', f), nctpdf(x, n*nu, sqrt(n)*mu));
Q = studentized_quantile(Kt, 0.05, 'order', 3);
fprintf('              5%% quantile of T, order 3: %.8f (exact %.8f)\n', Q.q*sqrt(n), nctinv(0.05, n*nu, sqrt(n)*mu));
