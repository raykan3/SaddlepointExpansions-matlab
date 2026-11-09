function K = logistic_regression_cgf(X, varargin)
%LOGISTIC_REGRESSION_CGF The sufficient statistic S = sum_i y_i x_i of a logistic regression, y_i
%   independent Bernoulli with log{p_i/(1 - p_i)} = x_i'beta, x_i' the rows of the n x d design
%   matrix X: K(t) = sum_i [log{1 + exp(x_i'(beta + t))} - log{1 + exp(x_i'beta)}], at
%   'beta', b (default zero).  With integer covariates S is a lattice variable ('lattice', true);
%   the conditional distribution of a component given the others, the basis of exact
%   conditional inference, is conditional_tailprob and conditional_density with 'lattice', true,
%   and it does not depend on the coefficients of the conditioning components.
pr = inputParser; pr.addParameter('beta', []); pr.parse(varargin{:});
Xm = double(X); d = size(Xm, 2);
b = pr.Results.beta;
if isempty(b), b = zeros(d, 1); end
b = reshape(double(b), [], 1);
if numel(b) ~= d, error('logistic_regression_cgf:beta', 'logistic_regression_cgf: beta must have length %d', d); end
eta = Xm*b;
c0 = sum(log1pexp(eta));
K = CGF(@(t) sum(log1pexp(eta + Xm*t)) - c0, d);
end

function v = log1pexp(z)
%  log(1 + e^z) elementwise, for numbers and series, without overflow for large constant terms:
%  z + log(1 + e^-z) where the constant term is positive, log(1 + e^z) elsewhere
sg = 2*(double(z) > 0) - 1;
v = double(sg > 0).*z + log(1 + exp(-sg.*z));
end
