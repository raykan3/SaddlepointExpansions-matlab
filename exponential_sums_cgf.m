function K = exponential_sums_cgf(A, varargin)
%EXPONENTIAL_SUMS_CGF The mean of n independent copies of A'U + l, U a vector of independent unit
%   exponential variables, A m x d: K(t) = -n sum_i log(1 - a_i't/n) + l't, a_i the rows of A,
%   domain a_i't < n.  Options 'n' (default 1) and 'shift' l (default 0).  The examples of
%   Wang (1990) and Kolassa and Li (2010) in the paper are of this form.
p = inputParser; p.addParameter('n', 1); p.addParameter('shift', []); p.parse(varargin{:});
n = p.Results.n; l = p.Results.shift; d = size(A, 2);
if ~(n > 0), error('exponential_sums_cgf:n', 'exponential_sums_cgf: n must be positive'); end
if isempty(l)
   K = CGF(@(t) -n*sum(log(1 - A*t/n)), d, 'domain', @(t) all(A*t < n));
else
   l = reshape(l, [], 1);
   K = CGF(@(t) -n*sum(log(1 - A*t/n)) + l.'*t, d, 'domain', @(t) all(A*t < n));
end
end
