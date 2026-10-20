function [pdf,Fup,tau,kappa,err,info] = spa_tri(Kfun,x,order,opts)
%SPA_TRI Trivariate saddlepoint approximations of arbitrary order: density and orthant probability
%   P[X1 >= x1, X2 >= x2, X3 >= x3] (Theorems 1 and 2 of the paper, d = 3).
%   [pdf,Fup,tau,kappa,err,info] = spa_tri(Kfun,x,order,opts)
%   Same inputs and outputs as spa_mv (x is 1 x 3).  The integrals I(nu) are computed by
%   deterministic quadrature in double precision (opts.method = 'quad': Gauss-Hermite quadrature of
%   the representation of Appendix B when all tau_j/sd(W_j) > 2.5, otherwise Gauss-Legendre
%   quadrature of the orthant integral in conditioned coordinates), and by the recursion of
%   Lemma 2 in variable precision ('vpa-recursion', formerly 'exact', or automatically with
%   opts.refine when a cancellation indicator exceeds 1e6 or a coefficient-error estimate
%   exceeds 1e-10; Kfun must then accept vpa arguments; the base normal orthant probabilities of
%   the recursion are still computed by quadrature).  For saddlepoints with negative components
%   use spa_orthant, which applies the inclusion-exclusion (17).
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Uniform saddlepoint expansions for multivariate tail probabilities"; free to use and modify with a citation of the paper.
if nargin < 4, opts = struct(); end
if numel(x) ~= 3, error('spa_tri: x must have three components'); end
[pdf,Fup,tau,kappa,err,info] = spa_mv(Kfun,x,order,opts);
end
