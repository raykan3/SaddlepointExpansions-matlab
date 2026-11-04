function v = lin_exp_Kfun(nu,t,A,n,lin)
%LIN_EXP_KFUN Derivative of order nu at t of the cumulant generating function of the mean of n
%   independent copies of A'U + lin, U a vector of independent unit exponential variables:
%      K(t) = n sum_i [-log(1 - a_i't/n)] + lin't,
%   a_i the rows of A (m x d).  nu: 1 x d multi-index; t: 1 x d; lin: 1 x d (default zero).
%   Returns NaN outside the domain.  Used by wang_table and kl_table (Tables 1 and 2 of the paper and S4 to S7 of the Supplementary Material).
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Uniform saddlepoint expansions for multivariate tail probabilities"; free to use and modify with a citation of the paper.
if nargin < 5, lin = zeros(size(t)); end
k = sum(nu);
z = 1-A*t(:)/n;
if any(z <= 0), v = NaN; return; end
if k == 0
   v = -n*sum(log(z))+lin*t(:);
   return
end
v = 0;
for i = 1:size(A,1)
   an = prod(A(i,:).^nu);
   if an == 0, continue; end
   v = v+n*factorial(k-1)*an/n^k/z(i)^k;
end
if k == 1, v = v+lin*nu(:); end
end
