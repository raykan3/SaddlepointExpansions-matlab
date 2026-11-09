function Kd = uni_derivs(K, s, N)
%UNI_DERIVS The derivatives K^(k)(s), k = 0..N, of a univariate CGF (a column), or [] outside the
%   domain (or where K'' is not positive).
P = taylor(K, s, N);
if isempty(P), Kd = []; return, end
Kd = coef(P, (0:N)');
if ~(all(isfinite(Kd)) && Kd(3) > 0), Kd = []; end
end
