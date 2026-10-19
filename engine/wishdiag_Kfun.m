function v = wishdiag_Kfun(nu,t,V,n)
%WISHDIAG_KFUN Derivative of order nu (1 x 3) at t of the cumulant generating function of the
%   diagonal of a Wishart W_3(n,V) matrix, K(t) = -(n/2) log det(I - 2 V^{1/2} diag(t) V^{1/2}),
%   the cgf of three quadratic forms: D_nu = n 2^{|nu|-1} nu! tr(F_nu)/|nu|, F_nu = sum_i (D A_i)
%   F_{nu-e_i}, A_i = V^{1/2} e_i e_i' V^{1/2}, D = (I - 2 sum_i t_i A_i)^{-1}.  NaN outside the
%   domain.  Memoized: all derivatives with |nu| <= L are computed at the first call at a new
%   (t,V,n), L = 2 when only derivatives of order at most two are requested, otherwise 16 or
%   |nu|.  Used by tri_table (Table 7 of the paper).
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Saddlepoint expansions of arbitrary order for multivariate distribution functions"; free to use and modify with a citation of the paper.
persistent lastt lastV lastn lastL D
if sum(nu) <= 2, L = 2; else, L = max(16,sum(nu)); end
if isempty(lastt) || ~isequal(t,lastt) || ~isequal(V,lastV) || n ~= lastn || L > lastL
   D = wishdiag_all(t,L,V,n); lastt = t; lastV = V; lastn = n; lastL = L;
end
if isempty(D), v = NaN; return; end
v = D(nu(1)+1,nu(2)+1,nu(3)+1);
end

function D = wishdiag_all(t,L,V,n)
d = 3;
Vh = sqrtm(V);
A = cell(d,1); C = zeros(d);
for i = 1:d, A{i} = Vh(:,i)*Vh(i,:); C = C+t(i)*A{i}; end
Q = eye(d)-2*C;
[R,flag] = chol((Q+Q')/2);
if flag, D = []; return; end
Dm = inv(Q);
DA = cell(d,1); for i = 1:d, DA{i} = Dm*A{i}; end
D = zeros(L+1,L+1,L+1);
F = cell(L+1,L+1,L+1);
D(1,1,1) = -n*sum(log(diag(R)));
F{1,1,1} = eye(d);
for k = 1:L
   for a1 = k:-1:0
      for a2 = k-a1:-1:0
         a3 = k-a1-a2; nu = [a1 a2 a3];
         Fn = zeros(d);
         for i = 1:3
            if nu(i) > 0
               b = nu; b(i) = b(i)-1;
               Fn = Fn+DA{i}*F{b(1)+1,b(2)+1,b(3)+1};
            end
         end
         F{a1+1,a2+1,a3+1} = Fn;
         D(a1+1,a2+1,a3+1) = n*2^(k-1)*prod(factorial(nu))*trace(Fn)/k;
      end
   end
end
end
