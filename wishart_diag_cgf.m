function K = wishart_diag_cgf(V, n)
%WISHART_DIAG_CGF The diagonal of a Wishart matrix W ~ W_d(n, V):
%   K(t) = -(n/2) log det(I - 2 V^{1/2} diag(t) V^{1/2}), the cgf of d quadratic forms z'e_ie_i'z
%   in z ~ N(0, V), summed over n independent copies (quadratic_forms_cgf with 'scale', n).
d = size(V, 1); E = cell(1, d);
for i = 1:d, Ei = zeros(d); Ei(i, i) = 1; E{i} = Ei; end
K = quadratic_forms_cgf(E, 'Sigma', V, 'scale', n);
end
