function P = stud_jet(K, s, t, N)
%STUD_JET Taylor polynomial (a tps in 2 variables) of degree N of K at (s, t); an error outside
%   the domain of K.
P = taylor(K, [s; t], N);
if isempty(P)
   error('CGF:domain', 'studentized: the point (s, t) = (%g, %g) is outside the domain of K', s, t);
end
end
