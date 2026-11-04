function P = stud_jet_dd(K, s, t, N)
%STUD_JET_DD double-double Taylor polynomial (tpsdd in 2 variables) of degree N of K at (s, t)
P = taylordd(K, [s; t], N);
if isempty(P)
   error('CGF:domain', 'studentized: the point (s, t) = (%g, %g) is outside the domain of K', double(s), double(t));
end
end
