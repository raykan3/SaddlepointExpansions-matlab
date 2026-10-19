function f = cgf_kwhite(K, maxdeg)
%CGF_KWHITE Kwhite(nu, u, M) for the engine (opts.Kwhite): the derivative of order nu at u of
%   K_Y(u) = K(M u), from the Taylor polynomial of K(M u + M v) in v; M may be rectangular
%   (d x k) for the marginal terms of the inclusion-exclusion.  Kept at the last (u, M).
lastu = []; lastM = []; lastp = []; lastN = -1; valid = false;
f = @Kwhite;
   function v = Kwhite(nu, u, M)
      if isa(u, 'sym') || isa(M, 'sym'), error('CGF:precision', 'the Taylor class works in double precision only'); end
      u = double(reshape(u, [], 1)); M = double(M);
      N = max(maxdeg, sum(nu));
      if ~valid || ~isequal(u, lastu) || ~isequal(M, lastM) || N > lastN
         lastp = taylor(K, M*u, N, M); lastu = u; lastM = M; lastN = N; valid = true;
      end
      if isempty(lastp), v = NaN; return; end
      v = coef(lastp, reshape(nu, 1, []));
   end
end
