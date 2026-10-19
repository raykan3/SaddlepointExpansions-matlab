function f = cgf_kfun(K, maxdeg)
%CGF_KFUN Kfun(nu, t) for the derivative interface of the engine: the derivative of order nu of K
%   at t, NaN outside the domain.  The Taylor polynomial at the last t is kept (degree 2 while only
%   derivatives of order <= 2 are requested, as in the saddlepoint iteration, otherwise maxdeg).
lastt = []; lastN = -1; lastp = []; valid = false;
f = @Kfun;
   function v = Kfun(nu, t)
      kk = sum(nu);
      if kk <= 2, N = 2; else, N = max(maxdeg, kk); end
      t = double(reshape(t, [], 1));
      if ~valid || ~isequal(t, lastt) || N > lastN
         lastp = taylor(K, t, N); lastt = t; lastN = N; valid = true;
      end
      if isempty(lastp), v = NaN; return; end
      v = coef(lastp, reshape(nu, 1, []));
   end
end
