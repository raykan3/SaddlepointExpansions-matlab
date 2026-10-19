function [Fup,pdf] = mvcc_exact(x,m,lam)
%MVCC_EXACT P[Q_1 >= x_1, ..., Q_d >= x_d] and the joint density of Q_j = X_j + X_0 (see
%   mvcc_Kfun) by one-dimensional integration over X_0 (relative tolerance 1e-12; the upper tails
%   are computed directly).
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Saddlepoint expansions of arbitrary order for multivariate distribution functions"; free to use and modify with a citation of the paper.
d = numel(x);
opts = {'AbsTol',0,'RelTol',1e-12};
f0 = @(s) ncx2pdf(s,m(d+1),lam(d+1));
   function v = fu(s)
      v = f0(s);
      for j = 1:d, v = v.*ncx2cdf(max(x(j)-s,0),m(j),lam(j),'upper'); end
   end
   function v = fd(s)
      v = f0(s);
      for j = 1:d, v = v.*ncx2pdf(max(x(j)-s,0),m(j),lam(j)); end
   end
br = sort(x(:)');
Fup = integral(@fu,0,br(end),opts{:},'Waypoints',br(1:end-1))+integral(@fu,br(end),Inf,opts{:});
pdf = integral(@fd,0,br(1),opts{:});
end
