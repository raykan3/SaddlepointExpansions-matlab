classdef (InferiorClasses = {?dd}) tpsdd
%TPSDD Truncated multivariate power series with double-double coefficients (about 32 digits).
%   The double-double counterpart of tps, with the same interface: a cumulant generating function
%   written for tps can be evaluated at t = t0 + M*v with v = tpsdd.vars(k, N) (t0 a dd or double
%   vector) to give its Taylor polynomial with double-double coefficients.  Products of series are
%   formed with error-free transformations and compensated (pairwise) sums; elementary functions of
%   a series p = p0 + q are evaluated by composition, sum_j f^(j)(p0)/j! q^j, with the derivatives at
%   the constant term computed in variable precision (40 digits) and rounded to double-double.
%      v = tpsdd.vars(k, N)       variables;  p = tpsdd.const(c, k, N)  constants (c double or dd)
%      coef(p, nu)                nu! times the coefficient of v^nu, a dd array
%      double(p), ddconst(p)      the constant terms in double and in double-double
%   CGF.taylordd(K, t, N) gives the double-double Taylor polynomial of a cumulant generating
%   function; studentized_density and studentized_tailprob use it ('precision', 'dd').
%
%   Raymond Kan and Simon A. Broda, 2026.  Part of SaddlepointExpansions for MATLAB.

properties (SetAccess = private)
   H        % high parts of the coefficients: ncoef x numel(array)
   L        % low parts
   sz       % size of the array
   k        % number of variables
   N        % truncation degree
end

methods
   function p = tpsdd(H, L, sz, k, N)
      if nargin == 0, H = 0; L = 0; sz = [1 1]; k = 1; N = 0; end
      p.H = H; p.L = L; p.sz = sz; p.k = k; p.N = N;
   end

   %---------------------------------------------------------------- size and indexing
   function varargout = size(p, dim)
      if nargin == 2, varargout{1} = p.sz(dim); return; end
      if nargout <= 1, varargout{1} = p.sz; else, for i = 1:nargout, if i <= numel(p.sz), varargout{i} = p.sz(i); else, varargout{i} = 1; end, end, end
   end
   function n = numel(p, varargin), n = prod(p.sz); end
   function n = length(p), if any(p.sz == 0), n = 0; else, n = max(p.sz); end, end
   function n = ndims(p), n = numel(p.sz); end
   function tf = isempty(p), tf = any(p.sz == 0); end
   function tf = isscalar(p), tf = prod(p.sz) == 1; end
   function tf = isvector(p), tf = numel(p.sz) == 2 && min(p.sz) == 1; end
   function n = numArgumentsFromSubscript(~, ~, ~), n = 1; end
   function e = end(p, kdim, ndim)
      if ndim == 1, e = prod(p.sz); elseif kdim < ndim, e = p.sz(kdim); else, e = prod(p.sz(kdim:end)); end
   end
   function varargout = subsref(p, s)
      switch s(1).type
         case '()'
            map = reshape(1:prod(p.sz), p.sz); m = map(s(1).subs{:});
            r = tpsdd(p.H(:, m(:)), p.L(:, m(:)), size(m), p.k, p.N);
            if numel(s) > 1, [varargout{1:nargout}] = subsref(r, s(2:end)); else, varargout{1} = r; end
         otherwise
            [varargout{1:nargout}] = builtin('subsref', p, s);
      end
   end
   function p = subsasgn(p, s, rhs)
      if ~strcmp(s(1).type, '()') || numel(s) > 1, p = builtin('subsasgn', p, s, rhs); return; end
      if ~isa(p, 'tpsdd'), r = rhs; p = tpsdd.lift(p, r.k, r.N); end
      if isa(rhs, 'tpsdd'), [p, rhs] = tpsdd.match(p, rhs); else, rhs = tpsdd.lift(rhs, p.k, p.N); end
      n = prod(p.sz); M = reshape(1:n, p.sz); M(s(1).subs{:}) = -1; tgt = find(M == -1);
      nC = size(p.H, 1); Hn = zeros(nC, numel(M)); Ln = Hn;
      old = M > 0; Hn(:, old) = p.H(:, M(old)); Ln(:, old) = p.L(:, M(old));
      if size(rhs.H, 2) == 1, Hn(:, tgt) = repmat(rhs.H, 1, numel(tgt)); Ln(:, tgt) = repmat(rhs.L, 1, numel(tgt));
      elseif size(rhs.H, 2) == numel(tgt), Hn(:, tgt) = rhs.H; Ln(:, tgt) = rhs.L;
      else, error('tpsdd:subsasgn', 'Assignment dimension mismatch.');
      end
      p.H = Hn; p.L = Ln; p.sz = size(M);
   end
   function r = reshape(p, varargin), map = reshape(1:prod(p.sz), varargin{:}); r = tpsdd(p.H(:, map(:)), p.L(:, map(:)), size(map), p.k, p.N); end
   function r = transpose(p), map = reshape(1:prod(p.sz), p.sz).'; r = tpsdd(p.H(:, map(:)), p.L(:, map(:)), size(map), p.k, p.N); end
   function r = ctranspose(p), r = transpose(p); end
   function r = horzcat(varargin), r = tpsdd.catn(2, varargin{:}); end
   function r = vertcat(varargin), r = tpsdd.catn(1, varargin{:}); end
   function r = cat(dim, varargin), r = tpsdd.catn(dim, varargin{:}); end
   function r = repmat(p, varargin), map = repmat(reshape(1:prod(p.sz), p.sz), varargin{:}); r = tpsdd(p.H(:, map(:)), p.L(:, map(:)), size(map), p.k, p.N); end

   %---------------------------------------------------------------- conversion
   function d = double(p), d = reshape(p.H(1, :), p.sz); end
   function c = ddconst(p), c = dd(reshape(p.H(1, :), p.sz), reshape(p.L(1, :), p.sz)); end
   function v = coef(p, nu)
      %COEF nu! times the coefficient of v^nu of the scalar series p, as a dd (one row of nu per value)
      T = tps.tables(p.k, p.N); nu = reshape(nu, [], p.k); m = size(nu, 1);
      h = zeros(m, 1); l = h;
      for i = 1:m
         if sum(nu(i, :)) > p.N, h(i) = NaN; continue; end
         idx = T.lookup(nu(i, :)*T.base + 1);
         [fh, fl] = tpsdd.factdd(nu(i, :));
         [h(i), l(i)] = dd.mul(p.H(idx, 1), p.L(idx, 1), fh, fl);
      end
      v = dd(h, l);
   end
   function c = rawcoef(p, idx), c = dd(p.H(idx, 1), p.L(idx, 1)); end
   function disp(p)
      fprintf('  tpsdd array of size %s in %d variable(s), degree %d; constant terms:\n', mat2str(p.sz), p.k, p.N);
      disp(double(p));
   end
   function tf = isfinite(p), tf = reshape(all(isfinite(p.H), 1), p.sz); end
   function tf = isreal(~), tf = true; end
   function r = real(p), r = p; end

   %---------------------------------------------------------------- arithmetic
   function r = plus(a, b)
      [a, b, map] = tpsdd.bcast(a, b);
      [h, l] = dd.add(a.H(:, map(1, :)), a.L(:, map(1, :)), b.H(:, map(2, :)), b.L(:, map(2, :)));
      r = tpsdd(h, l, tpsdd.bsz(a, b), a.k, a.N);
   end
   function r = minus(a, b), r = plus(a, uminus(tpsdd.liftany(b, a))); end
   function r = uminus(a), r = tpsdd(-a.H, -a.L, a.sz, a.k, a.N); end
   function r = uplus(a), r = a; end
   function r = times(a, b)
      if ~isa(a, 'tpsdd'), r = tpsdd.scale(b, a); return; end
      if ~isa(b, 'tpsdd'), r = tpsdd.scale(a, b); return; end
      [a, b, map] = tpsdd.bcast(a, b);
      [h, l] = tpsdd.mulC(a.H(:, map(1, :)), a.L(:, map(1, :)), b.H(:, map(2, :)), b.L(:, map(2, :)), a.k, a.N);
      r = tpsdd(h, l, tpsdd.bsz(a, b), a.k, a.N);
   end
   function r = mtimes(a, b)
      if isscalar(a) || isscalar(b), r = times(a, b); return; end
      if ~isa(a, 'tpsdd')                                   % constant matrix * series matrix
         A = dd(a); [m, n] = size(A); pp = b.sz(2); nc = size(b.H, 1);
         H = zeros(nc, m*pp); Lo = H;
         for i = 1:m, for j = 1:pp
            acc = tpsdd.lift(0, b.k, b.N);
            for q = 1:n
               acc = plus(acc, tpsdd.scale(subsref(b, substruct('()', {q, j})), subsref(A, substruct('()', {i, q}))));
            end
            H(:, i + (j-1)*m) = acc.H; Lo(:, i + (j-1)*m) = acc.L;
         end, end
         r = tpsdd(H, Lo, [m pp], b.k, b.N);
      elseif ~isa(b, 'tpsdd')                               % series matrix * constant matrix
         r = transpose(mtimes(transpose(b), transpose(a)));
      else
         [a, b] = tpsdd.match(a, b);
         m = a.sz(1); n = a.sz(2); pp = b.sz(2); nc = size(a.H, 1);
         H = zeros(nc, m*pp); Lo = H;
         for i = 1:m, for j = 1:pp
            acc = tpsdd.lift(0, a.k, a.N);
            for q = 1:n
               acc = plus(acc, times(subsref(a, substruct('()', {i, q})), subsref(b, substruct('()', {q, j}))));
            end
            H(:, i + (j-1)*m) = acc.H; Lo(:, i + (j-1)*m) = acc.L;
         end, end
         r = tpsdd(H, Lo, [m pp], a.k, a.N);
      end
   end
   function r = rdivide(a, b)
      if ~isa(b, 'tpsdd'), r = tpsdd.scale(a, rdivide(1, dd(b))); return; end
      r = times(a, power(b, -1));
   end
   function r = ldivide(a, b), r = rdivide(b, a); end
   function r = mrdivide(a, b)
      if isscalar(b), r = rdivide(a, b); else, error('tpsdd:mrdivide', 'Only division by a scalar is supported.'); end
   end
   function r = power(a, b)
      if isa(b, 'tpsdd'), if ~isa(a, 'tpsdd'), a = tpsdd.lift(a, b.k, b.N); end, r = exp(times(b, log(a))); return; end
      if ~isa(a, 'tpsdd'), error('tpsdd:power', 'Unexpected numeric base.'); end
      if isa(b, 'dd'), b = double(b); end
      if isscalar(b) && b == round(b) && b >= 0
         r = tpsdd.lift(ones(a.sz), a.k, a.N); x = a; e = b;
         while e > 0
            if mod(e, 2), r = times(r, x); end
            e = floor(e/2); if e > 0, x = times(x, x); end
         end
         return
      end
      if ~isscalar(b), error('tpsdd:power', 'Only scalar exponents are supported.'); end
      a0 = a.H(1, :);
      if any(a0 == 0) || (b ~= round(b) && any(a0 < 0)), error('tps:domain', 'power: not analytic at the constant term'); end
      r = tpsdd.composef(a, @(x, j) nchoosek_sym(b, j).*x.^(b - j));
   end
   function r = mpower(a, b)
      if isscalar(a), r = power(a, b); else, error('tpsdd:mpower', 'Only scalar powers are supported.'); end
   end
   function r = sqrt(a), if any(a.H(1, :) <= 0), error('tps:domain', 'sqrt: not analytic at a nonpositive constant term'); end, r = power(a, 0.5); end
   function r = exp(a), r = tpsdd.composef(a, @(x, j) exp(x)./factorial(sym(j))); end
   function r = expm1(a), r = exp(a); c = expm1(ddconst(subsref(a, substruct('()', {':'})))); r.H(1, :) = c.hi(:).'; r.L(1, :) = c.lo(:).'; end
   function r = log(a)
      if any(a.H(1, :) <= 0), error('tps:domain', 'log: not analytic at a nonpositive constant term'); end
      r = tpsdd.composef(a, @(x, j) logder(x, j));
   end
   function r = log1p(a)
      if any(a.H(1, :) <= -1), error('tps:domain', 'log1p: not analytic'); end
      r = log(plus(a, 1));
   end
   function r = sin(a), r = tpsdd.composef(a, @(x, j) sin(x + j*sym(pi)/2)./factorial(sym(j))); end
   function r = cos(a), r = tpsdd.composef(a, @(x, j) cos(x + j*sym(pi)/2)./factorial(sym(j))); end
   function r = sinh(a), r = tpsdd.composef(a, @(x, j) (mod(j, 2) == 0).*sinh(x)./factorial(sym(j)) + (mod(j, 2) == 1).*cosh(x)./factorial(sym(j))); end
   function r = cosh(a), r = tpsdd.composef(a, @(x, j) (mod(j, 2) == 0).*cosh(x)./factorial(sym(j)) + (mod(j, 2) == 1).*sinh(x)./factorial(sym(j))); end
   function r = tanh(a), e = exp(2*a); r = rdivide(minus(e, 1), plus(e, 1)); end
   function r = tan(a), r = rdivide(sin(a), cos(a)); end
   function r = atan(a), r = tpsdd.composef(a, @(x, j) atander(x, j)); end
   function r = abs(a)
      a0 = a.H(1, :); if any(a0 == 0), error('tps:domain', 'abs: not analytic at zero'); end
      r = tpsdd.scale(a, reshape(sign(a0), a.sz));
   end
   function r = sum(a, dim)
      if nargin < 2, dim = find(a.sz ~= 1, 1); if isempty(dim), dim = 1; end, end
      sz = a.sz; if dim > numel(sz), r = a; return; end
      idx = repmat({':'}, 1, numel(sz)); idx{dim} = 1; r = subsref(a, substruct('()', idx));
      for i = 2:sz(dim), idx{dim} = i; r = plus(r, subsref(a, substruct('()', idx))); end
   end
   function r = prod(a, dim)
      if nargin < 2, dim = find(a.sz ~= 1, 1); if isempty(dim), dim = 1; end, end
      sz = a.sz; idx = repmat({':'}, 1, numel(sz)); idx{dim} = 1; r = subsref(a, substruct('()', idx));
      for i = 2:sz(dim), idx{dim} = i; r = times(r, subsref(a, substruct('()', idx))); end
   end
   function r = dot(a, b), r = sum(times(a, b)); end
   function r = sinhc(a), r = tpsdd.composef(a, @(x, j) sinhcder(x, j)); end
   function r = sincu(a), r = tpsdd.composef(a, @(x, j) sincuder(x, j)); end

   %---------------------------------------------------------------- comparisons (constant terms)
   function r = lt(a, b), r = tpsdd.c0(a) < tpsdd.c0(b); end
   function r = le(a, b), r = tpsdd.c0(a) <= tpsdd.c0(b); end
   function r = gt(a, b), r = tpsdd.c0(a) > tpsdd.c0(b); end
   function r = ge(a, b), r = tpsdd.c0(a) >= tpsdd.c0(b); end
   function r = eq(a, b), r = tpsdd.c0(a) == tpsdd.c0(b); end
   function r = ne(a, b), r = tpsdd.c0(a) ~= tpsdd.c0(b); end
end

methods (Static)
   function v = vars(k, N)
      T = tps.tables(k, N); H = zeros(T.n, k);
      for i = 1:k, e = zeros(1, k); e(i) = 1; if N >= 1, H(T.lookup(e*T.base + 1), i) = 1; end, end
      v = tpsdd(H, zeros(size(H)), [k 1], k, N);
   end
   function p = const(c, k, N), p = tpsdd.lift(c, k, N); end
   function p = fromtps(q)
      %FROMTPS a tps (double coefficients) as a tpsdd
      p = tpsdd(q.C, zeros(size(q.C)), size(q), q.k, q.N);
   end
   function G = groups(k, N)
      %GROUPS the products of the multiplication table sorted by target coefficient: an n x maxlen
      %   index matrix into the products, padded with numel(products) + 1 (a zero), cached
      persistent cache
      if isempty(cache), cache = containers.Map(); end
      key = sprintf('%d_%d', k, N);
      if isKey(cache, key), G = cache(key); return; end
      T = tps.tables(k, N);
      [tgt, prodidx] = find(T.S);                      % product prodidx contributes to tgt
      np = numel(T.I);
      [tgt, o] = sort(tgt); prodidx = prodidx(o);
      cnt = accumarray(tgt, 1, [T.n 1]); maxlen = max(cnt);
      start = [0; cumsum(cnt(1:end-1))];
      pos = (1:numel(tgt))' - start(tgt);
      G = repmat(np + 1, T.n, maxlen);
      G(sub2ind(size(G), tgt, pos)) = prodidx;
      cache(key) = G;
   end
end

methods (Static, Hidden)
   function p = lift(c, k, N)
      T = tps.tables(k, N); c = dd(c);
      H = zeros(T.n, numel(c)); Lo = H; H(1, :) = c.hi(:).'; Lo(1, :) = c.lo(:).';
      p = tpsdd(H, Lo, size(c), k, N);
   end
   function b = liftany(b, a), if ~isa(b, 'tpsdd'), b = tpsdd.lift(b, a.k, a.N); end, end
   function c = c0(a)
      if isa(a, 'tpsdd'), c = ddconst(a); elseif isa(a, 'dd'), c = a; else, c = dd(a); end
   end
   function [a, b] = match(a, b)
      if ~isa(a, 'tpsdd'), a = tpsdd.lift(a, b.k, b.N); end
      if ~isa(b, 'tpsdd'), b = tpsdd.lift(b, a.k, a.N); end
      if a.k ~= b.k || a.N ~= b.N, error('tps:mismatch', 'Series in different numbers of variables or degrees.'); end
   end
   function [a, b, map] = bcast(a, b)
      [a, b] = tpsdd.match(a, b);
      ia = reshape(1:prod(a.sz), a.sz); ib = reshape(1:prod(b.sz), b.sz);
      ma = ia + 0*ib; mb = ib + 0*ia; map = [ma(:)'; mb(:)'];
   end
   function sz = bsz(a, b), sz = size(zeros(a.sz) + zeros(b.sz)); end
   function r = scale(a, s)
      % series array a times a constant (double or dd, scalar or array)
      s = dd(s);
      ia = reshape(1:prod(a.sz), a.sz); m = ia + 0*s.hi; sh = s.hi + 0*ia; sl = s.lo + 0*ia;
      nc = size(a.H, 1);
      [h, l] = dd.mul(a.H(:, m(:)), a.L(:, m(:)), repmat(sh(:).', nc, 1), repmat(sl(:).', nc, 1));
      r = tpsdd(h, l, size(m), a.k, a.N);
   end
   function [h, l] = mulC(Ah, Al, Bh, Bl, k, N)
      T = tps.tables(k, N); G = tpsdd.groups(k, N);
      m = size(Ah, 2); np = numel(T.I);
      [ph, pl] = dd.mul(Ah(T.I, :), Al(T.I, :), Bh(T.J, :), Bl(T.J, :));
      ph(np + 1, :) = 0; pl(np + 1, :) = 0;
      [n, maxlen] = size(G);
      h = zeros(n, m); l = h;
      for c = 1:m
         Mh = reshape(ph(G, c), n, maxlen); Ml = reshape(pl(G, c), n, maxlen);
         [sh, sl] = dd.colsum(Mh.', Ml.');
         h(:, c) = sh.'; l(:, c) = sl.';
      end
   end
   function r = composef(a, fder)
      % f(a0 + q) = sum_j fc_j q^j, fc_j = f^(j)(a0)/j! = fder(a0, j) evaluated in variable precision
      N = a.N; m = prod(a.sz); fc = cell(N + 1, 1);
      a0 = vpa(dd(a.H(1, :), a.L(1, :)), 40);
      for j = 0:N, fc{j + 1} = dd(vpa(fder(a0, j), 40)); end
      qH = a.H; qL = a.L; qH(1, :) = 0; qL(1, :) = 0;
      % Horner: R = fc_N; R = R q + fc_j
      Rh = zeros(size(qH)); Rl = Rh; Rh(1, :) = fc{N + 1}.hi; Rl(1, :) = fc{N + 1}.lo;
      for j = N-1:-1:0
         [Rh, Rl] = tpsdd.mulC(Rh, Rl, qH, qL, a.k, N);
         [Rh(1, :), Rl(1, :)] = dd.add(Rh(1, :), Rl(1, :), fc{j + 1}.hi, fc{j + 1}.lo);
      end
      r = tpsdd(Rh, Rl, a.sz, a.k, a.N);
      if m == 0, r = a; end
   end
   function [h, l] = factdd(nu)
      % nu! = prod(nu_i!) exactly as a double-double (exact while it fits in 106 bits)
      h = 1; l = 0;
      for n = nu(:)', for i = 2:n, [h, l] = dd.mul(h, l, i, 0); end, end
   end
   function r = catn(dim, varargin)
      ref = [];
      for i = 1:numel(varargin), if isa(varargin{i}, 'tpsdd'), ref = varargin{i}; break; end, end
      maps = cell(size(varargin)); Hs = maps; Ls = maps; off = 0;
      for i = 1:numel(varargin)
         x = varargin{i}; if ~isa(x, 'tpsdd'), x = tpsdd.lift(x, ref.k, ref.N); end
         [x, ~] = tpsdd.match(x, ref);
         n = prod(x.sz); maps{i} = off + reshape(1:n, x.sz); Hs{i} = x.H; Ls{i} = x.L; off = off + n;
      end
      map = cat(dim, maps{:}); H = [Hs{:}]; Lo = [Ls{:}];
      r = tpsdd(H(:, map(:)), Lo(:, map(:)), size(map), ref.k, ref.N);
   end
end
end

%---------------------------------------------------------------- derivative helpers (sym, elementwise)
function c = nchoosek_sym(b, j)
c = sym(1); bb = sym(b);
for i = 1:j, c = c*(bb - i + 1)/i; end
end
function d = logder(x, j)
if j == 0, d = log(x); else, d = (-1)^(j+1)./(j*x.^j); end
end
function d = atander(x, j)
% the j-th Taylor coefficient of atan at x, from the series of 1/(1 + (x+q)^2) integrated
if j == 0, d = atan(x); return; end
d = sym(zeros(size(x)));
for c = 1:numel(x)
   g = sym(zeros(j, 1)); den = [1 + x(c)^2, 2*x(c), sym(1)];
   for i = 0:j-1
      s = sym(i == 0); for l = 1:min(i, 2), s = s - den(l+1)*g(i-l+1); end
      g(i+1) = s/den(1);
   end
   d(c) = g(j)/j;
end
end
function d = sinhcder(x, j)
% j-th Taylor coefficient of sinh(x)/x, by differentiating symbolically
z = sym('z'); f = sinh(z)/z;
d = sym(zeros(size(x)));
for c = 1:numel(x)
   if x(c) == 0, if mod(j, 2) == 0, d(c) = 1/factorial(sym(j + 1)); end
   else, d(c) = subs(diff(f, z, j), z, x(c))/factorial(sym(j)); end
end
end
function d = sincuder(x, j)
z = sym('z'); f = sin(z)/z;
d = sym(zeros(size(x)));
for c = 1:numel(x)
   if x(c) == 0, if mod(j, 2) == 0, d(c) = (-1)^(j/2)/factorial(sym(j + 1)); end
   else, d(c) = subs(diff(f, z, j), z, x(c))/factorial(sym(j)); end
end
end
