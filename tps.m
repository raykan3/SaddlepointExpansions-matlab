classdef tps
%TPS Truncated multivariate power series (Taylor-mode automatic differentiation).
%   An array of polynomials in k variables v = (v_1,...,v_k), truncated at total degree N.  A
%   cumulant generating function written as an ordinary MATLAB function of a column vector t
%   (arithmetic, matrix products with constant matrices, exp, log, sqrt, powers, sin, cos,
%   sinh, cosh, tanh, atan, sum, prod, ...) can be evaluated at t = t0 + M*v with v = tps.vars(k,N)
%   to give the Taylor polynomial of K(t0 + M v) of degree N, from which all derivatives of
%   order at most N follow:  K^(nu)(t0) = coef(p, nu) (the coefficient of v^nu times nu!).
%
%   v = tps.vars(k,N)        k x 1 array of the variables v_1, ..., v_k
%   p = tps.const(c,k,N)     constant series (c may be an array)
%   coef(p,nu)               nu! times the coefficient of v^nu (p a scalar series)
%   double(p)                constant terms
%
%   Functions of a series p = p0 + q (q without constant term) are evaluated by composition,
%   f(p) = sum_j f^(j)(p0)/j! q^j, j = 0..N.  Where f is not analytic at p0 (log or sqrt of a
%   nonpositive constant term, a noninteger power of a negative one, division by zero) an error
%   with identifier 'tps:domain' is raised; CGF treats it as "outside the domain of K".
%   Comparisons (<, <=, >, >=, ==, ~=) compare the constant terms, so that code with tests on
%   the argument can be differentiated.  Everything is in double precision.
%
%   Raymond Kan and Simon A. Broda, 2026. MATLAB counterpart of SaddlepointExpansions.jl; free to
%   use and modify with a citation of Broda and Kan.

properties (SetAccess = private)
   C        % coefficients: ncoef x numel(array), column j holds element j (column-major)
   sz       % size of the array
   k        % number of variables
   N        % truncation degree
end

methods
   function p = tps(C, sz, k, N)
      if nargin == 0, C = 0; sz = [1 1]; k = 1; N = 0; end
      p.C = C; p.sz = sz; p.k = k; p.N = N;
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
      if ndim == 1, e = prod(p.sz);
      elseif kdim < ndim, e = p.sz(kdim);
      else, e = prod(p.sz(kdim:end));
      end
   end
   function varargout = subsref(p, s)
      switch s(1).type
         case '()'
            map = reshape(1:prod(p.sz), p.sz);
            m = map(s(1).subs{:});
            r = tps(p.C(:, m(:)), size(m), p.k, p.N);
            if numel(s) > 1, [varargout{1:nargout}] = subsref(r, s(2:end)); else, varargout{1} = r; end
         otherwise
            [varargout{1:nargout}] = builtin('subsref', p, s);
      end
   end
   function p = subsasgn(p, s, rhs)
      if ~strcmp(s(1).type, '()') || numel(s) > 1, p = builtin('subsasgn', p, s, rhs); return; end
      if ~isa(p, 'tps'), r = tps.like(rhs); p = tps.lift(p, r.k, r.N); end   % assignment into a double array
      if isa(rhs, 'tps'), [p, rhs] = tps.match(p, rhs); else, rhs = tps.lift(rhs, p.k, p.N); end
      n = prod(p.sz);
      M = reshape(1:n, p.sz);
      M(s(1).subs{:}) = -1;
      tgt = find(M == -1);
      nC = size(p.C, 1);
      Cn = zeros(nC, numel(M));
      old = M > 0; Cn(:, old) = p.C(:, M(old));
      if numel(rhs.C(1, :)) == 1, Cn(:, tgt) = repmat(rhs.C, 1, numel(tgt));
      elseif size(rhs.C, 2) == numel(tgt), Cn(:, tgt) = rhs.C;
      else, error('tps:subsasgn', 'Assignment dimension mismatch.');
      end
      p.C = Cn; p.sz = size(M);
   end
   function r = reshape(p, varargin)
      map = reshape(1:prod(p.sz), varargin{:});
      r = tps(p.C(:, map(:)), size(map), p.k, p.N);
   end
   function r = transpose(p), map = reshape(1:prod(p.sz), p.sz).'; r = tps(p.C(:, map(:)), size(map), p.k, p.N); end
   function r = ctranspose(p), r = transpose(p); end
   function r = horzcat(varargin), r = tps.catn(2, varargin{:}); end
   function r = vertcat(varargin), r = tps.catn(1, varargin{:}); end
   function r = cat(dim, varargin), r = tps.catn(dim, varargin{:}); end
   function r = repmat(p, varargin)
      map = repmat(reshape(1:prod(p.sz), p.sz), varargin{:});
      r = tps(p.C(:, map(:)), size(map), p.k, p.N);
   end
   function r = diag(p)
      map = diag(reshape(1:prod(p.sz), p.sz));
      if min(p.sz) == 1   % vector -> diagonal matrix
         n = numel(map); r = tps.lift(zeros(n), p.k, p.N);
         r.C(:, (0:n-1)*n + (1:n)) = p.C;
      else
         r = tps(p.C(:, map(:)), size(map), p.k, p.N);
      end
   end

   %---------------------------------------------------------------- conversion and display
   function d = double(p), d = reshape(p.C(1, :), p.sz); end
   function v = coef(p, nu)
      %COEF nu! times the coefficient of v^nu of the scalar series p (= the derivative of
      %   order nu at the expansion point).  nu: 1 x k nonnegative integers (one row per
      %   multi-index for several).
      T = tps.tables(p.k, p.N);
      nu = reshape(nu, [], p.k);
      v = zeros(size(nu, 1), 1);
      for i = 1:size(nu, 1)
         if sum(nu(i, :)) > p.N, v(i) = NaN; continue; end
         idx = T.lookup(nu(i, :)*T.base + 1);
         v(i) = p.C(idx, 1)*T.fact(idx);
      end
   end
   function disp(p)
      fprintf('  tps array of size %s in %d variable(s), degree %d; constant terms:\n', mat2str(p.sz), p.k, p.N);
      disp(double(p));
   end
   function tf = isfinite(p), tf = reshape(all(isfinite(p.C), 1), p.sz); end
   function tf = isreal(~), tf = true; end
   function r = real(p), r = p; end

   %---------------------------------------------------------------- arithmetic
   function r = plus(a, b), [a, b, map] = tps.bcast(a, b); r = tps(a.C(:, map(1, :)) + b.C(:, map(2, :)), tps.bsz(a, b), a.k, a.N); end
   function r = minus(a, b), [a, b, map] = tps.bcast(a, b); r = tps(a.C(:, map(1, :)) - b.C(:, map(2, :)), tps.bsz(a, b), a.k, a.N); end
   function r = uminus(a), r = tps(-a.C, a.sz, a.k, a.N); end
   function r = uplus(a), r = a; end
   function r = times(a, b)
      if isnumeric(a), r = tps.scale(b, a); return; end
      if isnumeric(b), r = tps.scale(a, b); return; end
      [a, b, map] = tps.bcast(a, b);
      r = tps(tps.mulC(a.C(:, map(1, :)), b.C(:, map(2, :)), a.k, a.N), tps.bsz(a, b), a.k, a.N);
   end
   function r = mtimes(a, b)
      if isscalar(a) || isscalar(b), r = times(a, b); return; end
      if isnumeric(a)                                   % double matrix * series matrix
         [m, n] = size(a); pp = b.sz(2); nc = size(b.C, 1);
         C = zeros(nc, m*pp);
         for j = 1:pp, C(:, (j-1)*m+(1:m)) = b.C(:, (j-1)*n+(1:n))*a.'; end
         r = tps(C, [m pp], b.k, b.N);
      elseif isnumeric(b)                               % series matrix * double matrix
         m = a.sz(1); n = a.sz(2); pp = size(b, 2); nc = size(a.C, 1);
         C = reshape(reshape(a.C, nc*m, n)*b, nc, m*pp);
         r = tps(C, [m pp], a.k, a.N);
      else
         [a, b] = tps.match(a, b);
         m = a.sz(1); n = a.sz(2); pp = b.sz(2); nc = size(a.C, 1);
         C = zeros(nc, m*pp);
         for i = 1:m, for j = 1:pp
            ai = a.C(:, i + (0:n-1)*m); bj = b.C(:, (j-1)*n + (1:n));
            C(:, i + (j-1)*m) = sum(tps.mulC(ai, bj, a.k, a.N), 2);
         end, end
         r = tps(C, [m pp], a.k, a.N);
      end
   end
   function r = rdivide(a, b)
      if isnumeric(b), r = tps.scale(a, 1./b); return; end
      r = times(a, power(b, -1));
   end
   function r = ldivide(a, b), r = rdivide(b, a); end
   function r = mrdivide(a, b)
      if isscalar(b), r = rdivide(a, b); else, error('tps:mrdivide', 'Only division by a scalar is supported.'); end
   end
   function r = power(a, b)
      if isa(b, 'tps')                                  % a.^b = exp(b log a)
         if isnumeric(a), a = tps.lift(a, b.k, b.N); end
         r = exp(times(b, log(a))); return;
      end
      if isnumeric(a), error('tps:power', 'Unexpected numeric base.'); end
      if isscalar(b) && b == round(b) && b >= 0          % nonnegative integer: repeated multiplication
         r = tps.lift(ones(a.sz), a.k, a.N); x = a; e = b;
         while e > 0
            if mod(e, 2), r = times(r, x); end
            e = floor(e/2); if e > 0, x = times(x, x); end
         end
         return
      end
      if ~isscalar(b), error('tps:power', 'Only scalar exponents are supported.'); end
      a0 = a.C(1, :);
      if any(a0 == 0) || (b ~= round(b) && any(a0 < 0)), error('tps:domain', 'power: not analytic at the constant term'); end
      j = (0:a.N)';
      bin = ones(a.N+1, 1); for i = 1:a.N, bin(i+1) = bin(i)*(b - i + 1)/i; end
      fc = bin .* (a0 .^ (b - j));
      r = tps.compose(a, fc);
   end
   function r = mpower(a, b)
      if isscalar(a), r = power(a, b); else, error('tps:mpower', 'Only scalar powers are supported.'); end
   end
   function r = sqrt(a), if any(a.C(1, :) <= 0), error('tps:domain', 'sqrt: not analytic at a nonpositive constant term'); end, r = power(a, 0.5); end
   function r = exp(a), e = exp(a.C(1, :)); fc = e ./ factorial((0:a.N)'); r = tps.compose(a, fc); end
   function r = expm1(a), r = exp(a); r.C(1, :) = expm1(a.C(1, :)); end
   function r = log(a)
      a0 = a.C(1, :);
      if any(a0 <= 0), error('tps:domain', 'log: not analytic at a nonpositive constant term'); end
      j = (1:a.N)';
      fc = [log(a0); ((-1).^(j+1)./j) .* (1 ./ a0).^j];
      r = tps.compose(a, fc);
   end
   function r = log1p(a)
      a0 = a.C(1, :);
      if any(a0 <= -1), error('tps:domain', 'log1p: not analytic'); end
      j = (1:a.N)';
      fc = [log1p(a0); ((-1).^(j+1)./j) .* (1 ./ (1 + a0)).^j];
      r = tps.compose(a, fc);
   end
   function r = sin(a), j = (0:a.N)'; fc = sin(a.C(1, :) + j*pi/2) ./ factorial(j); r = tps.compose(a, fc); end
   function r = cos(a), j = (0:a.N)'; fc = cos(a.C(1, :) + j*pi/2) ./ factorial(j); r = tps.compose(a, fc); end
   function r = sinh(a), j = (0:a.N)'; a0 = a.C(1, :); fc = (mod(j, 2) == 0).*sinh(a0) + (mod(j, 2) == 1).*cosh(a0); r = tps.compose(a, fc ./ factorial(j)); end
   function r = cosh(a), j = (0:a.N)'; a0 = a.C(1, :); fc = (mod(j, 2) == 0).*cosh(a0) + (mod(j, 2) == 1).*sinh(a0); r = tps.compose(a, fc ./ factorial(j)); end
   function r = tanh(a), e = exp(2*a); r = (e - 1)./(e + 1); end
   function r = tan(a), r = sin(a)./cos(a); end
   function r = atan(a)
      % derivative 1/(1 + (a0+q)^2) as a univariate series in q, then integrated
      a0 = a.C(1, :); N = a.N; m = numel(a0); fc = zeros(N+1, m); fc(1, :) = atan(a0);
      for c = 1:m
         den = zeros(N+1, 1); den(1) = 1 + a0(c)^2; if N >= 1, den(2) = 2*a0(c); end, if N >= 2, den(3) = 1; end
         g = zeros(N, 1);   % coefficients of 1/den
         for i = 0:N-1
            s = (i == 0); for l = 1:min(i, 2), s = s - den(l+1)*g(i-l+1); end
            g(i+1) = s/den(1);
         end
         fc(2:end, c) = g ./ (1:N)';
      end
      r = tps.compose(a, fc);
   end
   function r = abs(a)
      a0 = a.C(1, :); if any(a0 == 0), error('tps:domain', 'abs: not analytic at zero'); end
      r = tps.scale(a, reshape(sign(a0), a.sz));
   end
   function r = sum(a, dim)
      if nargin < 2, dim = find(a.sz ~= 1, 1); if isempty(dim), dim = 1; end, end
      sz = a.sz; if dim > numel(sz), r = a; return; end
      szr = sz; szr(dim) = 1;
      G = reshape(1:prod(szr), szr);
      rep = ones(1, numel(sz)); rep(dim) = sz(dim);
      G = repmat(G, rep);
      S = sparse(1:prod(sz), G(:), 1, prod(sz), prod(szr));
      r = tps(a.C*S, szr, a.k, a.N);
   end
   function r = prod(a, dim)
      if nargin < 2, dim = find(a.sz ~= 1, 1); if isempty(dim), dim = 1; end, end
      sz = a.sz; idx = repmat({':'}, 1, numel(sz));
      idx{dim} = 1; r = subsref(a, substruct('()', idx));
      for i = 2:sz(dim), idx{dim} = i; r = times(r, subsref(a, substruct('()', idx))); end
   end
   function r = dot(a, b), r = sum(times(a, b)); end

   %---------------------------------------------------------------- comparisons (constant terms)
   function r = lt(a, b), r = tps.c0(a) < tps.c0(b); end
   function r = le(a, b), r = tps.c0(a) <= tps.c0(b); end
   function r = gt(a, b), r = tps.c0(a) > tps.c0(b); end
   function r = ge(a, b), r = tps.c0(a) >= tps.c0(b); end
   function r = eq(a, b), r = tps.c0(a) == tps.c0(b); end
   function r = ne(a, b), r = tps.c0(a) ~= tps.c0(b); end
end

methods (Static)
   function v = vars(k, N)
      %VARS k x 1 array of the variables v_1..v_k, truncated at degree N
      T = tps.tables(k, N);
      C = zeros(T.n, k);
      for i = 1:k, e = zeros(1, k); e(i) = 1; if N >= 1, C(T.lookup(e*T.base + 1), i) = 1; end, end
      v = tps(C, [k 1], k, N);
   end
   function p = const(c, k, N), p = tps.lift(c, k, N); end
   function T = tables(k, N)
      %TABLES multi-indices of total degree <= N in k variables (graded order) and the
      %   multiplication table, cached
      persistent cache
      if isempty(cache), cache = containers.Map(); end
      key = sprintf('%d_%d', k, N);
      if isKey(cache, key), T = cache(key); return; end
      mi = zeros(0, k);
      for l = 0:N, mi = [mi; tps.comps(l, k)]; end %#ok<AGROW>
      n = size(mi, 1); deg = sum(mi, 2);
      base = (N+1).^(0:k-1)';
      code = mi*base;
      lookup = sparse(code + 1, 1, 1:n, (N+1)^k, 1);
      I = cell(n, 1); J = I; Tt = I;
      for i = 1:n
         j = find(deg <= N - deg(i));
         I{i} = repmat(i, numel(j), 1); J{i} = j; Tt{i} = full(lookup(code(i) + code(j) + 1));
      end
      I = vertcat(I{:}); J = vertcat(J{:}); Tt = vertcat(Tt{:});
      S = sparse(Tt, 1:numel(Tt), 1, n, numel(Tt));
      T = struct('mi', mi, 'deg', deg, 'n', n, 'base', base, 'lookup', lookup, 'I', I, 'J', J, 'S', S, ...
                 'fact', prod(factorial(mi), 2));
      cache(key) = T;
   end
   function C = comps(l, k)
      if k == 1, C = l; return; end
      C = zeros(0, k);
      for i = l:-1:0, R = tps.comps(l - i, k - 1); C = [C; repmat(i, size(R, 1), 1) R]; end %#ok<AGROW>
   end
end

methods (Static, Hidden)
   function p = lift(c, k, N)
      T = tps.tables(k, N); C = zeros(T.n, numel(c)); C(1, :) = c(:).';
      p = tps(C, size(c), k, N);
   end
   function r = like(x), if isa(x, 'tps'), r = x; else, error('tps:like', 'no series'); end, end
   function c = c0(a), if isa(a, 'tps'), c = double(a); else, c = a; end, end
   function [a, b] = match(a, b)
      if isnumeric(a), a = tps.lift(a, b.k, b.N); end
      if isnumeric(b), b = tps.lift(b, a.k, a.N); end
      if a.k ~= b.k || a.N ~= b.N, error('tps:mismatch', 'Series in different numbers of variables or degrees.'); end
   end
   function [a, b, map] = bcast(a, b)
      [a, b] = tps.match(a, b);
      ia = reshape(1:prod(a.sz), a.sz); ib = reshape(1:prod(b.sz), b.sz);
      ma = ia + 0*ib; mb = ib + 0*ia;   % implicit expansion
      map = [ma(:)'; mb(:)'];
   end
   function sz = bsz(a, b), sz = size(zeros(a.sz) + zeros(b.sz)); end
   function r = scale(a, s)
      if isscalar(s), r = tps(a.C*s, a.sz, a.k, a.N); return; end
      ia = reshape(1:prod(a.sz), a.sz); m = ia + 0*s; ss = s + 0*ia;
      r = tps(a.C(:, m(:)) .* ss(:).', size(m), a.k, a.N);
   end
   function C = mulC(A, B, k, N)
      T = tps.tables(k, N);
      C = T.S*(A(T.I, :).*B(T.J, :));
   end
   function r = compose(a, fc)
      % f(a0 + q) = sum_j fc(j+1,:) q^j
      N = a.N; q = a.C; q(1, :) = 0;
      R = zeros(size(q)); R(1, :) = fc(1, :);
      qp = zeros(size(q)); qp(1, :) = 1;
      for j = 1:N
         qp = tps.mulC(qp, q, a.k, N);
         R = R + qp .* fc(j+1, :);
      end
      r = tps(R, a.sz, a.k, a.N);
   end
   function r = catn(dim, varargin)
      ref = [];
      for i = 1:numel(varargin), if isa(varargin{i}, 'tps'), ref = varargin{i}; break; end, end
      maps = cell(size(varargin)); Cs = cell(size(varargin)); off = 0;
      for i = 1:numel(varargin)
         x = varargin{i}; if ~isa(x, 'tps'), x = tps.lift(x, ref.k, ref.N); end
         [x, ~] = tps.match(x, ref);
         n = prod(x.sz); maps{i} = off + reshape(1:n, x.sz); Cs{i} = x.C; off = off + n;
      end
      map = cat(dim, maps{:}); C = [Cs{:}];
      r = tps(C(:, map(:)), size(map), ref.k, ref.N);
   end
end
end
