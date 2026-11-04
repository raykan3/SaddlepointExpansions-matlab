classdef dd
%DD Double-double numbers: x = hi + lo with |lo| <= ulp(hi)/2, about 32 significant digits.
%   x = dd(a)          from a double array (exact) or a sym/vpa array (rounded to double-double)
%   x = dd(hi, lo)     from two double arrays (normalized)
%   double(x), vpa(x)  conversions; x.hi, x.lo the parts
%   Arithmetic + - .* ./ (elementwise, with implicit expansion; operands double or dd), scalar
%   *, /, ^ with an integer exponent, sqrt, abs, sum (compensated), comparisons, indexing and
%   concatenation are in double-double arithmetic (error-free transformations, Dekker's
%   splitting).  exp, log, log1p, expm1, sin, cos, tan, sinh, cosh, tanh, atan, erfc and powers
%   with non-integer exponents are evaluated elementwise in variable precision (40 digits) and
%   rounded to double-double; they are meant for scalars and short vectors.  Real values only.
%
%   Raymond Kan and Simon A. Broda, 2026.  Part of SaddlepointExpansions for MATLAB.

properties (SetAccess = private)
   hi = 0
   lo = 0
end

methods
   function x = dd(a, b)
      if nargin == 0, return; end
      if isa(a, 'dd'), x = a; return; end
      if nargin == 2
         [x.hi, x.lo] = dd.twosum(double(a), double(b));
         return
      end
      if isa(a, 'sym')
         h = double(a); x.hi = h; x.lo = double(a - sym(h, 'f'));   % 'f': the exact binary value
         return
      end
      if ~isreal(a), error('dd:complex', 'dd: real values only'); end
      x.hi = double(a); x.lo = zeros(size(x.hi));
   end

   %------------------------------------------------------------------ conversion and display
   function d = double(x), d = x.hi; end
   function v = vpa(x, n)
      if nargin < 2, n = 40; end
      v = vpa(sym(x.hi, 'f'), n) + vpa(sym(x.lo, 'f'), n);   % 'f': exact binary values, no rational guessing
   end
   function disp(x)
      if isempty(x.hi), fprintf('  empty dd\n'); return; end
      v = vpa(x, 32);
      fprintf('  dd array of size %s:\n', mat2str(size(x.hi)));
      disp(v);
   end
   function tf = isreal(~), tf = true; end
   function r = real(x), r = x; end
   function tf = isfinite(x), tf = isfinite(x.hi) & isfinite(x.lo); end
   function tf = isnan(x), tf = isnan(x.hi); end

   %------------------------------------------------------------------ size and indexing
   function varargout = size(x, varargin), [varargout{1:max(nargout, 1)}] = size(x.hi, varargin{:}); end
   function n = numel(x, varargin), n = numel(x.hi); end
   function n = length(x), n = length(x.hi); end
   function n = ndims(x), n = ndims(x.hi); end
   function tf = isempty(x), tf = isempty(x.hi); end
   function tf = isscalar(x), tf = isscalar(x.hi); end
   function tf = isvector(x), tf = isvector(x.hi); end
   function n = numArgumentsFromSubscript(~, ~, ~), n = 1; end
   function e = end(x, k, n)
      if n == 1, e = numel(x.hi); else, e = size(x.hi, k); end
   end
   function varargout = subsref(x, s)
      switch s(1).type
         case '()'
            r = dd(); r.hi = x.hi(s(1).subs{:}); r.lo = x.lo(s(1).subs{:});
            if numel(s) > 1, [varargout{1:nargout}] = subsref(r, s(2:end)); else, varargout{1} = r; end
         otherwise
            [varargout{1:nargout}] = builtin('subsref', x, s);
      end
   end
   function x = subsasgn(x, s, rhs)
      if ~strcmp(s(1).type, '()') || numel(s) > 1, x = builtin('subsasgn', x, s, rhs); return; end
      if ~isa(x, 'dd'), x = dd(x); end
      rhs = dd(rhs);
      x.hi(s(1).subs{:}) = rhs.hi; x.lo(s(1).subs{:}) = rhs.lo;
   end
   function r = reshape(x, varargin), r = dd(); r.hi = reshape(x.hi, varargin{:}); r.lo = reshape(x.lo, varargin{:}); end
   function r = transpose(x), r = dd(); r.hi = x.hi.'; r.lo = x.lo.'; end
   function r = ctranspose(x), r = transpose(x); end
   function r = horzcat(varargin), r = dd.catn(2, varargin{:}); end
   function r = vertcat(varargin), r = dd.catn(1, varargin{:}); end
   function r = cat(dim, varargin), r = dd.catn(dim, varargin{:}); end
   function r = repmat(x, varargin), r = dd(); r.hi = repmat(x.hi, varargin{:}); r.lo = repmat(x.lo, varargin{:}); end

   %------------------------------------------------------------------ arithmetic
   function r = plus(a, b)
      [ah, al, bh, bl] = dd.parts(a, b);
      r = dd(); [r.hi, r.lo] = dd.add(ah, al, bh, bl);
   end
   function r = minus(a, b)
      [ah, al, bh, bl] = dd.parts(a, b);
      r = dd(); [r.hi, r.lo] = dd.add(ah, al, -bh, -bl);
   end
   function r = uminus(a), r = dd(); r.hi = -a.hi; r.lo = -a.lo; end
   function r = uplus(a), r = a; end
   function r = times(a, b)
      [ah, al, bh, bl] = dd.parts(a, b);
      r = dd(); [r.hi, r.lo] = dd.mul(ah, al, bh, bl);
   end
   function r = rdivide(a, b)
      [ah, al, bh, bl] = dd.parts(a, b);
      r = dd(); [r.hi, r.lo] = dd.div(ah, al, bh, bl);
   end
   function r = ldivide(a, b), r = rdivide(b, a); end
   function r = mtimes(a, b)
      if isscalar(a) || isscalar(b), r = times(a, b); return; end
      a = dd(a); b = dd(b);
      [m, k] = size(a); [k2, n] = size(b);
      if k ~= k2, error('dd:mtimes', 'Inner matrix dimensions must agree.'); end
      r = dd(zeros(m, n));
      for i = 1:m
         for j = 1:n
            p = times(subsref(a, substruct('()', {i, ':'})), transpose(subsref(b, substruct('()', {':', j}))));
            r = subsasgn(r, substruct('()', {i, j}), sum(p));
         end
      end
   end
   function r = mrdivide(a, b)
      if isscalar(b), r = rdivide(a, b); else, error('dd:mrdivide', 'Only division by a scalar is supported.'); end
   end
   function r = power(a, b)
      if isa(b, 'dd'), b = double(b); end
      if isscalar(b) && b == round(b)
         a = dd(a);
         if b < 0, r = rdivide(1, power(a, -b)); return; end
         r = dd(ones(size(a))); x = a; e = b;
         while e > 0
            if mod(e, 2), r = times(r, x); end
            e = floor(e/2); if e > 0, x = times(x, x); end
         end
         return
      end
      r = dd.viavpa(@(v) v.^b, a);
   end
   function r = mpower(a, b)
      if isscalar(a), r = power(a, b); else, error('dd:mpower', 'Only scalar powers are supported.'); end
   end
   function r = sqrt(a)
      if any(a.hi(:) < 0), error('dd:domain', 'sqrt of a negative number'); end
      r = dd(); h = sqrt(a.hi);
      [ph, pl] = dd.twoprod(h, h);                       % a - h^2, exactly to double-double
      [eh, el] = dd.add(a.hi, a.lo, -ph, -pl);
      c = (eh + el)./(2*h); c(h == 0) = 0;
      [r.hi, r.lo] = dd.qts(h, c);
   end
   function r = abs(a), r = a; s = a.hi < 0 | (a.hi == 0 & a.lo < 0); r.hi(s) = -a.hi(s); r.lo(s) = -a.lo(s); end
   function r = sign(a), r = sign(a.hi); r(a.hi == 0) = sign(a.lo(a.hi == 0)); end
   function r = sum(a, dim)
      %SUM compensated sum along a dimension (default: the first non-singleton)
      if nargin < 2, dim = find(size(a.hi) ~= 1, 1); if isempty(dim), dim = 1; end, end
      h = a.hi; l = a.lo;
      perm = [dim, setdiff(1:max(ndims(h), dim), dim)];
      h = permute(h, perm); l = permute(l, perm); sz = size(h);
      h = reshape(h, sz(1), []); l = reshape(l, sz(1), []);
      [h, l] = dd.colsum(h, l);
      sz(1) = 1; r = dd(); r.hi = ipermute(reshape(h, sz), perm); r.lo = ipermute(reshape(l, sz), perm);
   end
   function r = prod(a)
      r = dd(1); for i = 1:numel(a.hi), r = times(r, subsref(a, substruct('()', {i}))); end
   end
   function r = dot(a, b), r = sum(times(a, b)); end

   %------------------------------------------------------------------ elementary functions
   function r = exp(a), r = dd.viavpa(@exp, a); end
   function r = expm1(a), r = dd.viavpa(@(v) exp(v) - 1, a); end
   function r = log(a), if any(a.hi(:) <= 0), error('dd:domain', 'log of a nonpositive number'); end, r = dd.viavpa(@log, a); end
   function r = log1p(a), r = dd.viavpa(@(v) log(1 + v), a); end
   function r = sin(a), r = dd.viavpa(@sin, a); end
   function r = cos(a), r = dd.viavpa(@cos, a); end
   function r = tan(a), r = dd.viavpa(@tan, a); end
   function r = sinh(a), r = dd.viavpa(@sinh, a); end
   function r = cosh(a), r = dd.viavpa(@cosh, a); end
   function r = tanh(a), r = dd.viavpa(@tanh, a); end
   function r = atan(a), r = dd.viavpa(@atan, a); end
   function r = erfc(a), r = dd.viavpa(@erfc, a); end

   %------------------------------------------------------------------ comparisons
   function r = lt(a, b), [ah, al, bh, bl] = dd.parts(a, b); r = ah < bh | (ah == bh & al < bl); end
   function r = le(a, b), [ah, al, bh, bl] = dd.parts(a, b); r = ah < bh | (ah == bh & al <= bl); end
   function r = gt(a, b), r = lt(b, a); end
   function r = ge(a, b), r = le(b, a); end
   function r = eq(a, b), [ah, al, bh, bl] = dd.parts(a, b); r = ah == bh & al == bl; end
   function r = ne(a, b), r = ~eq(a, b); end
   function r = max(a, b), r = a; s = lt(a, b); r = subsasgn(r, substruct('()', {s}), subsref(dd(b), substruct('()', {s}))); end
end

methods (Static)
   %------------------------------------------------------------------ error-free kernels
   function [s, e] = twosum(a, b)
      s = a + b; bb = s - a; e = (a - (s - bb)) + (b - bb);
   end
   function [s, e] = qts(a, b)
      % quick-two-sum, |a| >= |b|
      s = a + b; e = b - (s - a);
   end
   function [h, l] = split(a)
      c = 134217729*a; h = c - (c - a); l = a - h;
   end
   function [p, e] = twoprod(a, b)
      p = a.*b;
      [ah, al] = dd.split(a); [bh, bl] = dd.split(b);
      e = ((ah.*bh - p) + ah.*bl + al.*bh) + al.*bl;
   end
   function [h, l] = add(ah, al, bh, bl)
      % accurate double-double addition
      [s, e] = dd.twosum(ah, bh); [t, f] = dd.twosum(al, bl);
      e = e + t; [s, e] = dd.qts(s, e); e = e + f; [h, l] = dd.qts(s, e);
   end
   function [h, l] = mul(ah, al, bh, bl)
      [p, e] = dd.twoprod(ah, bh);
      e = e + (ah.*bl + al.*bh);
      [h, l] = dd.qts(p, e);
   end
   function [h, l] = div(ah, al, bh, bl)
      q1 = ah./bh;
      [ph, pl] = dd.mul(q1, 0*q1, bh, bl); [rh, rl] = dd.add(ah, al, -ph, -pl);
      q2 = rh./bh;
      [ph, pl] = dd.mul(q2, 0*q2, bh, bl); [rh, rl] = dd.add(rh, rl, -ph, -pl);
      q3 = rh./bh;
      [h, l] = dd.qts(q1, q2); [h, l] = dd.add(h, l, q3, 0*q3);
   end
   function [h, l] = colsum(h, l)
      % pairwise compensated sums of the columns of the double-double matrix (h, l), n x m -> 1 x m
      while size(h, 1) > 1
         if mod(size(h, 1), 2), h(end+1, :) = 0; l(end+1, :) = 0; end %#ok<AGROW>
         [h, l] = dd.add(h(1:2:end, :), l(1:2:end, :), h(2:2:end, :), l(2:2:end, :));
      end
   end
end

methods (Static, Hidden)
   function [ah, al, bh, bl] = parts(a, b)
      if isa(a, 'dd'), ah = a.hi; al = a.lo; else, if ~isreal(a), error('dd:complex', 'dd: real values only'); end, ah = double(a); al = zeros(size(ah)); end
      if isa(b, 'dd'), bh = b.hi; bl = b.lo; else, if ~isreal(b), error('dd:complex', 'dd: real values only'); end, bh = double(b); bl = zeros(size(bh)); end
      if ~isequal(size(ah), size(bh))                    % implicit expansion
         z = zeros(size(ah + bh)); ah = ah + z; al = al + z; bh = bh + z; bl = bl + z;
      end
   end
   function r = viavpa(f, a)
      a = dd(a); v = f(vpa(a, 40)); r = dd(v);
   end
   function r = catn(dim, varargin)
      H = cell(size(varargin)); L = H;
      for i = 1:numel(varargin), x = dd(varargin{i}); H{i} = x.hi; L{i} = x.lo; end
      r = dd(); r.hi = cat(dim, H{:}); r.lo = cat(dim, L{:});
   end
end
end
