classdef CGF
%CGF Cumulant generating function of a random vector, and the saddlepoint expansions from it.
%   K = CGF(fun, d)  or  CGF(fun, d, 'domain', pred)
%   fun:    a function of a column vector t (d x 1) returning K(t), written with ordinary
%           arithmetic and elementary functions so that it can be evaluated on truncated power
%           series (see tps): +, -, *, /, .^, exp, log, sqrt, sin, cos, sinh, cosh, tanh, atan,
%           sum, prod, products with constant matrices, indexing t(j).  Outside the domain it may
%           raise an error with identifier 'tps:domain' (log or sqrt of a nonpositive number do)
%           or return a non-finite value.
%   d:      the dimension.
%   pred:   optional domain predicate @(t) logical, cheaper and more robust when the formula is
%           finite outside the domain (1/(1-t) for t > 1, say).
%   Options 'Taylor', tfun: a function tfun(t, N, M) returning the Taylor polynomial (a scalar
%   tps in size(M,2) variables) of K(t + M v) of degree N, used instead of automatic
%   differentiation (M = [] means the identity); the library functions use it where closed
%   forms are faster (quadratic_forms_cgf).
%
%   Expansions (orders 1..order, relative error O(n^-M) for the approximation of order M, the n
%   absorbed in K):
%      r = tailprob(K, x, 'order', 3)        orthant probability P(X >= x) (Theorem 1), with
%                                            'lower', true for P(X <= x) and 'signs', s for a
%                                            mixed corner; inclusion-exclusion for saddlepoints
%                                            with negative components.  r.F(M), r.tau, r.status,
%                                            r.info
%      r = density(K, x, 'order', 3)         density (Theorem 3); r.f(M)
%      r = ratio_density(K, r0, 'order', 3)  joint density of R_j = X_j/Y_j, j = 1..numel(r0),
%                                            Y_j > 0 (Theorem 4); K of (X, Y) (dimension 2d) or
%                                            of (X, Y) with a common denominator (d + 1); r.f(M)
%      r = ratio_tailprob(K, r0, 'order', 3) P(R >= r0) ('lower', true: <=; 'signs', s); r.F(M)
%   Studentized means Q = X/sqrt(Y), Y > 0, K the joint cgf of (X, Y) (dimension 2; Bao, Broda
%   and Kan, "Inversion formulas and saddlepoint expansions for Studentized means"):
%      r = studentized_density(K, q, 'order', 3)   density of Q at q; r.f(M)
%      r = studentized_tailprob(K, q, 'order', 3)  P(Q >= q) ('lower', true: P(Q <= q)); r.F(M)
%      r = studentized_quantile(K, p, 'order', 3)  the q with P(Q <= q) = p under the expansion
%      sd = studentized_saddlepoint(K, q), studentized_locus(K, s, tguess)
%      [tau, info] = saddlepoint(K, x)       the saddlepoint K'(tau) = x
%      D = derivatives(K, t, N)              all derivatives of order <= N at t: D.nu, D.value
%   Other name-value pairs of tailprob and density are passed to the engine (spa_orthant2,
%   spa_orthant, spa_bi, spa_mv): 'nG', 'qtol', 'logscale', 'method', ...
%   Transformations: reflect(K, s), marginal(K, idx), linear_combination(K, B), sample_mean(K, n),
%   shift(K, c), K1 + K2 (independent sum).
%
%   Example:
%      K = CGF(@(t) -log(1 - t(1)) - log(1 - t(2)) - log(1 - t(1) - t(2)), 2, ...
%              'domain', @(t) t(1) < 1 && t(2) < 1 && t(1) + t(2) < 1);
%      Kn = sample_mean(K, 5);
%      r = tailprob(Kn, [3 3], 'order', 4);   r.F   % exact 0.02215196
%
%   Raymond Kan and Simon A. Broda, 2026. MATLAB counterpart of SaddlepointExpansions.jl; free to
%   use and modify with a citation of Broda and Kan.

properties (SetAccess = private)
   fun          % K as a function of a d x 1 vector
   d            % dimension
   domain       % [] or a predicate @(t) logical
   taylorfun    % [] or @(t, N, M) -> tps, the Taylor polynomial of K(t + M v)
end

methods
   function obj = CGF(fun, d, varargin)
      if nargin == 0, return; end
      if ~(isscalar(d) && d >= 1 && d == round(d)), error('CGF:dim', 'CGF: the dimension must be a positive integer'); end
      p = inputParser; p.addParameter('domain', []); p.addParameter('Taylor', []);
      p.parse(varargin{:});
      obj.fun = fun; obj.d = d; obj.domain = p.Results.domain; obj.taylorfun = p.Results.Taylor;
   end
   function v = eval(obj, t), v = obj.fun(reshape(t, [], 1)); end
   function disp(obj)
      fprintf('  CGF of dimension %d', obj.d);
      if ~isempty(obj.domain), fprintf(' with domain predicate'); end
      if ~isempty(obj.taylorfun), fprintf(' and closed-form Taylor polynomials'); end
      fprintf('\n');
   end

   %---------------------------------------------------------------- Taylor polynomials
   function p = taylor(obj, t, N, M)
      %TAYLOR Taylor polynomial of degree N in v of K(t + M v) (M = [] for the identity), a scalar
      %   tps; [] when t is outside the domain.
      if nargin < 4, M = []; end
      t = double(reshape(t, [], 1)); N = max(N, 1);
      if ~isempty(obj.domain) && ~obj.domain(t), p = []; return; end
      if ~isempty(obj.taylorfun)
         p = obj.taylorfun(t, N, M);
      else
         if isempty(M), k = obj.d; else, k = size(M, 2); end
         v = tps.vars(k, N);
         if isempty(M), arg = t + v; else, arg = t + M*v; end
         try
            p = obj.fun(arg);
         catch err
            if strcmp(err.identifier, 'tps:domain'), p = []; return; end
            rethrow(err);
         end
         if isnumeric(p), if ~isfinite(p), p = []; return; end, p = tps.const(p, k, N); end
      end
      if isempty(p) || ~isfinite(p.C(1)), p = []; end
   end
   function p = taylordd(obj, t, N, M)
      %TAYLORDD Taylor polynomial of degree N in v of K(t + M v) with double-double coefficients
      %   (a scalar tpsdd); t a double or dd vector, M = [] for the identity; [] outside the
      %   domain.  K is evaluated through its function on double-double series (automatic
      %   differentiation); a closed-form 'Taylor' function, which works in double precision, is
      %   not used, so the function must be written with operations that tpsdd supports.
      if nargin < 4, M = []; end
      t = reshape(dd(t), [], 1); N = max(N, 1);
      if ~isempty(obj.domain) && ~obj.domain(double(t)), p = []; return; end
      if isempty(M), k = obj.d; else, k = size(M, 2); end
      v = tpsdd.vars(k, N);
      if isempty(M), arg = tpsdd.const(t, k, N) + v; else, arg = tpsdd.const(t, k, N) + M*v; end
      try
         p = obj.fun(arg);
      catch err
         if any(strcmp(err.identifier, {'tps:domain', 'dd:domain'})), p = []; return; end
         rethrow(err);
      end
      if ~isa(p, 'tpsdd'), p = tpsdd.const(p, k, N); end
      if ~isfinite(p), p = []; end
   end
   function D = derivatives(obj, t, N)
      %DERIVATIVES all derivatives of K of orders <= N at t: D.nu (one multi-index per row, in
      %   graded order) and D.value; [] outside the domain.  D.value(1) is K(t).
      p = taylor(obj, t, N);
      if isempty(p), D = []; return; end
      T = tps.tables(obj.d, max(N, 1)); keep = T.deg <= N;
      D = struct('nu', T.mi(keep, :), 'value', p.C(keep, 1).*T.fact(keep));
   end

   %---------------------------------------------------------------- saddlepoint and expansions
   function [tau, info] = saddlepoint(obj, x, tau0)
      %SADDLEPOINT the saddlepoint K'(tau) = x by Newton's method with backtracking (spa_saddle);
      %   tau0: optional starting value (default zeros).  info: iterations, Newton decrement,
      %   residual.
      if nargin < 3, tau0 = []; end
      x = reshape(x, 1, []); if numel(x) ~= obj.d, error('CGF:x', 'saddlepoint: x must have length %d', obj.d); end
      [tau, info] = spa_saddle(cgf_kfun(obj, 2), x, tau0);
   end
   function r = tailprob(obj, x, varargin)
      %TAILPROB orthant probability P(X >= x) by the expansion of Theorem 1, orders 1..order.
      %   r = tailprob(K, x, 'order', M)                  P(X >= x)
      %   r = tailprob(K, x, 'order', M, 'lower', true)   P(X <= x)
      %   r = tailprob(K, x, 'order', M, 'signs', s)      mixed corner, s_j = +1 (>=) or -1 (<=)
      %   Saddlepoints with negative components are handled by inclusion-exclusion (Theorem 2).
      %   r.F(M): the approximation of order M (relative error O(n^-M)); r.tau: the saddlepoint;
      %   r.status: 'ok' or a warning; r.info: diagnostics of the engine.  Other name-value
      %   pairs go to the engine: 'nG', 'qtol', 'nGmax', 'method' ('quad' or 'qmc'), 'nqmc',
      %   'nshift', 'seed', 'logscale' (d = 2); see spa_bi, spa_tri, spa_mv.
      %   d = 1: Hauschildt-Robinson (Corollary 2); d = 2: spa_orthant2; d >= 3: spa_orthant.
      [order, lower, signs, opts] = parseopts('tailprob', obj.d, varargin{:});
      x = reshape(x, 1, []);
      if numel(x) ~= obj.d, error('CGF:x', 'tailprob: x must have length %d', obj.d); end
      if ~all(isfinite(x)), error('CGF:x', 'tailprob: x must be finite'); end
      sg = cornersigns(obj.d, lower, signs);
      Kr = obj; if any(sg ~= 1), Kr = reflect(obj, sg); end
      xr = sg .* x; maxdeg = 2*order + 1;
      Kf = cgf_kfun(Kr, maxdeg);
      if obj.d == 1
         [s, sinfo] = spa_saddle(Kf, xr, []);
         Kd = arrayfun(@(k) Kf(k, s), 0:maxdeg);
         if ~all(isfinite(Kd)), error('CGF:domain', 'tailprob: the saddlepoint is outside the domain of K'); end
         [~, ~, ~, upp, kappa] = spa_uni(Kd, s, xr, order);
         r = struct('F', upp(:), 'tau', sg.*s, 'status', 'ok', 'info', struct('kappa', kappa, 'saddle', sinfo));
      elseif obj.d == 2
         opts.Kwhite = cgf_kwhite(Kr, maxdeg);
         [F, tau, canc, kappa, info] = spa_orthant2(Kf, xr, order, opts);
         info.canc = canc; info.kappa = kappa;
         r = struct('F', F(:), 'tau', sg.*tau, 'status', info.status, 'info', info);
      else
         opts.Kwhite = cgf_kwhite(Kr, maxdeg);
         [F, tau, err] = spa_orthant(Kf, xr, order, opts);
         r = struct('F', F(:), 'tau', sg.*reshape(tau, 1, []), 'status', 'ok', 'info', struct('err', err));
      end
   end
   function r = density(obj, x, varargin)
      %DENSITY density of X at x by the expansion of Theorem 3 (Daniels, Corollary 1, for d = 1).
      %   r = density(K, x, 'order', M).  r.f(M): the approximation of order M (relative error
      %   O(n^-M)); r.tau: the saddlepoint; r.status; r.info (for d >= 3, kappa_pdf: the
      %   cancellation among the terms).  Other name-value pairs go to spa_bi or spa_mv.
      [order, ~, ~, opts] = parseopts('density', obj.d, varargin{:});
      x = reshape(x, 1, []);
      if numel(x) ~= obj.d, error('CGF:x', 'density: x must have length %d', obj.d); end
      maxdeg = 2*order + 1; Kf = cgf_kfun(obj, maxdeg);
      if obj.d == 1
         [s, sinfo] = spa_saddle(Kf, x, []);
         Kd = arrayfun(@(k) Kf(k, s), 0:maxdeg);
         if ~all(isfinite(Kd)), error('CGF:domain', 'density: the saddlepoint is outside the domain of K'); end
         pdf = spa_uni(Kd, s, x, order);
         r = struct('f', pdf(:), 'tau', s, 'status', 'ok', 'info', struct('saddle', sinfo));
      elseif obj.d == 2
         opts.Kwhite = cgf_kwhite(obj, maxdeg);
         [pdf, ~, tau, ~, ~, ~, info] = spa_bi(Kf, x, order, [], [], [], opts);
         st = 'ok'; if contains(info.status, 'coefficients inaccurate'), st = 'coefficients inaccurate'; end
         r = struct('f', pdf(:), 'tau', tau, 'status', st, 'info', info);
      else
         opts.Kwhite = cgf_kwhite(obj, maxdeg); opts.method = 'none';
         [pdf, ~, tau, ~, ~, info] = spa_mv(Kf, x, order, opts);
         st = 'ok';
         if contains(info.status, 'coefficients inaccurate'), st = 'coefficients inaccurate'; elseif contains(info.status, 'refinement failed'), st = 'refinement failed'; end
         r = struct('f', pdf(:), 'tau', reshape(tau, 1, []), 'status', st, 'info', info);
      end
   end
   function r = ratio_density(obj, r0, varargin)
      %RATIO_DENSITY density of R = (X_1/Y_1, ..., X_d/Y_d) at r0 (d = numel(r0), Y_j > 0).
      %   K is the cgf of (X_1..X_d, Y_1..Y_d) (dimension 2d), or of (X_1..X_d, Y) (dimension
      %   d + 1) for a common denominator Y.  d = 1: Rice's expansion (Corollary 5); d >= 2:
      %   Theorem 4 and Corollary 3.  r.f(M), M = 1..order; r.shat the saddlepoint of K_W.
      [order, ~, ~, opts] = parseopts('ratio_density', numel(r0), varargin{:});
      r0 = reshape(r0, 1, []); d = numel(r0); K2 = ratiocgf(obj, d, 'ratio_density');
      if d == 1 && ~isfield(opts, 'general')          % 'general', true: the d-ratio code for d = 1
         KW = linear_combination(K2, [1; -r0]);
         [s, sinfo] = spa_saddle(cgf_kfun(KW, 2), 0, []);
         N = 2*order + 1;
         P = taylor(K2, [s; -r0*s], N, [1 0; -r0 1]);   % K(s + v1, -r(s + v1) + v2)
         if isempty(P), error('CGF:domain', 'ratio_density: the saddlepoint is outside the domain of K'); end
         KWd = coef(P, [(0:2*order)' zeros(2*order+1, 1)]).';
         gd = coef(P, [(0:2*order-2)' ones(2*order-1, 1)]).';
         f = spa_ratio_pdf(KWd, gd, s, order);
         r = struct('f', f(:), 'shat', s, 'status', 'ok', 'info', struct('KWd', KWd, 'gd', gd, 'saddle', sinfo));
      else
         [f, info] = ratio_mv(K2, r0, order);
         r = struct('f', f(:), 'shat', info.shat, 'status', 'ok', 'info', info);
      end
   end
   function r = ratio_tailprob(obj, r0, varargin)
      %RATIO_TAILPROB P(R >= r0) for R = (X_1/Y_1, ..., X_d/Y_d), Y_j > 0, as the orthant
      %   probability P(W >= 0) of W = X - r0.*Y (Corollary 4 for d = 1, Theorems 1 and 2
      %   otherwise); 'lower', true gives P(R <= r0) and 'signs', s a mixed corner.  K as in
      %   ratio_density.  Other name-value pairs go to tailprob.  r.F(M), r.shat.
      [order, lower, signs, opts] = parseopts('ratio_tailprob', numel(r0), varargin{:});
      r0 = reshape(r0, 1, []); d = numel(r0); K2 = ratiocgf(obj, d, 'ratio_tailprob');
      KW = linear_combination(K2, [eye(d); -diag(r0)]);
      args = [fieldnames(opts)'; struct2cell(opts)'];
      if ~isempty(signs), args = [args(:)' {'signs', signs}]; end
      t = tailprob(KW, zeros(1, d), 'order', order, 'lower', lower, args{:});
      r = struct('F', t.F, 'shat', t.tau, 'status', t.status, 'info', t.info);
   end

   %---------------------------------------------------------------- Studentized means
   function sd = studentized_saddlepoint(obj, q)
      %STUDENTIZED_SADDLEPOINT the saddlepoint (s, t, v) of the expansions for Q = X/sqrt(Y) at q:
      %   the root of K_s = q v, K_t = v^2, q s + 2 t v = 0 with v > 0, K the joint cgf of (X, Y),
      %   Y > 0, by Newton's method with continuation in q from q0 = E(X)/E(Y)^(1/2), where
      %   s = t = 0.  sd.s, sd.t, sd.v, sd.q0.
      studcheck(obj, 'studentized_saddlepoint'); sd = stud_saddle(obj, scalarq(q, 'studentized_saddlepoint'));
   end
   function r = studentized_locus(obj, s, tguess)
      %STUDENTIZED_LOCUS the point of the saddlepoint locus s K_s + 2 t K_t = 0 with abscissa s,
      %   by Newton's method in t from tguess; r.t, r.v = K_t^(1/2) and r.q = K_s/v, the q whose
      %   saddlepoint it is.  The saddlepoints of all q lie on this curve through the origin.
      studcheck(obj, 'studentized_locus'); r = stud_locus(obj, s, tguess);
   end
   function r = studentized_density(obj, q, varargin)
      %STUDENTIZED_DENSITY saddlepoint expansions of orders 1..order of the density at q of
      %   Q = X/sqrt(Y), Y > 0, from the joint cgf K of (X, Y) (Theorem 2 of Bao, Broda and Kan):
      %   r.f(M) keeps the terms m <= M - 1 of the expansion in powers of 1/n (relative error
      %   O(n^-M), the n absorbed in K).  With psi(s, t) = K(s, t) + q^2 s^2/(4t), the amplitude
      %   expansion of a bivariate integral about the saddlepoint (s, t, v), evaluated in the
      %   coordinates of a complex factorization of psi'' (indefinite where t < 0).
      %   r = studentized_density(K, q, 'order', M, 'saddle', sd).  r.saddle; r.status;
      %   r.info.kappa: the cancellation indicator, large when |t| is large.  When it exceeds 1e6
      %   the computation is repeated in double-double arithmetic (about 32 digits: the Taylor
      %   polynomial of K by tpsdd, the saddlepoint refined, all sums; r.info.refined is then
      %   'double-double'), as the Julia package repeats it in BigFloat; 'refine', false keeps the
      %   double-precision result (status 'coefficients inaccurate' when the indicator exceeds
      %   1e10), and 'precision', 'dd' computes in double-double from the start.  The refinement
      %   evaluates K on double-double series (CGF/taylordd), so K must be written with operations
      %   that tpsdd supports; if it fails, the status is 'refinement failed'.
      studcheck(obj, 'studentized_density');
      [order, ~, saddle, ~, refine, prec] = studopts('studentized_density', varargin{:});
      q = scalarq(q, 'studentized_density');
      if strcmp(prec, 'dd')
         if isempty(saddle), saddle = stud_saddle(obj, q); end
         r = stud_density_dd(obj, q, order, saddle);
         return
      end
      r = stud_density(obj, q, order, saddle);
      if refine && r.info.kappa(end) > 1e6
         % the terms cancel strongly: repeat the computation in double-double arithmetic
         try
            r = stud_density_dd(obj, q, order, r.saddle);
         catch err
            r.status = 'refinement failed'; r.info.refine_error = err.message;
         end
      end
   end
   function r = studentized_tailprob(obj, q, varargin)
      %STUDENTIZED_TAILPROB saddlepoint expansions of orders 1..order of P(Q >= q) for
      %   Q = X/sqrt(Y), Y > 0, from the joint cgf K of (X, Y) (Theorem 3 of Bao, Broda and Kan):
      %   r.F(M) keeps the terms of orders n^(-m/2), m <= 2M - 1 (relative error O(n^-M));
      %   r.info.Fp(j+1), j = 0..2M-1, the expansions with the terms up to n^(-j/2).
      %   r = studentized_tailprob(K, q, 'order', M, 'lower', true) gives P(Q <= q); 'saddle', sd
      %   supplies the saddlepoint.  Hauschildt-Robinson type in s after the t-integral is
      %   expanded by Laplace's method; r.info.kappa: the cancellation indicator.  'precision',
      %   'dd' computes in double-double arithmetic (CGF/taylordd; the integrals I_m in variable
      %   precision).
      studcheck(obj, 'studentized_tailprob');
      [order, lower, saddle, ~, ~, prec] = studopts('studentized_tailprob', varargin{:});
      q = scalarq(q, 'studentized_tailprob');
      if strcmp(prec, 'dd')
         if isempty(saddle), saddle = stud_saddle(obj, q); end
         r = stud_tailprob_dd(obj, q, order, lower, saddle);
      else
         r = stud_tailprob(obj, q, order, lower, saddle);
      end
   end
   function r = studentized_quantile(obj, prob, varargin)
      %STUDENTIZED_QUANTILE the quantile of the saddlepoint distribution function of order
      %   'order' (studentized_tailprob with 'lower', true): P(Q <= q) = prob.  The search runs
      %   over s along the saddlepoint locus (studentized_locus), so that no saddlepoint equation
      %   is solved for a trial q.  r = studentized_quantile(K, prob, 'order', M, 'tol', 1e-10);
      %   r.q, r.saddle.
      studcheck(obj, 'studentized_quantile');
      [order, ~, ~, tol] = studopts('studentized_quantile', varargin{:});
      if ~(isscalar(prob) && prob > 0 && prob < 1), error('CGF:prob', 'studentized_quantile: prob must be in (0, 1)'); end
      r = stud_quantile(obj, prob, order, tol);
   end

   %---------------------------------------------------------------- transformations
   function K2 = linear_combination(obj, B)
      %LINEAR_COMBINATION cgf of Y = B'X for a d x k matrix B: K_Y(u) = K(B u).
      if size(B, 1) ~= obj.d, error('CGF:B', 'linear_combination: B must have %d rows', obj.d); end
      dom = []; if ~isempty(obj.domain), dom = @(u) obj.domain(B*u); end
      tf = @(u, N, M) lc_taylor(obj, B, u, N, M);
      K2 = CGF(@(u) obj.fun(B*u), size(B, 2), 'domain', dom, 'Taylor', tf);
   end
   function K2 = reflect(obj, signs)
      %REFLECT cgf of (s_1 X_1, ..., s_d X_d) for signs s_j = +-1 (default: all -1).
      if nargin < 2, signs = -ones(1, obj.d); end
      if numel(signs) ~= obj.d || ~all(abs(signs) == 1), error('CGF:signs', 'reflect: signs must be %d values +-1', obj.d); end
      K2 = linear_combination(obj, diag(signs));
   end
   function K2 = marginal(obj, idx)
      %MARGINAL cgf of the components idx of X.
      if numel(unique(idx)) ~= numel(idx) || any(idx < 1 | idx > obj.d), error('CGF:idx', 'marginal: idx must be distinct indices in 1:%d', obj.d); end
      E = zeros(obj.d, numel(idx)); E(sub2ind(size(E), idx(:)', 1:numel(idx))) = 1;
      K2 = linear_combination(obj, E);
   end
   function K2 = sample_mean(obj, n)
      %SAMPLE_MEAN cgf of the mean of n independent copies: n K(t/n).
      if ~(n > 0), error('CGF:n', 'sample_mean: n must be positive'); end
      dom = []; if ~isempty(obj.domain), dom = @(t) obj.domain(t/n); end
      tf = @(t, N, M) scaled(taylor(obj, t/n, N, ifelse(isempty(M), eye(obj.d)/n, M/n)), n);
      K2 = CGF(@(t) n*obj.fun(t/n), obj.d, 'domain', dom, 'Taylor', tf);
   end
   function K2 = shift(obj, c)
      %SHIFT cgf of X + c: K(t) + c't.
      c = reshape(c, [], 1); if numel(c) ~= obj.d, error('CGF:c', 'shift: c must have length %d', obj.d); end
      tf = @(t, N, M) addlinear(taylor(obj, t, N, M), c, t, ifelse(isempty(M), eye(obj.d), M));
      K2 = CGF(@(t) obj.fun(t) + c.'*t, obj.d, 'domain', obj.domain, 'Taylor', tf);
   end
   function K3 = plus(K1, K2)
      %PLUS cgf of the sum of two independent vectors of the same dimension.
      if K1.d ~= K2.d, error('CGF:dim', '+: the cumulant generating functions must have the same dimension'); end
      if isempty(K1.domain), dom = K2.domain; elseif isempty(K2.domain), dom = K1.domain; else, dom = @(t) K1.domain(t) && K2.domain(t); end
      tf = @(t, N, M) addseries(taylor(K1, t, N, M), taylor(K2, t, N, M));
      K3 = CGF(@(t) K1.fun(t) + K2.fun(t), K1.d, 'domain', dom, 'Taylor', tf);
   end
end
end

%---------------------------------------------------------------- local helpers
function y = ifelse(c, a, b), if c, y = a; else, y = b; end, end
function p = lc_taylor(obj, B, u, N, M)
   if isempty(M), p = taylor(obj, B*u, N, B); else, p = taylor(obj, B*u, N, B*M); end
end
function K2 = ratiocgf(obj, d, name)
   if obj.d == 2*d, K2 = obj;
   elseif obj.d == d + 1, K2 = linear_combination(obj, [eye(d) zeros(d); zeros(1, d) ones(1, d)]);
   else, error('CGF:dim', '%s: K must have dimension 2d (numerators, denominators) or d + 1 (common denominator), d = %d', name, d);
   end
end
function p = scaled(p, n), if ~isempty(p), p = n*p; end, end
function p = addseries(p1, p2), if isempty(p1) || isempty(p2), p = []; else, p = p1 + p2; end, end
function p = addlinear(p, c, t, M)
   if isempty(p), return; end
   v = tps.vars(size(M, 2), p.N);
   p = p + c.'*t + (c.'*M)*v;
end
function [order, lower, signs, opts] = parseopts(name, d, varargin)
   order = 3; lower = false; signs = []; opts = struct();
   if mod(numel(varargin), 2), error('CGF:opts', '%s: options must be name-value pairs', name); end
   for i = 1:2:numel(varargin)
      key = varargin{i}; val = varargin{i+1};
      switch lower_(key)
         case 'order', order = val;
         case 'lower', lower = logical(val);
         case 'signs', signs = val;
         otherwise, opts.(key) = val;
      end
   end
   if ~(isscalar(order) && order >= 1 && order == round(order)), error('CGF:order', '%s: order must be a positive integer', name); end
   if ~isempty(signs) && (numel(signs) ~= d || ~all(abs(signs) == 1)), error('CGF:signs', '%s: signs must be %d values +-1', name, d); end
end
function s = lower_(s), s = lower(s); end
function studcheck(obj, name)
   if obj.d ~= 2, error('CGF:dim', '%s: K must be the joint cumulant generating function of (X, Y), of dimension 2', name); end
end
function q = scalarq(q, name)
   if ~(isnumeric(q) && isscalar(q) && isreal(q) && isfinite(q)), error('CGF:q', '%s: q must be a finite real scalar', name); end
   q = double(q);
end
function [order, lower, saddle, tol, refine, prec] = studopts(name, varargin)
   order = 3; lower = false; saddle = []; tol = 1e-10; refine = true; prec = 'double';
   if mod(numel(varargin), 2), error('CGF:opts', '%s: options must be name-value pairs', name); end
   for i = 1:2:numel(varargin)
      switch lower_(varargin{i})
         case 'order', order = varargin{i+1};
         case 'lower', lower = logical(varargin{i+1});
         case 'saddle', saddle = varargin{i+1};
         case 'tol', tol = varargin{i+1};
         case 'refine', refine = logical(varargin{i+1});
         case 'precision'
            prec = lower_(varargin{i+1});
            if ~any(strcmp(prec, {'double', 'dd'})), error('CGF:opts', '%s: precision must be ''double'' or ''dd''', name); end
         otherwise, error('CGF:opts', '%s: unknown option %s', name, varargin{i});
      end
   end
   if ~(isscalar(order) && order >= 1 && order == round(order)), error('CGF:order', '%s: order must be a positive integer', name); end
end
function sg = cornersigns(d, lower, signs)
   if ~isempty(signs), sg = reshape(signs, 1, []); elseif lower, sg = -ones(1, d); else, sg = ones(1, d); end
end
