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
%   Conditional distributions (Broda and Kan, "Conditional and lattice saddlepoint expansions of
%   arbitrary order"); A the indices of the tail components, the others, B, conditioned on:
%      r = conditional_tailprob(K, x, A, 'order', 3)  P(X_A >= x_A | X_B = x_B) (Theorem 3);
%                                            'lower', 'signs' as in tailprob; r.F(M), r.tau,
%                                            r.taut, r.status, r.info
%      r = conditional_density(K, x, A, 'order', 3)   the conditional density (Theorem 4); r.f(M)
%   Lattice variables ('lattice', true; integer components in A, Theorem 5): continuity-
%   corrected P(S_A >= k_A | S_B = k_B), and P(S_A = k_A | S_B = k_B); without conditioning,
%   tailprob(K, k, 'lattice', true) for P(S >= k) and density(K, k, 'lattice', true) for
%   P(S = k); 'sinh', 'saddle' (default) or 'origin' for the expansion of s/(2 sinh(s/2)).
%   Quantiles of the expansions, by Newton's method in saddlepoint space:
%      r = quantile(K, p, 'order', 3)        the q with P(X <= q) = p ('lower', false: P(X >= q));
%                                            r.q, r.tau, r.F, r.iterations, r.status
%      r = ratio_quantile(K, p, 'order', 3)  the r with P(X_1/X_2 <= r) = p; r.r, r.shat, ...
%   Normalized densities: 'normalize', true in density, ratio_density (one ratio) and
%   studentized_density divides the expansion of every order by its mass (r.info.mass), computed
%   by quadrature in saddlepoint space; 'normalize', r.info.mass reuses it.
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
      %   r = tailprob(K, k, 'order', M, 'lattice', true): X = S integer-valued and so is x = k;
      %   r.F is then P(S >= k) with the continuity correction and the amplitude
      %   prod_j s_j/(2 sinh(s_j/2)), by Corollary 1 of Broda and Kan, "Conditional and lattice
      %   saddlepoint expansions of arbitrary order", the case without conditioning components (the
      %   computation of conditional_tailprob with A = 1:d, and the same numbers): the
      %   saddlepoint is taken at k - 1/2 in every component, and 'sinh', 'saddle' (default)
      %   expands the amplitude about the saddlepoint, 'sinh', 'origin' about the origin, keeping
      %   s^(2M-2) at order M (Kolassa 2003).  In one dimension r.F(1), with the terms of orders
      %   n^0 and n^(-1/2), has the relative error O(n^-1) of the continuity-corrected
      %   approximations of Daniels (1987).  'lower', true gives P(S <= k); saddlepoints with
      %   negative components are handled by the inclusion-exclusion with
      %   P(S_j <= k_j - 1) = P(-S_j >= -k_j + 1), the components not kept in a term being
      %   marginalized.  r.info: Fp (the expansions with the terms up to n^(-j/2)), kappa, err,
      %   nG ('nG' is the only engine option of this path).  Where the continuity-corrected point
      %   is not inside the convex hull of the support (P(S >= 0) of a count) there is no
      %   saddlepoint and an error with identifier 'spa_cond:support' is raised.
      [order, lower, signs, opts] = parseopts('tailprob', obj.d, varargin{:});
      [lattice, sinhmode, opts] = latticeopts('tailprob', opts);
      x = reshape(x, 1, []);
      if numel(x) ~= obj.d, error('CGF:x', 'tailprob: x must have length %d', obj.d); end
      if ~all(isfinite(x)), error('CGF:x', 'tailprob: x must be finite'); end
      sg = cornersigns(obj.d, lower, signs);
      Kr = obj; if any(sg ~= 1), Kr = reflect(obj, sg); end
      xr = sg .* x; maxdeg = 2*order + 1;
      Kf = cgf_kfun(Kr, maxdeg);
      if lattice
         % Corollary 1 of the conditional paper (B empty): the same computation as
         % conditional_tailprob(K, x, 1:d, 'lattice', true)
         nG = 64; if isfield(opts, 'nG'), nG = opts.nG; end
         [F, ~, ci] = spa_cond(Kf, xr, 1:obj.d, order, struct('density', false, 'lattice', true, ...
            'sinh', sinhmode, 'nG', nG, 'Kwhite', cgf_kwhite(Kr, maxdeg)));
         r = struct('F', F, 'tau', sg.*ci.tau, 'status', ci.status, ...
            'info', struct('Fp', ci.Pp, 'kappa', ci.kappaP, 'err', ci.err, 'nG', nG));
         return
      end
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
      %   r = density(K, k, 'order', M, 'lattice', true): S integer-valued and x = k integer; r.f is
      %   the probability P(S = k), the same expansion at k, the saddlepoint approximation of
      %   Daniels (1987) at order one (Corollary 1 of Broda and Kan, "Conditional and lattice
      %   saddlepoint expansions of arbitrary order", without conditioning components); it is
      %   computed as conditional_density with A = 1:d, in whitened coordinates in any dimension,
      %   and the numbers are the same.
      %   r = density(K, x, 'order', M, 'normalize', true) divides the expansion of every order M
      %   by its total mass c_M = int f_M(x) dx (Daniels's normalization), returned in r.info.mass;
      %   the integral is computed in saddlepoint space, x = K'(s), dx = det K''(s) ds, so that the
      %   saddlepoint is known at every quadrature node and no saddlepoint equation is solved there:
      %   by adaptive Gauss-Kronrod quadrature (the algorithm of QuadGK.jl, relative tolerance
      %   1e-12 in one dimension and 1e-9 in more) over the domain of K, nested over the
      %   coordinates in more than one dimension (each inner line centred at the regression of the
      %   inner coordinates on the outer ones under K''(0)), the tails truncated where the leading
      %   term has fallen by e^-50 relative to its largest value.  The masses do not depend on x:
      %   'normalize', r.info.mass (a numeric vector) reuses them.  One quadrature costs a few
      %   hundred evaluations of the expansion in one dimension and the square of that in two; in
      %   more dimensions it is expensive.  For the normal distribution c_M = 1; for the gamma
      %   distribution with shape a, c_1 = Gamma(a) e^a a^(1/2-a)/sqrt(2 pi), and the normalized
      %   expansion of every order is exact.
      [order, ~, ~, opts] = parseopts('density', obj.d, varargin{:});
      [lattice, ~, opts] = latticeopts('density', opts);
      [nrm, opts] = normopts('density', opts);
      if ~isequal(nrm, false)
         if lattice, error('CGF:normalize', 'density: normalize is for continuous variables'); end
         if isequal(nrm, true), mass = mass_density(obj, order); else, mass = double(nrm(:)); end
         if numel(mass) < order, error('CGF:normalize', 'density: normalize must be true or a vector of %d masses', order); end
         args = [fieldnames(opts)'; struct2cell(opts)'];
         r = density(obj, x, 'order', order, args{:});
         r.f = r.f./mass(1:order); r.info.mass = mass(1:order);
         return
      end
      x = reshape(x, 1, []);
      if numel(x) ~= obj.d, error('CGF:x', 'density: x must have length %d', obj.d); end
      maxdeg = 2*order + 1; Kf = cgf_kfun(obj, maxdeg);
      if lattice
         % P(S = k): the density expansion at k (Corollary 1 of the conditional paper, B empty),
         % by the computation of conditional_density(K, x, 1:d, 'lattice', true)
         [~, f, ci] = spa_cond(Kf, x, 1:obj.d, order, struct('tail', false, 'Kwhite', cgf_kwhite(obj, maxdeg)));
         r = struct('f', f, 'tau', ci.tauf, 'status', ci.status, 'info', struct('kappa', ci.kappaf));
         return
      end
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
      %   For one ratio, 'normalize', true divides the expansion of every order by its total mass,
      %   returned in r.info.mass ('normalize', r.info.mass reuses it): the integral over r is
      %   computed in saddlepoint space, over the saddlepoint shat along the curve
      %   s K_1 + t K_2 = 0, t = -r shat, on which r = K_1/K_2 (see ratio_quantile), with
      %   dr = (dr/dshat) dshat, by adaptive Gauss-Kronrod quadrature (relative tolerance 1e-10),
      %   the tails truncated where the leading term has fallen by e^-50; each node gives its t by
      %   a one-dimensional Newton iteration on the curve, and no saddlepoint equation in r is
      %   solved.
      [order, ~, ~, opts] = parseopts('ratio_density', numel(r0), varargin{:});
      [nrm, opts] = normopts('ratio_density', opts);
      if ~isequal(nrm, false)
         if numel(r0) ~= 1, error('CGF:normalize', 'ratio_density: normalize is for one ratio'); end
         if isequal(nrm, true), mass = mass_ratio(ratiocgf(obj, 1, 'ratio_density'), order); else, mass = double(nrm(:)); end
         if numel(mass) < order, error('CGF:normalize', 'ratio_density: normalize must be true or a vector of %d masses', order); end
         args = [fieldnames(opts)'; struct2cell(opts)'];
         r = ratio_density(obj, r0, 'order', order, args{:});
         r.f = r.f./mass(1:order); r.info.mass = mass(1:order);
         return
      end
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

   %---------------------------------------------------------------- conditional expansions
   function r = conditional_tailprob(obj, x, A, varargin)
      %CONDITIONAL_TAILPROB saddlepoint expansions of orders 1..order of the conditional
      %   distribution function P(X_A >= x_A | X_B = x_B) (componentwise in A) at the point x (a
      %   vector of length d); A holds the indices of the components whose tail is wanted, the
      %   others, B, are conditioned on (Broda and Kan, "Conditional and lattice saddlepoint
      %   expansions of arbitrary order", Theorem 3).
      %   r = conditional_tailprob(K, x, A, 'order', M)
      %   r.F(M): the numerator, the orthant-probability integral with the poles in the components
      %   of A only, with the terms of orders n^(-m/2), m <= 2M-1, over the density expansion of
      %   X_B with M terms; its relative error is O(n^-M) (the n absorbed in K).  With one tail
      %   component, r.F(1) adds the terms of order n^(-1/2) to the conditional analogue
      %   (u/z) exp((u^2-w^2)/2) Phibar(u) of the Hauschildt-Robinson approximation, in the
      %   notation of Skovgaard (1987).  With B empty the expansion is that of tailprob.
      %   'lower', true gives P(X_A <= x_A | X_B = x_B), and 'signs', s (one +-1 per component of
      %   A) a mixed corner; the tail components are reflected (reflect).  When the saddlepoint has
      %   negative components in A the expansion is applied through the inclusion-exclusion
      %   identity of Remark 1 of the paper, every term with its own saddlepoint.
      %   'lattice', true: the components in A are integer-valued and so is x(A); r.F is then
      %   P(S_A >= k_A | S_B = k_B) by Theorem 5, with the continuity correction and the
      %   amplitude prod_{j in A} s_j/(2 sinh(s_j/2)) expanded about the saddlepoint ('sinh',
      %   'saddle', the default) or about the origin ('sinh', 'origin', Kolassa 2003).  'nG'
      %   (64): Gauss-Legendre nodes per panel of the orthant integrals.
      %   r.tau: the saddlepoint of K at x (at the corrected point for lattice variables);
      %   r.taut: that of the marginal of X_B, embedded as (0_A, s_B); r.status: 'ok' or a
      %   description of a problem (see spa_cond); r.info: Fp (the expansions with the terms of
      %   the numerator up to n^(-j/2), j = 0..2*order-1, r.F(M) = Fp(2M); Fp(1) is the leading
      %   term), kappa (the cancellation indicator of the numerator), kappa_den (that of the
      %   denominator), err (the change with half the quadrature nodes) and nG.
      %   All Taylor coefficients are generated by Taylor-mode differentiation directly in the
      %   canonical coordinates of Section 5.2 of the paper, Y_A = X_A - C X_B, Y_B = L^{-1} X_B, in
      %   which the tail components are uncorrelated with the conditioning components under the
      %   tilted distribution and the latter are whitened; this is what keeps the terms from
      %   cancelling in double precision.  See spa_cond (folder engine) for the details and the
      %   derivative interface.
      [order, lower, signs, lattice, sinhmode, nG] = condopts('conditional_tailprob', varargin{:});
      dim = obj.d; x = reshape(x, 1, []);
      if numel(x) ~= dim, error('CGF:x', 'conditional_tailprob: x must have length %d', dim); end
      Av = condset('conditional_tailprob', A, dim);
      sg = condsigns('conditional_tailprob', dim, Av, lower, signs);
      Kr = obj; if any(sg ~= 1), Kr = reflect(obj, sg); end
      xr = sg .* x; maxdeg = 2*order + 1;
      [F, ~, ci] = spa_cond(cgf_kfun(Kr, maxdeg), xr, Av, order, struct('density', false, 'lattice', lattice, ...
         'sinh', sinhmode, 'nG', nG, 'Kwhite', cgf_kwhite(Kr, maxdeg)));
      r = struct('F', F, 'tau', sg.*ci.tau, 'taut', ci.taut, 'status', ci.status, 'info', ...
         struct('Fp', ci.Pp, 'kappa', ci.kappaP, 'kappa_den', ci.kappaB, 'err', ci.err, 'nG', nG));
   end
   function r = conditional_density(obj, x, A, varargin)
      %CONDITIONAL_DENSITY saddlepoint expansions of orders 1..order of the conditional density
      %   f(x_A | x_B) of the components A of X given the others, B, at the point x (Theorem 4 of
      %   Broda and Kan, "Conditional and lattice saddlepoint expansions of arbitrary order"): the
      %   ratio of the density expansions of X at its saddlepoint and of X_B at its own, both with
      %   the terms m <= M-1 at order M, relative error O(n^-M).  The leading term is the double
      %   saddlepoint density approximation of Barndorff-Nielsen and Cox (1979).  Unlike the
      %   distribution function, the expansion holds for saddlepoints in any orthant.
      %   r = conditional_density(K, x, A, 'order', M); with 'lattice', true (integer-valued S)
      %   the same formula is P(S_A = k_A | S_B = k_B) (Theorem 5), evaluated at the uncorrected
      %   point x; the option only records this interpretation.  r.f(M); r.tau: the saddlepoint
      %   of K at x; r.taut: that of the marginal of X_B, embedded as (0_A, s_B); r.status;
      %   r.info: kappa (the cancellation indicator of the density expansion of X) and kappa_den
      %   (that of X_B).  Both expansions are evaluated in whitened coordinates, with the Taylor
      %   coefficients generated there directly.
      [order, ~, ~, ~, ~, ~, used] = condopts('conditional_density', varargin{:});
      if any(ismember(used, {'lower', 'signs', 'sinh', 'ng'})), error('CGF:opts', 'conditional_density: the options are order and lattice'); end
      dim = obj.d; x = reshape(x, 1, []);
      if numel(x) ~= dim, error('CGF:x', 'conditional_density: x must have length %d', dim); end
      Av = condset('conditional_density', A, dim);
      maxdeg = 2*order + 1;
      [~, f, ci] = spa_cond(cgf_kfun(obj, maxdeg), x, Av, order, struct('tail', false, 'Kwhite', cgf_kwhite(obj, maxdeg)));
      r = struct('f', f, 'tau', ci.tauf, 'taut', ci.taut, 'status', ci.status, ...
         'info', struct('kappa', ci.kappaf, 'kappa_den', ci.kappaB));
   end

   %---------------------------------------------------------------- quantiles
   function r = quantile(obj, p, varargin)
      %QUANTILE the quantile of the saddlepoint distribution function of order 'order' of a
      %   univariate random variable with cumulant generating function K (Hauschildt-Robinson
      %   expansion, tailprob): the point r.q with P(X <= q) = p under the expansion ('lower',
      %   false: P(X >= q) = p, for upper quantiles of small probability, which are then computed
      %   directly).  r = quantile(K, p, 'order', 3, 'lower', true, 'tol', 1e-12).
      %   The root search runs in saddlepoint space: the point is parametrized by its saddlepoint,
      %   x = K'(s), so that each iterate gives the point and its saddlepoint at once and no
      %   saddlepoint equation is solved; the derivative of the distribution function in s is the
      %   density expansion of the same order times K''(s), and the Newton iteration starts at
      %   s = 0 and is safeguarded by bracketing and bisection, with steps limited to four standard
      %   deviations and pulled back into the domain of K.  Converged when the distribution
      %   function is within tol*min(p, 1-p) of p.  r.q; r.tau, the saddlepoint of q; r.F, the
      %   value of the expansion there; r.iterations; r.status ('ok', or 'bracket at rounding
      %   level' when the bracket of the root has shrunk to rounding level before the tolerance
      %   was met).  For continuous variables (the expansion of a lattice variable is a step
      %   function of k).  A method of the class CGF, so that quantile of a numeric array is
      %   MATLAB's own.
      [order, lower, tol] = quantopts('quantile', varargin{:});
      if obj.d ~= 1, error('CGF:dim', 'quantile: K must be univariate (see ratio_quantile and studentized_quantile)'); end
      if ~(isnumeric(p) && isscalar(p) && isreal(p) && p > 0 && p < 1), error('CGF:prob', 'quantile: p must be in (0, 1)'); end
      r = quantile_uni(obj, double(p), order, lower, tol);
   end
   function r = ratio_quantile(obj, p, varargin)
      %RATIO_QUANTILE the quantile of the saddlepoint distribution function of order 'order' of
      %   the ratio R = X_1/X_2, X_2 > 0, with K the joint cumulant generating function of
      %   (X_1, X_2) (ratio_tailprob): r.r with P(R <= r) = p under the expansion ('lower', false:
      %   P(R >= r) = p).  r = ratio_quantile(K, p, 'order', 3, 'lower', true, 'tol', 1e-12).
      %   The search runs in saddlepoint space, over the saddlepoint shat of W = X_1 - r X_2: the
      %   saddlepoints of all r lie on the curve s K_1(s,t) + t K_2(s,t) = 0, t = -r shat, on which
      %   r = K_1/K_2, so that each trial shat gives its t (by a one-dimensional Newton iteration
      %   on the curve, continued from the previous iterate), its r and its distribution function
      %   directly, without the saddlepoint equation of W for a trial r.  The Newton iteration in
      %   shat uses the derivative f_R(r) dr/dshat, with the density expansion of the ratio of the
      %   same order (ratio_density) and dr/dshat along the curve, safeguarded as in quantile.
      %   r.r; r.shat, the saddlepoint; r.F, the value of the expansion at r.r; r.iterations;
      %   r.status.
      [order, lower, tol] = quantopts('ratio_quantile', varargin{:});
      if obj.d ~= 2, error('CGF:dim', 'ratio_quantile: K must be the joint cumulant generating function of the numerator and the denominator (d = 2)'); end
      if ~(isnumeric(p) && isscalar(p) && isreal(p) && p > 0 && p < 1), error('CGF:prob', 'ratio_quantile: p must be in (0, 1)'); end
      r = quantile_ratio(obj, double(p), order, lower, tol);
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
      %   'normalize', true divides the expansion of every order by its total mass, returned in
      %   r.info.mass ('normalize', r.info.mass reuses it).  The integral over q is computed in
      %   saddlepoint space, over shat along the locus s K_s + 2 t K_t = 0 (studentized_locus), with
      %   dq = (dq/dshat) dshat, by adaptive Gauss-Kronrod quadrature (relative tolerance 1e-9) on
      %   the two sides of shat = 0 (where t = 0), the tails truncated where the leading term has
      %   fallen by e^-50; each node gives t by the one-dimensional Newton iteration of the locus,
      %   so that no saddlepoint equation in q is solved.  The expansions at the nodes are refined
      %   where they cancel, as with 'refine', true, which is what makes the masses of the higher
      %   orders meaningful in the tails, where |t| is large.
      studcheck(obj, 'studentized_density');
      [order, ~, saddle, ~, refine, prec, nrm] = studopts('studentized_density', varargin{:});
      q = scalarq(q, 'studentized_density');
      if ~isequal(nrm, false)
         if isequal(nrm, true), mass = mass_stud(obj, order); else, mass = double(nrm(:)); end
         if numel(mass) < order, error('CGF:normalize', 'studentized_density: normalize must be true or a vector of %d masses', order); end
      end
      if strcmp(prec, 'dd')
         if isempty(saddle), saddle = stud_saddle(obj, q); end
         r = stud_density_dd(obj, q, order, saddle);
      else
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
      if ~isequal(nrm, false), r.f = r.f./mass(1:order); r.info.mass = mass(1:order); end
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
function [order, lower, saddle, tol, refine, prec, nrm] = studopts(name, varargin)
   order = 3; lower = false; saddle = []; tol = 1e-10; refine = true; prec = 'double'; nrm = false;
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
         case 'normalize', nrm = normvalue(name, varargin{i+1});
         otherwise, error('CGF:opts', '%s: unknown option %s', name, varargin{i});
      end
   end
   if ~(isscalar(order) && order >= 1 && order == round(order)), error('CGF:order', '%s: order must be a positive integer', name); end
end
function [lattice, sinhmode, opts] = latticeopts(name, opts)
   % the options lattice and sinh of tailprob and density, removed from the engine options
   lattice = false; sinhmode = 'saddle';
   if isfield(opts, 'lattice'), lattice = logical(opts.lattice); opts = rmfield(opts, 'lattice'); end
   if isfield(opts, 'sinh'), sinhmode = sinhvalue(name, opts.sinh); opts = rmfield(opts, 'sinh'); end
end
function v = sinhvalue(name, v)
   if ~((ischar(v) || isstring(v)) && any(strcmp(v, {'saddle', 'origin'}))), error('CGF:opts', '%s: sinh must be ''saddle'' or ''origin''', name); end
   v = char(v);
end
function [nrm, opts] = normopts(name, opts)
   % the option normalize, removed from the engine options
   nrm = false;
   if isfield(opts, 'normalize'), nrm = normvalue(name, opts.normalize); opts = rmfield(opts, 'normalize'); end
end
function v = normvalue(name, v)
   % true or false (logical), or a numeric vector of masses
   if islogical(v) && isscalar(v), return, end
   if isnumeric(v) && ~isempty(v) && all(isfinite(v(:))) && all(v(:) > 0), v = double(v(:)); return, end
   error('CGF:normalize', '%s: normalize must be true, false or a vector of masses', name);
end
function [order, lower, signs, lattice, sinhmode, nG, used] = condopts(name, varargin)
   order = 3; lower = false; signs = []; lattice = false; sinhmode = 'saddle'; nG = 64; used = {};
   if mod(numel(varargin), 2), error('CGF:opts', '%s: options must be name-value pairs', name); end
   for i = 1:2:numel(varargin)
      key = lower_(varargin{i}); val = varargin{i+1}; used{end+1} = key; %#ok<AGROW>
      switch key
         case 'order', order = val;
         case 'lower', lower = logical(val);
         case 'signs', signs = val;
         case 'lattice', lattice = logical(val);
         case 'sinh', sinhmode = sinhvalue(name, val);
         case 'ng', nG = val;
         otherwise, error('CGF:opts', '%s: unknown option %s', name, varargin{i});
      end
   end
   if ~(isscalar(order) && order >= 1 && order == round(order)), error('CGF:order', '%s: order must be a positive integer', name); end
   if ~(isscalar(nG) && nG >= 8 && nG == round(nG)), error('CGF:opts', '%s: nG must be an integer of at least 8', name); end
end
function A = condset(name, A, d)
   % A as a sorted row of distinct indices in 1:d
   A = reshape(A, 1, []);
   if isempty(A) || ~isnumeric(A) || any(A ~= round(A)) || any(A < 1 | A > d) || numel(unique(A)) ~= numel(A)
      error('CGF:A', '%s: A must be a nonempty set of distinct indices in 1:%d', name, d);
   end
   A = sort(double(A));
end
function sg = condsigns(name, d, A, lower, signs)
   % the sign vector of the tail components for lower and mixed tails (ones on B)
   sg = ones(1, d);
   if isempty(signs)
      if lower, sg(A) = -1; end
      return
   end
   if numel(signs) ~= numel(A), error('CGF:signs', '%s: signs must have one entry per tail component (%d)', name, numel(A)); end
   if ~all(signs(:) == 1 | signs(:) == -1), error('CGF:signs', '%s: signs must be +-1', name); end
   sg(A) = reshape(signs, 1, []);
end
function [order, lower, tol] = quantopts(name, varargin)
   order = 3; lower = true; tol = 1e-12;
   if mod(numel(varargin), 2), error('CGF:opts', '%s: options must be name-value pairs', name); end
   for i = 1:2:numel(varargin)
      switch lower_(varargin{i})
         case 'order', order = varargin{i+1};
         case 'lower', lower = logical(varargin{i+1});
         case 'tol', tol = varargin{i+1};
         otherwise, error('CGF:opts', '%s: unknown option %s', name, varargin{i});
      end
   end
   if ~(isscalar(order) && order >= 1 && order == round(order)), error('CGF:order', '%s: order must be a positive integer', name); end
   if ~(isscalar(tol) && tol > 0), error('CGF:opts', '%s: tol must be positive', name); end
end
function sg = cornersigns(d, lower, signs)
   if ~isempty(signs), sg = reshape(signs, 1, []); elseif lower, sg = -ones(1, d); else, sg = ones(1, d); end
end
