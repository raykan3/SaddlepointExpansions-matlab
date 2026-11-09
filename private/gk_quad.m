function I = gk_quad(f, pts, rtol)
%GK_QUAD Adaptive Gauss-Kronrod quadrature of a vector-valued function over the union of the
%   intervals (pts(1), pts(2)), (pts(2), pts(3)), ..., by the algorithm of QuadGK.jl with order 7
%   (the 15-point Kronrod rule with the embedded 7-point Gauss rule for the error estimate), so
%   that the normalizations agree with those of the Julia package: the segment with the largest
%   error estimate (the Euclidean norm of the difference of the two rules) is bisected until the
%   sum of the estimates is at most rtol times the Euclidean norm of the integral (or 1e7
%   evaluations, or a segment at rounding level).  An infinite end point is handled by the same
%   changes of variable, x = s0 +- t/(1 - t) for one infinite end (s0 the finite one), and
%   x = t/(1 - t^2) for two.  f(x) returns a column vector; it is never evaluated at the ends.
xk = [0.991455371120812639206854697526329; 0.949107912342758524526189684047851; ...
      0.864864423359769072789712788640926; 0.741531185599394439863864773280788; ...
      0.586087235467691130294144845693013; 0.405845151377397166906606412076961; ...
      0.207784955007898467600689403773245; 0];
wk = [0.022935322010529224963732008058970; 0.063092092629978553290700663189204; ...
      0.104790010322250183839876322541518; 0.140653259715525918745189590510238; ...
      0.169004726639267902826583426598550; 0.190350578064785409913256402421014; ...
      0.204432940075298892414161999234649; 0.209482141084727828012999174891714];
wg = [0.129484966168869693270611432679082; 0.279705391489276667901467771423780; ...
      0.381830050505118944950369775488975; 0.417959183673469387755102040816327];
x = -xk;                                          % the nodes in [-1, 0], as in QuadGK
pts = reshape(double(pts), 1, []);
s1 = pts(1); s2 = pts(end); g = f;
if isinf(s1) || isinf(s2)
   if isinf(s1) && isinf(s2)                      % x = t/(1 - t^2)
      g = @(t) inf2(f, t);
      tp = pts;
      for i = 1:numel(pts)
         if isinf(pts(i)), tp(i) = sign(pts(i)); else, tp(i) = 2*pts(i)/(1 + hypot(1, 2*pts(i))); end
      end
      pts = tp;
   elseif isinf(s1)                               % x = s0 - t/(1 - t), s0 = s2
      s0 = s2; g = @(t) inf1(f, t, s0, -1);
      pts = fliplr(1./(1 + 1./(s0 - pts)));
   else                                           % x = s0 + t/(1 - t), s0 = s1
      s0 = s1; g = @(t) inf1(f, t, s0, 1);
      pts = 1./(1 + 1./(pts - s0));
   end
end
maxevals = 1e7;
ns = numel(pts) - 1;
sa = pts(1:ns); sb = pts(2:ns+1); SE = zeros(1, ns); SI = [];
for i = 1:ns
   [Ik, E] = rule(sa(i), sb(i));
   if i == 1, SI = zeros(numel(Ik), ns); end
   SI(:, i) = Ik; SE(i) = E; %#ok<AGROW>
end
[I, E] = resum(SI, SE);
numevals = 15*ns;
if numevals >= maxevals || E <= 0 || E <= rtol*norm(I), return, end
while E > 0 && E > rtol*norm(I) && numevals < maxevals
   [~, k] = max(SE);
   a = sa(k); b = sb(k); mid = (a + b)/2;
   if roundoff(a, mid) || roundoff(mid, b), break, end
   [I1, E1] = rule(a, mid); [I2, E2] = rule(mid, b);
   I = (I - SI(:, k)) + I1 + I2;
   E = (E - SE(k)) + E1 + E2;
   numevals = numevals + 30;
   sb(k) = mid; SI(:, k) = I1; SE(k) = E1;
   sa(end+1) = mid; sb(end+1) = b; SI(:, end+1) = I2; SE(end+1) = E2; %#ok<AGROW>
end
I = resum(SI, SE);

   function [Ik, E] = rule(a, b)
      s = 0.5*(b - a);
      f0 = g(a + s);
      Ig = f0*wg(4);
      Ik = f0*wk(8) + (g(a + (1 + x(7))*s) + g(a + (1 - x(7))*s))*wk(7);
      for j = 1:3
         fg = g(a + (1 + x(2*j))*s) + g(a + (1 - x(2*j))*s);
         fk = g(a + (1 + x(2*j-1))*s) + g(a + (1 - x(2*j-1))*s);
         Ig = Ig + fg*wg(j);
         Ik = Ik + fg*wk(2*j) + fk*wk(2*j-1);
      end
      Ik = Ik*s; Ig = Ig*s;
      E = norm(Ik - Ig);
      if ~isfinite(E), error('CGF:normalize', 'normalize: the integrand produced %g in the interval (%g, %g)', E, a, b); end
   end
   function tf = roundoff(a, b)
      c = 0.5*(b - a);
      tf = a == a + (1 + x(1))*c || b == a + (1 - x(1))*c;
   end
end

function [I, E] = resum(SI, SE)
I = SI(:, 1); E = SE(1);
for i = 2:numel(SE), I = I + SI(:, i); E = E + SE(i); end
end

function v = inf2(f, t)
t2 = t*t; den = 1/(1 - t2);
v = f(t*den)*(1 + t2)*den*den;
end

function v = inf1(f, t, s0, sgn)
den = 1/(1 - t);
if sgn > 0, v = f(s0 + t*den)*den*den; else, v = f(s0 - t*den)*den*den; end
end
