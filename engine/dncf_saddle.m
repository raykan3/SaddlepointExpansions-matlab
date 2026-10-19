function shat = dncf_saddle(c,m,n,lam1,lam2)
%DNCF_SADDLE Saddlepoint for the doubly noncentral F: the root of
%   K_W'(s) = m/(1-2s) + lam1/(1-2s)^2 - c n/(1+2cs) - c lam2/(1+2cs)^2 = 0
%   in (-1/(2c), 1/2), W = Q1 - c Q2, which is unique because K_W'' > 0, K_W' -> -inf at the
%   left end and +inf at the right end.  Starting value: clearing denominators gives the cubic
%   s^3 + a2 s^2 + a1 s + a0 = 0 with
%      a  = 8 c^2 (m+n),
%      a2 = -4c{c lam1 + c m + 2c n - lam2 - 2m - n}/a,
%      a1 =  2{c^2 n - 2c(lam1+lam2+m+n) + m}/a,
%      a0 = -{m + lam1 - c(n+lam2)}/a,
%   whose middle root (Butler and Paolella, 2002) is computed by the trigonometric formula;
%   it is then refined by safeguarded Newton iteration (bisection on the bracket when a Newton
%   step leaves it), which also repairs the cases in which the trigonometric formula loses
%   accuracy or returns a root outside the domain.
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Saddlepoint expansions of arbitrary order for multivariate distribution functions"; free to use and modify with a citation of the paper.
lo = -1/(2*c); hi = 1/2;
a  = 8*c^2*(m+n);
a2 = -4*c*(c*lam1+c*m+2*c*n-lam2-2*m-n)/a;
a1 = 2*(c^2*n-2*c*(lam1+lam2+m+n)+m)/a;
a0 = -(m+lam1-c*(n+lam2))/a;
P = a1-a2^2/3; Q = 2*a2^3/27-a1*a2/3+a0;              % depressed cubic t^3 + P t + Q
s = NaN;
if P < 0
   arg = 3*Q/(2*P)*sqrt(-3/P); arg = max(-1,min(1,arg));
   t = 2*sqrt(-P/3)*cos(acos(arg)/3-2*pi*(0:2)/3)-a2/3;
   t = sort(t); s = t(2);
end
if ~(s > lo && s < hi)
   s = (m+lam1-c*(n+lam2))/(2*(m+lam1)*(1+c)^2/(1+c)+2*c*(n+lam2)); % crude start, then bracket
   s = max(lo+0.25*(hi-lo),min(hi-0.25*(hi-lo),s));
end
g = @(s) m/(1-2*s)+lam1/(1-2*s)^2-c*n/(1+2*c*s)-c*lam2/(1+2*c*s)^2;
dg = @(s) 2*m/(1-2*s)^2+4*lam1/(1-2*s)^3+2*c^2*n/(1+2*c*s)^2+4*c^2*lam2/(1+2*c*s)^3;
for it = 1:100
   gs = g(s);
   if gs < 0, lo = s; else, hi = s; end
   step = -gs/dg(s); sn = s+step;
   if ~(sn > lo && sn < hi), sn = (lo+hi)/2; end   % bisection when Newton leaves the bracket
   if abs(sn-s) <= 4*eps*max(abs(s),abs(sn)) || sn == s, s = sn; break, end
   s = sn;
end
shat = s;
end
