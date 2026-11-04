function [P,part,info] = kl_bivar(Kfun,x,ord,hrel)
%KL_BIVAR First-order bivariate tail approximation of Kolassa and Li (2010), P[X1 >= x1, X2 >= x2],
%   as written out for d = 2 in Li (2009, Chapter 3), with n = 1 (pass the cgf of the vector itself).
%   Kfun(nu,t): derivative of order nu = [nu1 nu2] of the cgf at t, NaN outside the domain.
%   ord = [1 2] (default) or [2 1]: order of the coordinates in the nested signed roots.
%   P = part(1)+part(2)+part(3)+part(4): main quadratic term C*Pbar(t;G), its cubic correction
%   (third-degree terms of g, Kolassa 2003), I^{1}, I^{2}.  info: tau-hat, w-hat, w-check, v', v''.
%   P is NaN when a needed value of the cgf is outside its domain.  The second derivative v'' of the
%   nested signed root is computed by central differences with step hrel*max(1,|tau1|) (default
%   hrel = 1e-3).  The formula has removable singularities on the lines tau1 = 0 and
%   x2 = E(X2); this routine evaluates it directly in double precision, and near those lines the
%   result is dominated by rounding (Table S7 of the Supplementary Material).  Requires the Statistics and Machine
%   Learning Toolbox (mvncdf).  Used by kl_table.
%   Raymond Kan (raymond.kan@rotman.utoronto.ca), 2026. Programs accompanying Broda and Kan,
%   "Uniform saddlepoint expansions for multivariate tail probabilities"; free to use and modify with a citation of the paper.
if nargin < 3, ord = [1 2]; end
if nargin < 4, hrel = 1e-3; end
if isequal(ord,[2 1])
   K = @(nu,t) Kfun(nu([2 1]),t([2 1])); x = x([2 1]);
else
   K = Kfun;
end
part = nan(1,4); P = NaN; info = struct();
tau = saddle2(K,x,[0 0]);
K0 = K([0 0],tau)-tau*x';
t20 = inner(K,0,x(2),tau(2));                          % tau~2(0)
if isnan(t20), info.reason = 'tau~2(0)'; return; end
f0 = K([0 0],[0 t20])-t20*x(2);                        % -w2hat^2/2
w2h = sign(t20)*sqrt(max(-2*f0,0));
w1h = sign(tau(1))*sqrt(max(-2*(K0-f0),0));
Kt10 = K([0 0],[tau(1) 0]);
if isnan(Kt10), info.reason = 'K(tau1,0)'; return; end
w2c = w2h-sign(tau(2))*sqrt(max(2*(Kt10-tau(1)*x(1)-K0),0));   % w-check_2
H = [K([2 0],tau) K([1 1],tau); K([1 1],tau) K([0 2],tau)];
fpp = H(1,1)-H(1,2)^2/H(2,2);                          % f''(tau1-hat)
% v = w~2 as a function of w1, through tau1:  w1(tau1), W(tau1)
w1f = @(s) w1h+sign(s-tau(1))*sqrt(max(2*(prof(K,s,x,tau(2))-K0),0));
Wf  = @(s) w2h+sign(-inner(K,s,x(2),tau(2)))*sqrt(max(2*(K([0 0],[s 0])-s*x(1)-prof(K,s,x,tau(2))),0));
vp = (K([1 0],[tau(1) 0])-x(1))/((w2c-w2h)*sqrt(fpp));             % v'(w1-hat)
h = hrel*max(1,abs(tau(1)));
s = tau(1)+h*(-2:2);
w1v = arrayfun(w1f,s); Wv = arrayfun(Wf,s);
d1w = (w1v(4)-w1v(2))/(2*h); d2w = (w1v(4)-2*w1v(3)+w1v(2))/h^2;
d1W = (Wv(4)-Wv(2))/(2*h);   d2W = (Wv(4)-2*Wv(3)+Wv(2))/h^2;
vp_fd = d1W/d1w;
vpp = (d2W*d1w-d1W*d2w)/d1w^3;
info = struct('tau',tau,'t20',t20,'w1h',w1h,'w2h',w2h,'w2c',w2c,'vp',vp,'vp_fd',vp_fd,'vpp',vpp);
if ~all(isfinite([vp vpp w1h w2h w2c])), info.reason = 'nonfinite (0/0)'; return; end
G = [1+vp^2 vp; vp 1];
uh = [w1h, w2h-w2c];
tt = (G*uh')';
C = exp(-0.5*(w1h^2+w2h^2)+0.5*uh*G*uh');
Pbar = mvncdf(-tt,[0 0],G);
% cubic correction: g_111 = 3 v' v'', g_112 = v'', others 0;  sum g_nu/nu! I(nu)
I = Inu3(G,tt,uh);
part(1) = C*Pbar;
part(2) = C*(3*vp*vpp/6*I(4,1)+vpp/2*I(3,2));
% I^{1} (Li 2009, (3.1.23)) and I^{2} ((3.1.15))
hh = 1/(tau(1)*sqrt(fpp))-1/w1h;
a1 = 1+vp^2;
part(3) = C*hh/sqrt(a1)*normpdf(tt(1)/sqrt(a1))*normcdf((w2h-w2c)/sqrt(a1),'upper');
K22 = K([0 2],[0 t20]);
part(4) = (1/(t20*sqrt(K22))-1/w2h)*normcdf(w1h,'upper')*normpdf(w2h);
P = sum(part);
info = struct('tau',tau,'t20',t20,'w1h',w1h,'w2h',w2h,'w2c',w2c,'vp',vp,'vp_fd',vp_fd,'vpp',vpp,'w1v',w1v,'Wv',Wv);
end

function v = prof(K,s,x,g0)
%  f(s) = min_g K(s,g) - s x1 - g x2
g = inner(K,s,x(2),g0);
v = K([0 0],[s g])-s*x(1)-g*x(2);
end

function g = inner(K,s,x2,g0)
%  solve K_2(s,g) = x2 by Newton with step halving (NaN if it fails)
g = g0; if ~isfinite(K([0 0],[s g])), g = 0; end
for it = 1:200
   f1 = K([0 1],[s g])-x2; f2 = K([0 2],[s g]);
   st = -f1/f2; lam = 1;
   while ~isfinite(K([0 0],[s g+lam*st])) && lam > 1e-12, lam = lam/2; end
   g = g+lam*st;
   if abs(lam*st) < 1e-14*(1+abs(g)), break; end
end
if ~isfinite(K([0 0],[s g])) || abs(K([0 1],[s g])-x2) > 1e-8*(1+abs(x2)), g = NaN; end
end

function I = Inu3(S,t,sh)
%  I(nu) = I(S,-1,nu,t,sh), |nu| <= 3, from the binomial sum and the derivatives J(r) of Pbar(t;S);
%  I(nu1+1,nu2+1).  Analytic in t and sh (no sign restriction on sh).
L = 3; Si = inv(S);
He = zeros(L+1,L+1); He(1,1) = 1; y = Si*t';
for k = 1:L
   for n1 = 0:k
      n2 = k-n1; if n1 > 0, j = 1; pr = [n1-1 n2]; else, j = 2; pr = [n1 n2-1]; end
      v = y(j)*He(pr(1)+1,pr(2)+1);
      for kk = 1:2
         if pr(kk) > 0, pp = pr; pp(kk) = pp(kk)-1; v = v-Si(j,kk)*pr(kk)*He(pp(1)+1,pp(2)+1); end
      end
      He(n1+1,n2+1) = v;
   end
end
phi2 = exp(-t*Si*t'/2)/(2*pi*sqrt(det(S)));
J = zeros(L+1,L+1);                        % J(r1+2,r2+2), r >= -1
for a = 0:L-1, for b = 0:L-1-a, J(a+2,b+2) = phi2*He(a+1,b+1); end, end
for i = 1:2
   j = 3-i; Sjj = S(j,j); Sij = S(i,j); cv = S(i,i)-Sij^2/Sjj;
   g = normpdf(t(j)/sqrt(Sjj))/sqrt(Sjj)*normcdf((t(i)-Sij/Sjj*t(j))/sqrt(cv),'upper');
   Jr = zeros(L,1); Jr(1) = g;
   for rj = 0:L-2
      if i == 1, Jri = J(2,rj+2); else, Jri = J(rj+2,2); end
      v = t(j)/Sjj*Jr(rj+1)-Sij/Sjj*Jri; if rj > 0, v = v-rj/Sjj*Jr(rj); end
      Jr(rj+2) = v;
   end
   if i == 1, J(1,2:L+1) = Jr'; else, J(2:L+1,1) = Jr; end
end
J(1,1) = mvncdf(-t,[0 0],S);
I = zeros(L+1,L+1);
for k = 0:L
   for n1 = 0:k
      n2 = k-n1;
      w1 = arrayfun(@(m) nchoosek(n1,m),0:n1).*(-sh(1)).^(0:n1);
      w2 = arrayfun(@(m) nchoosek(n2,m),0:n2).*(-sh(2)).^(0:n2);
      I(n1+1,n2+1) = w1*J(n1+1:-1:1,n2+1:-1:1)*w2';
   end
end
end

function tau = saddle2(K,x,tau0)
tau = tau0;
obj = @(t) K([0 0],t)-t*x';
f = obj(tau);
for it = 1:200
   g = [K([1 0],tau) K([0 1],tau)]-x;
   H = [K([2 0],tau) K([1 1],tau); K([1 1],tau) K([0 2],tau)];
   st = -(H\g')';
   if norm(st) < 1e-14*(1+norm(tau)), break; end
   lam = 1;
   while true
      tn = tau+lam*st; fn = obj(tn);
      if isfinite(fn) && fn <= f+1e-4*lam*(g*st'), break; end
      lam = lam/2; if lam < 1e-14, break; end
   end
   if norm(lam*st) < 1e-13*(1+norm(tau)), tau = tn; break; end
   tau = tn; f = fn;
end
% the line search stalls at about sqrt(eps) because of rounding in f; polish by full Newton steps on
% the gradient, which v'' (a second difference of signed roots centred at tau) needs
for it = 1:4
   g = [K([1 0],tau) K([0 1],tau)]-x;
   H = [K([2 0],tau) K([1 1],tau); K([1 1],tau) K([0 2],tau)];
   tn = tau-(H\g')';
   if ~isfinite(K([0 0],tn)), break; end
   tau = tn;
end
end
