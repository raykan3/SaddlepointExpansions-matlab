# Further reference values from SaddlepointExpansions.jl (version 0.3.0) for the MATLAB tests of the
# conditional expansions, lattice sums, quantiles and normalized densities (test_spx.m), for the
# cases that test/reference_values.jl of the Julia package does not have: normalized densities in
# two dimensions and of a Studentized mean, three tail components and negative saddlepoints of a
# Gaussian vector, the derivative interface of spa_cond with closed-form derivatives, more
# quantiles, and conditional expansions evaluated in BigFloat (256 bits; the double-precision
# MATLAB values are compared with them). The Julia package is not modified.
# Run from this folder:  julia julia_conditional_reference.jl   (writes conditional_reference.csv)
using Pkg
pkg = joinpath(@__DIR__, "..", "..", "SaddlepointExpansions.jl")
Pkg.activate(pkg)
Pkg.instantiate()
using SaddlepointExpansions, LinearAlgebra, Printf
rows = String[]
add(case, q, v) = for (M, x) in enumerate(v); push!(rows, @sprintf("%s,%s,%d,%.17g", case, q, M, Float64(x))); end
# normalized densities: a linear transformation of two independent gamma variables, a bivariate
# normal, and the Studentized mean of the noncentral t
Kl = linear_combination(CGF(t -> -2log(1 - t[1]) - 3log(1 - t[2]), 2; domain=t -> t[1] < 1 && t[2] < 1), [1.0 0.5; 0.3 1.0])
let r = density(Kl, [1.0, 2.0]; order=2, normalize=true); add("norm2", "lc_mass", r.info.mass); add("norm2", "lc_f", r.f); end
add("norm2", "normal_mass", density(normal_cgf([0.0, 1.0], [1.0 0.5; 0.5 2.0]), [0.3, 0.2]; order=2, normalize=true).info.mass)
mu, nu, n = 1.0, 3.0, 16
Kn = sample_mean(CGF(t -> mu*t[1] + t[1]^2/2 - nu/2*log(1 - 2t[2]/nu), 2; domain=t -> t[2] < nu/2), n)
let r = studentized_density(Kn, 1.5; order=2, normalize=true); add("normstud", "mass", r.info.mass); add("normstud", "f", r.f); end
add("normstud", "mass3", studentized_density(Kn, 1.5; order=3, normalize=true).info.mass)
add("norm1", "gamma05_mass", density(gamma_cgf(0.5), 2.0; order=4, normalize=true).info.mass)
add("norm1", "normal_mass", density(normal_cgf(1.0, 2.0), 0.3; order=3, normalize=true).info.mass)
# a Gaussian vector: one, two and three tail components, saddlepoints of every sign
Sig = [2.0 0.6 0.5 0.3; 0.6 1.5 -0.4 0.2; 0.5 -0.4 1.0 0.1; 0.3 0.2 0.1 0.8]; m4 = [0.5, -1.0, 0.2, 0.3]
Kg = normal_cgf(m4, Sig)
for (i, (x, A)) in enumerate((([1.0, 0.0, 0.5, 0.0], [1]), ([-1.0, 0.0, 0.5, 0.0], [1]), ([1.5, 0.0, 0.5, 0.3], [1, 2]),
                            ([0.2, -2.5, 0.5, 0.3], [1, 2]), ([-1.0, -2.5, 0.5, 0.3], [2, 1]), ([1.2, -0.8, 0.9, 0.3], [1, 2, 3])))
    r = conditional_tailprob(Kg, x, A; order=(length(A) == 3 ? 2 : 3))
    add("gauss", "F$i", r.F); add("gauss", "tau$i", r.tau)
    add("gauss", "f$i", conditional_density(Kg, x, A; order=3).f)
end
add("gauss", "Fsigns", conditional_tailprob(Kg, [0.2, -2.5, 0.5, 0.3], [1, 2]; order=2, signs=[-1, 1]).F)
add("gauss", "Flower", conditional_tailprob(Kg, [0.2, -2.5, 0.5, 0.3], [1, 2]; order=2, lower=true).F)
# the derivative interface: closed-form derivatives of the gamma sum, d = 3 (polynomial substitution)
N = 16; a, b, c = 1.0, 2.0, 0.5
function Kf3(nu, t)
    n = sum(nu); p1 = 1 - t[1] - t[2]; p2 = 1 - t[2] - t[3]; p3 = 1 - t[2]
    (p1 <= 0 || p2 <= 0 || p3 <= 0) && return NaN
    n == 0 && return -N*(a*log(p1) + b*log(p2) + c*log(p3))
    v = 0.0
    nu[3] == 0 && (v += a*factorial(n - 1)/p1^n)
    nu[1] == 0 && (v += b*factorial(n - 1)/p2^n)
    (nu[1] == 0 && nu[3] == 0) && (v += c*factorial(n - 1)/p3^n)
    return N*v
end
x3 = [19.0, 14*N/4, 8*N/4]
let r = spa_cond(Kf3, x3, 1, 4)
    add("kf3", "P", r.P); add("kf3", "f", r.f); add("kf3", "Pp", r.Pp); add("kf3", "tau", r.tau); add("kf3", "taut", r.taut)
end
# Dirichlet: two tail components given the total
Kd = CGF(t -> -5log(1 - t[1] - t[3]) - 7.5log(1 - t[2] - t[3]) - 10log(1 - t[3]), 3;
         domain=t -> t[1] + t[3] < 1 && t[2] + t[3] < 1 && t[3] < 1)
for (i, (w1, w2)) in enumerate(((0.35, 0.45), (0.15, 0.40), (0.30, 0.25), (0.15, 0.25)))
    r = conditional_tailprob(Kd, [w1*22.5, w2*22.5, 22.5], [1, 2]; order=4)
    add("dir", "F$i", r.F); add("dir", "f$i", conditional_density(Kd, [w1*22.5, w2*22.5, 22.5], [1, 2]; order=4).f)
end
# lattice extras
K1 = poisson_cgf(5.0)
add("lat", "pois7_origin", tailprob(K1, 7; order=3, lattice=true, sinh=:origin).F)
add("lat", "pois7_saddle", tailprob(K1, 7; order=3, lattice=true).F)
add("lat", "pois1", tailprob(K1, 1; order=4, lattice=true).F)
for k in (6, 18, 25)
    add("lat", "binom_F$k", tailprob(binomial_cgf(40, 0.3), k; order=4, lattice=true).F)
    add("lat", "binom_f$k", density(binomial_cgf(40, 0.3), k; order=4, lattice=true).f)
end
for k in (2, 7, 20)
    add("lat", "negbin_F$k", tailprob(negative_binomial_cgf(5.0, 0.4), k; order=4, lattice=true).F)
    add("lat", "negbin_f$k", density(negative_binomial_cgf(5.0, 0.4), k; order=4, lattice=true).f)
end
Kbp = CGF(t -> 2*(exp(t[1]) - 1) + 2*(exp(t[2]) - 1) + (exp(t[1] + t[2]) - 1), 2)
add("lat", "bipois_Flow", tailprob(Kbp, [2, 1]; order=3, lattice=true, lower=true).F)
add("lat", "bipois_Fmix", tailprob(Kbp, [6, 1]; order=3, lattice=true, signs=[1, -1]).F)
# quantiles
Kga = gamma_cgf(3.0)
for p in (1e-6, 0.05, 0.5, 0.95), lower in (true, false)
    q = quantile(Kga, p; order=4, lower)
    add("quant", "gamma_$(p)_$(lower)", [q.q, q.tau])
end
add("quant", "normal", [quantile(normal_cgf(1.0, 4.0), 0.025; order=2).q])
KF = CGF(t -> ncx2_cgf(4, 3)([t[1]]) + ncx2_cgf(12)([t[2]]), 2; domain=t -> t[1] < 0.5 && t[2] < 0.5)
for p in (0.01, 0.5)
    q = ratio_quantile(KF, p; order=4); add("quant", "ncf_$(p)", [q.r, q.shat])
end
let q = ratio_quantile(KF, 1e-4; order=4, lower=false); add("quant", "ncf_upper", [q.r, q.shat]); end
# BigFloat (256 bits): the gamma sum, d = 3, N = 4, q = 0.99; the Dirichlet; the logistic regression
setprecision(BigFloat, 256) do
    K = CGF(t -> -4log(1 - t[1] - t[2]) - 8log(1 - t[2] - t[3]) - 2log(1 - t[2]), 3;
            domain=t -> t[1] + t[2] < 1 && t[2] + t[3] < 1 && t[2] < 1)
    w = SaddlepointExpansions.beta_inc_inv(4.0, 2.0, 0.99)[1]; x = [6w, 14.0, 8.0]
    add("big", "beta_x", x)
    add("big", "beta_F", conditional_tailprob(K, big.(x), 1; order=5).F)
    add("big", "beta_f", conditional_density(K, big.(x), 1; order=5).f)
    Kd = CGF(t -> -5log(1 - t[1] - t[3]) - 7.5log(1 - t[2] - t[3]) - 10log(1 - t[3]), 3)
    add("big", "dir_F", conditional_tailprob(Kd, big.([0.3*22.5, 0.45*22.5, 22.5]), [1, 2]; order=3).F)
    Kl = logistic_regression_cgf([ones(Int, 30) mod.(0:29, 5) mod.(0:29, 3)])
    add("big", "logit_F", conditional_tailprob(Kl, big.([15.0, 37, 15]), 2; order=3, lattice=true).F)
    add("big", "pois_F", tailprob(poisson_cgf(5.0), big(11); order=4, lattice=true).F)
end
open(joinpath(@__DIR__, "conditional_reference.csv"), "w") do io
    println(io, "case,quantity,order,value")
    foreach(r -> println(io, r), rows)
end
println("wrote ", length(rows), " rows")
