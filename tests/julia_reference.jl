# Reference values from SaddlepointExpansions.jl for the MATLAB tests (test_spx.m).
# Run from this folder:  julia julia_reference.jl  (it activates the Julia package next to the
# MATLAB package, C:\work\SPA\SaddlepointExpansions.jl, and writes julia_reference.csv).
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "SaddlepointExpansions.jl"))
Pkg.instantiate()
using SaddlepointExpansions, LinearAlgebra, Printf

toep(r, n) = [r^abs(i - j) for i in 1:n, j in 1:n]
rows = String[]
add(case, q, v) = for (M, x) in enumerate(v); push!(rows, @sprintf("%s,%s,%d,%.17g", case, q, M, x)); end

# 1. exponential sums, the README example
K = CGF(t -> -log(1 - t[1]) - log(1 - t[2]) - log(1 - t[1] - t[2]), 2;
        domain = t -> t[1] < 1 && t[2] < 1 && t[1] + t[2] < 1)
Kn = sample_mean(K, 5)
add("expsum2", "F", tailprob(Kn, [3.0, 3.0]; order=4).F)
add("expsum2", "f", density(Kn, [3.0, 3.0]; order=4).f)
add("expsum2", "Flow", tailprob(Kn, [1.5, 1.8]; order=4, lower=true).F)
add("expsum2", "Fmix", tailprob(Kn, [2.5, 1.5]; order=4, signs=[1, -1]).F)
# 2. gamma and noncentral chi-squared
add("gamma", "F", tailprob(gamma_cgf(5.0), 8.0; order=5).F)
add("gamma", "Flow", tailprob(gamma_cgf(5.0), 3.0; order=5, lower=true).F)
add("gamma", "f", density(gamma_cgf(5.0), 8.0; order=5).f)
add("ncx2", "F", tailprob(ncx2_cgf(10.0, 5.0), 25.0; order=5).F)
add("ncx2", "f", density(ncx2_cgf(10.0, 5.0), 25.0; order=5).f)
# 3. uniform mean of 4
Ku = sample_mean(uniform_cgf(), 4)
add("uniform", "F", tailprob(Ku, 0.4; order=4).F)
add("uniform", "f", density(Ku, 0.4; order=4).f)
# 4. common component, bivariate (Table 1 point) and trivariate
Kc = common_component_cgf([15.0, 20.0, 10.0], [5.0, 10.0, 8.0])
add("cc2", "F", tailprob(Kc, [48.09950493836207795336339, 65.23368793961408597955183]; order=5).F)
add("cc2", "f", density(Kc, [48.09950493836207795336339, 65.23368793961408597955183]; order=5).f)
Kc3 = common_component_cgf([4.0, 5.0, 6.0, 7.0])
add("cc3", "F", tailprob(Kc3, [14.0, 16.0, 17.0]; order=3).F)
add("cc3", "f", density(Kc3, [14.0, 16.0, 17.0]; order=3).f)
# 5. quadratic forms
n = 8
Kq = quadratic_forms_cgf([toep(0.5, n), Matrix(1.0I, n, n)]; Sigma=toep(0.3, n), mu=fill(0.5, n))
add("qf2", "F", tailprob(Kq, [14.0, 13.0]; order=4).F)
add("qf2", "f", density(Kq, [14.0, 13.0]; order=4).f)
D = derivatives(Kq, [0.05, -0.02], 5)
for nu in [(0, 0), (1, 0), (0, 1), (2, 1), (3, 2), (1, 4)]
    push!(rows, @sprintf("qf2,deriv_%d_%d,1,%.17g", nu[1], nu[2], D[nu]))
end
# 6. ratios
add("ratio_qf", "f", ratio_density(Kq, 1.2; order=4).f)
add("ratio_qf", "F", ratio_tailprob(Kq, 1.2; order=4).F)
add("ratio_qf", "Flow", ratio_tailprob(Kq, 0.9; order=4, lower=true).F)
# 7. Wishart diagonal and normal
Kw = wishart_diag_cgf([1.0 0.4; 0.4 2.0], 10)
add("wishart", "F", tailprob(Kw, [12.0, 25.0]; order=4).F)
Kz = normal_cgf([0.0, 1.0], [1.0 0.5; 0.5 2.0])
add("normal", "F", tailprob(Kz, [0.5, 1.5]; order=2).F)
# 8. exponential sums in three dimensions, saddlepoint with a negative component
A = [1.0 0 0; 0 1 0; 0 0 1; 1 1 0; 0 1 1]
Ke = exponential_sums_cgf(A; n=6)
add("expsum3", "F", tailprob(Ke, [2.2, 2.6, 1.9]; order=3).F)
add("expsum3", "f", density(Ke, [2.2, 2.6, 1.9]; order=3).f)

open(joinpath(@__DIR__, "julia_reference.csv"), "w") do io
    println(io, "case,quantity,order,value")
    foreach(r -> println(io, r), rows)
end
println("wrote ", length(rows), " rows")
