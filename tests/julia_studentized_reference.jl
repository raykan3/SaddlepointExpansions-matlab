# High-precision reference values of the Studentized-mean expansions from SaddlepointExpansions.jl,
# for the double-double tests of the MATLAB package (test_spx.m): the expansions evaluated in
# BigFloat with 256 bits, at Float64 arguments, for the noncentral t and the Dickey-Fuller t-ratio.
# Run from this folder:  julia julia_studentized_reference.jl   (writes studentized_reference.csv)
using Pkg
pkg = joinpath(@__DIR__, "..", "..", "SaddlepointExpansions.jl")
Pkg.activate(pkg)
Pkg.instantiate()
using SaddlepointExpansions, Printf
include(joinpath(pkg, "test", "dickey_fuller_cgf.jl"))
rows = String[]
put(case, qty, j, x) = push!(rows, @sprintf("%s,%s,%d,%s", case, qty, j, string(Float64(x)) * "," * @sprintf("%.25e", x)))
setprecision(BigFloat, 256) do
    mu, nu, n = 1.0, 3.0, 16
    K1 = CGF(t -> mu*t[1] + t[1]^2/2 - nu/2*log(1 - 2t[2]/nu), 2; domain = t -> t[2] < nu/2)
    Kn = sample_mean(K1, n)
    for q in (0.5, 1.5, 2.5)
        d = studentized_density(Kn, big(q); order=6)
        for j in 1:6; put(@sprintf("nct_%.1f", q), "f", j, d.f[j]); end
        r = studentized_tailprob(Kn, big(q); order=4, lower=true)
        for j in 1:8; put(@sprintf("nct_%.1f", q), "Fp", j-1, r.info.Fp[j]); end
    end
    for (m, q) in ((1, -1.941), (2, -2.862), (3, -3.411))
        K = dickey_fuller_cgf(m)
        d = studentized_density(K, big(q); order=6)
        for j in 1:6; put("df$(m)", "f", j, d.f[j]); end
        r = studentized_tailprob(K, big(q); order=4, lower=true)
        for j in 1:8; put("df$(m)", "Fp", j-1, r.info.Fp[j]); end
    end
    K = dickey_fuller_cgf(3; c=-5.0)
    r = studentized_tailprob(K, big(-2.025180818400757); order=3, lower=true)
    for j in 1:6; put("df3_c-5", "Fp", j-1, r.info.Fp[j]); end
end
open(joinpath(@__DIR__, "studentized_reference.csv"), "w") do io
    println(io, "case,quantity,index,value,value25")
    foreach(r -> println(io, r), rows)
end
println("wrote ", length(rows), " rows")
