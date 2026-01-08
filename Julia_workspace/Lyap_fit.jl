using CSV, Tables, DataFrames
using CairoMakie, LsqFit

path = "/Users/jinyuanshang/Nutstore Files/file_numerical/Lyap_spec/Julia_workspace/data/"

N_trj = 10
L_arr = 6:2:20
fname = vcat("eta.csv",["Lyap_L$i.csv" for i in L_arr]...)
eta = vec(Matrix(CSV.read(path*fname[1],DataFrame,header=false)))
Lyap_mat = zeros(N_trj,length(eta),length(L_arr))
for i in eachindex(L_arr)
    Lyap_mat[:,:,i] = reshape(vec(Matrix(CSV.read(path*fname[i+1],DataFrame,header=false))),(N_trj,length(eta)))
end

### visualize
gap = zeros(length(eta),length(L_arr))
f1 = Figure()
ax1 = Axis(f1[1,1],yscale=log10)
for i in eachindex(L_arr)
    gap[:,i] = diff(mapslices(x -> partialsort(x, 1:2), Lyap_mat[:,:,i], dims=1), dims=1)[:]
    CairoMakie.scatter!(ax1,eta,abs.(gap[:,i]))
end
f1

### fit
@. model(L, p) = p[1] + p[2] * (p[3])^(-L)

fit_delta = zeros(length(eta))
fit_a = zeros(length(eta))
fit_b = zeros(length(eta))

p0 = [0.5, 1.0, 1.1] 

for i in 1:length(eta)
    fit = curve_fit(model, L_arr, gap[i, :], p0, lower=[-Inf, -Inf, 1e-5])
    fit_delta[i] = fit.param[1]
    fit_a[i]     = fit.param[2]
    fit_b[i]     = fit.param[3]

    p0 = fit.param
end

f2 = Figure()
ax2 = Axis(f2[1,1])
CairoMakie.scatter!(ax2,eta[1:18],fit_delta[1:18])
f2