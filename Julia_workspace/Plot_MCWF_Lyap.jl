using CSV, DataFrames, Tables, LaTeXStrings
using CairoMakie, LsqFit
using Statistics, StatsBase

L = 25
path = "/Users/jinyuanshang/Nutstore Files/file_numerical/Lyap_spec/Julia_workspace/data/trajectory/L$L/"
N_vol = 10
N_trj = 100
F_arr = 0.0:0.1:5.0 
Lyap_mat = zeros(N_vol,N_trj,length(F_arr))
for i in eachindex(F_arr)
    fname = "lim_anw_F$(i-1).csv"
    Lyap_mat[:,:,i] = reshape(Matrix(CSV.read(path*fname,DataFrame,header=false)),N_vol,N_trj)
end

### test of sort
tt = zeros(N_trj, length(F_arr))
for n in 1:N_trj
    for m in 1:length(F_arr)
        tt[n,m] = issorted(Lyap_mat[:,n,m],rev=true) ? 1 : 0
    end
end
tt
tt1 = findall(x->x==0,tt)
issorted(Lyap_mat[:,61,10],rev=true)
Lyap_mat[:,1,19]

### distribution with trajectory
f1 = Figure(size=(1000,450))
ax1 = Axis(f1[1,1],
    xlabel=L"F", ylabel=latexstring("\\lambda_{\\text{max}}"),
    xlabelsize=30, ylabelsize = 30
)
for idx = 1:50
    CairoMakie.scatter!(ax1, F_arr, Lyap_mat[1,idx,:],markersize=8)
end
CairoMakie.hlines!(ax1, [0.0], color=:red, linestyle=(:dash))
ax2 = Axis(f1[1,2],
    xlabel = L"F", ylabel = latexstring("\\Delta\\lambda"),
    xlabelsize = 30, ylabelsize = 30
)
for idx = 1:50
    CairoMakie.scatter!(ax2, F_arr, Lyap_mat[1,idx,:] .- Lyap_mat[2,idx,:],markersize=8)
end
CairoMakie.hlines!(ax2, [0.5], color=:red, linestyle=(:dash))
f1
path = "/Users/jinyuanshang/Nutstore Files/file_numerical/Lyap_spec/Julia_workspace/pic/"
save(path*"Lyap_dpt_trj.pdf", f1)


### trajectory mean
f2 = Figure(size=(1000,450))
ax1 = Axis(f2[1,1],
    xlabel=L"F", ylabel=latexstring("\\lambda_{\\text{max}}"),
    xlabelsize=30, ylabelsize = 30
)
CairoMakie.scatter!(ax1, F_arr, vec(mean(Lyap_mat[1,:,:],dims=1)))
CairoMakie.hlines!(ax1, [0.0], color=:red, linestyle=(:dash))
ax11 = Axis(f2[1,1],
    width=Relative(0.35),height=Relative(0.35),halign=0.1,valign=0.9,
)
CairoMakie.scatter!(ax11, F_arr, vec(var(Lyap_mat[1,:,:],dims=1)))
ax2 = Axis(f2[1,2],
    xlabel = L"F", ylabel = latexstring("\\Delta\\lambda"),
    xlabelsize = 30, ylabelsize = 30
)
CairoMakie.scatter!(ax2, F_arr, vec(mean(Lyap_mat[1,:,:] .- Lyap_mat[2,:,:],dims=1)))
CairoMakie.hlines!(ax2, [0.5], color=:red, linestyle=(:dash))
ax21 = Axis(f2[1,2],
    width=Relative(0.35),height=Relative(0.35),halign=0.2,valign=0.1,
)
CairoMakie.scatter!(ax21, F_arr, vec(var(Lyap_mat[1,:,:] .- Lyap_mat[2,:,:],dims=1)))
f2

path = "/Users/jinyuanshang/Nutstore Files/file_numerical/Lyap_spec/Julia_workspace/pic/"
save(path*"Lyap_dpt_mean.pdf", f2)

f4 = Figure()
ax1 = Axis(f4[1,1])
CairoMakie.scatter!(ax1, F_arr, m_Lyap[1,:] .- m_Lyap[2,:])
f4

f5 = Figure()
ax1 = Axis(f5[1,1])
CairoMakie.hist!(ax1, Lyap_mat[1,:,30], bins=20)
f5

path = "/Users/jinyuanshang/Nutstore Files/file_numerical/Lyap_spec/Julia_workspace/data/trajectory/"
fname = path*"time_F12_T18.csv"
tt4 = vec(Matrix(CSV.read(fname,DataFrame,header=false)))
fname = path*"anw_F12_T18.csv"
tt5 = reshape(vec(Matrix(CSV.read(fname,DataFrame,header=false))),(10,length(tt4)))

tt6 = cumsum(tt5', dims=1) 
path = "/Users/jinyuanshang/Nutstore Files/file_numerical/Lyap_spec/Julia_workspace/"
CSV.write(path * "cum_anw_F12_T18.csv", Tables.table(tt6[:,1]), writeheader=false)
CSV.write(path * "time_F12_T18.csv", Tables.table(tt4), writeheader=false)
[estimate_limit(tt4, tt6[:,i]) for i in axes(tt6,2)]
w2 = boot_est(tt6[:,2])
b = estimate_limit(tt4,tt6[:,4])
b = scan_fit(tt4[50:end],tt6[50:end,4])

starts, limits, rmses = estimate_limit(tt4, tt6[:,1])
y = tt6[1:end,1]
f6 = Figure()
ax1 = Axis(f6[1,1])
# CairoMakie.scatter!(ax1, y[1:end-1], y[2:end])
# CairoMakie.scatter!(ax1, y[1:end-1], lin_model(y[1:end-1], fit_lin.param), color=:red)
CairoMakie.scatter!(ax1, tt4, tt6[:,1])
CairoMakie.lines!(ax1,tt4, b[1]*tt4 .+b[2])
# CairoMakie.lines!(ax1, tt4, b[1] .+ b[2] * exp.(-b[3]*tt4),color=:red)
# CairoMakie.scatter!(ax1, x, model(x,fit.param), color=:red)
# CairoMakie.scatter!(ax1, starts, limits)
f6

path = "/Users/jinyuanshang/Nutstore Files/file_numerical/Lyap_spec/Julia_workspace/data/trajectory/"
fname = path*"time_F33_T18.csv"
tt1 = vec(Matrix(CSV.read(fname,DataFrame,header=false)))
fname = path*"anw_F33_T18.csv"
tt2 = reshape(vec(Matrix(CSV.read(fname,DataFrame,header=false))),(10,length(tt1)))

tt3 = cumsum(tt2', dims=1)
path = "/Users/jinyuanshang/Nutstore Files/file_numerical/Lyap_spec/Julia_workspace/"
CSV.write(path * "cum_anw_F33_T18.csv", Tables.table(tt3[:,1]), writeheader=false)
CSV.write(path * "time_F33_T18.csv", Tables.table(tt1), writeheader=false)
[estimate_limit(tt1, tt3[:,i]) for i in axes(tt3,2)]
w1 = boot_est(tt3[:,1];k=200)
[boot_est(tt3[:,i];k=50).limit_estimate for i in axes(tt3,2)]
x = tt1[div(end,2):end]
y = tt3[div(end,2):end,1]
b = [scan_fit(x,tt3[div(end,2):end,i]) for i in axes(tt3,2)]

starts, limits, rmses = estimate_limit(tt1, tt3[:,1])
y = tt3[1:end,1]
k =100
f6 = Figure()
ax1 = Axis(f6[1,1])
CairoMakie.scatter!(ax1, tt1, tt3[:,1])
# CairoMakie.scatter!(ax1, x, b[1] .+ b[2] * exp.(-b[3] * x))
# CairoMakie.scatter!(ax1, y[1:k:end-k], y[1+k:k:end])
# CairoMakie.hlines!(ax1, w1.limit_estimate)
# CairoMakie.scatter!(ax1, y[1:end-1], lin_model(y[1:end-1], fit_lin.param), color=:red)
# CairoMakie.scatter!(ax1, x, model(x,fit.param), color=:red)
# CairoMakie.scatter!(ax1, starts, limits)
f6
@. lin_model(x, p) = p[1] * x + p[2]
p0 = [-1.0, 0.0]
fit_lin = curve_fit(lin_model, tt4,tt6[:,1], p0)
b = fit_lin.param
stderror(fit_lin)
margin_error(fit_lin,0.05)
confint(fit_lin; level=0.95)
b[2] / (1 - b[1])

x = tt1[3000:end]
y = tt3[3000:end,1]
@. model(t, p) = p[1] + p[2] * exp(-p[3] * t)
p0 = [y[end], y[1] - y[end], 0.05]
fit = curve_fit(model, x, y, p0)
fit.param[1]