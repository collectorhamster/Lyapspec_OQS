using LinearAlgebra
using CairoMakie
using QuantumToolbox
using DifferentialEquations
using LsqFit, Statistics
using CSV, DataFrames, Tables

function qr_pos!(StateMat)
    F = qr(StateMat)
    
    R_diag = @view F.R[diagind(F.R)]
    d = sign.(R_diag)
    replace!(d, 0 => 1)

    # Q absorbs d, R gets multiplied by conj(d) to cancel the phase
    Q_new = Matrix(F.Q) .* d' 
    R_new = conj.(d) .* F.R   # CHANGED: d -> conj.(d)
    
    return Q_new, R_new
end

# function jump(state0, Lm, gamma, dt)
#     state = copy(state0)
    
#     Mid = 0
#     p = zeros(length(Lm)-1)
#     Heff = copy(Lm[1])

#     for (k, i) in enumerate(2:length(Lm))
#         jp = sqrt(gamma) * Lm[i]
#         p[k] = max(real(dot(state, jp' * jp * state)), 0.0) * dt
#         Heff -= (im * jp' * jp) / 2
#     end

#     r1 = rand()
#     totalp = sum(p)
#     if r1 > totalp
#         state .= state .- im * dt * (Heff * state)
#         state ./= norm(state)
#         Mid = 1
#     else
#         r2 = rand()
#         if totalp > 0
#             p ./= totalp
#         end
#         cumsum_p = cumsum(p)
#         idx = searchsortedfirst(cumsum_p, r2)
#         jp = sqrt(gamma) * Lm[idx + 1]
#         state .= jp * state
#         state ./= norm(state)
#         Mid = idx + 1
#     end
#     return state, Mid
# end

# function evolution(state_mat, ob, Lm, gamma, dt, Nt)
#     N_trj = size(state_mat, 2)
#     obexp = zeros(Nt, N_trj)
#     Mseq = zeros(Nt, N_trj)
#     for ti in 1:Nt
#         for tj in 1:N_trj
#             state = view(state_mat, :, tj)
#             obexp[ti, tj] = real.(state' * ob * state)
#             state_mat[:, tj], Mseq[ti,tj] = jump(state, Lm, gamma, dt)
#         end
#     end
#     return obexp, Mseq
# end

# ### benchmark
# sigx = [0.0 1.0; 1.0 0.0]
# sigy = [0.0 -im; im 0.0]
# sigz = [1.0 0.0; 0.0 -1.0]
# sig0 = [1.0 0.0; 0.0 1.0]
# sigp = (sigx + im * sigy) / 2
# sigm = (sigx - im * sigy) / 2

# Omeg, Delt, Gamma = 1.0, 0.0, 1.0/6
# H = -(Omeg/2) * sigx - Delt * sigp * sigm
# Lm = [H, sigm]

# # v0 = rand(ComplexF64, 2)
# v0 = complex([1.0, 0.0])
# N_trj = 1000
# v0_mat = repeat(v0, 1, N_trj)
# ob = (sig0 - sigz) / 2
# dt, Nt = 0.005, 8000

# anw, Mseq = evolution(v0_mat, ob, Lm, Gamma, dt, Nt)

# t_arr = 0:dt:(Nt-1)*dt
# f1 = Figure()
# ax1 = Axis(f1[1,1], limits=(nothing, nothing, 0, 1))
# CairoMakie.plot!(ax1, 0:dt:(Nt-1)*dt, vec(mean(anw, dims=2)))
# f1


# ### generate sequence, QuantumToolbox provides a better solution
# Na = 30
# a = destroy(Na).data

# Gamma = 1.0
# U0, F0 = 10.0*Gamma, 1*Gamma
# N = 10
# U, F = U0/N, F0*sqrt(N)
# Delt = 10*Gamma

# H = -Delt * a' * a + (U/2) * a' * a' * a * a + F * (a' + a)
# Lm = [H, a]
# ob = a' * a

# # v0 = rand(ComplexF64, 2)
# v0 = fock(Na, Na-1)
# N_trj = 1000
# v0_mat = repeat(v0.data, 1, N_trj)
# dt, Nt = 0.002,  20000

# @time obexp, Mseq = evolution(v0_mat, ob, Lm, Gamma, dt, Nt)

# f1 = Figure()
# ax1 = Axis(f1[1,1], limits=(nothing, nothing, 0, Na/N))
# CairoMakie.plot!(ax1, 0:dt:(Nt-1)*dt, vec(mean(obexp, dims=2))/N)
# f1

### QuantumToolbox as benchmark
tlist = LinRange(0.0, 1000.0, 8000)
Na = 30
v0 = fock(Na, 0)
a  = destroy(Na)

Gamma = 1.0
U0, F0 = 10.0*Gamma, 4*Gamma
N = 10
U, F = U0/N, F0*sqrt(N)
Delt = 10*Gamma

H = -Delt * a' * a + (U/2) * a' * a' * a * a + F * (a' + a)
c_ops = [sqrt(Gamma) * a]
e_ops = [a' * a]

sol = mcsolve(H, v0, tlist, c_ops, e_ops=e_ops; ntraj=1000)

# plot by CairoMakie.jl
fig = Figure(size = (500, 350))
ax = Axis(fig[1, 1],
    xlabel = "Time",
    ylabel = "Expectation values",
    title = "Monte Carlo time evolution (500 trajectories)",
)
lines!(ax, tlist, real(sol.expect[1,:])/N, label = "cavity photon number", linestyle = :solid)
axislegend(ax, position = :rt)
fig


### evolution with matrices sequence
function Lyap(dim, Heff, Lm, N_trj, sol, tlist, traj_idx)
    jumps_t = copy(sol.col_times[traj_idx])   # Times when jumps occurred
    jumps_idx = copy(sol.col_which[traj_idx])
    push!(jumps_t, tlist[end])  # Ensure we go to the end time
    push!(jumps_idx, 0) 

    # a  = destroy(dim)
    # Delt, U0, F0 = pH[:] * Gamma
    # N = 10
    # U, F = U0/N, F0*sqrt(N)

    # Heff = -Delt * a' * a + (U/2) * a' * a' * a * a + F * (a' + a) - 0.5im * sum(op' * op for op in Lm)
    Heff_dense = Matrix(Heff.data)

    function Heff_evo!(du, u, p, t)
        mul!(du, p, u) # du = p * u  (equivalent to du = -iH * u)
    end

    anw = Vector{Float64}[]
    jump_counter = Ref(1)
    function jump_qr_affect!(integrator)
        k = jump_counter[]
        if k > length(jumps_idx)
            return "error: jump index out of bounds"
        end
        op_index = jumps_idx[k]
        
        if op_index > 0
            integrator.u .= Lm[op_index].data * integrator.u
        end

        Q, R = qr_pos!(integrator.u)
        push!(anw, log.((diag(R))))
        integrator.u .= Matrix(Q)

        jump_counter[] += 1
    end

    Q0 = Matrix(qr(randn(ComplexF64, dim, N_trj)).Q)
    jump_counter[] = 1
    # Setup Callback
    cb = PresetTimeCallback(jumps_t, jump_qr_affect!)
    p_ode = -1im * Heff_dense
    prob = ODEProblem(Heff_evo!, Q0, (0.0, jumps_t[end]), p_ode)
    sim = solve(prob, Tsit5(), callback=cb, save_everystep=false)
    
    new_anw = hcat(anw...)'
    # gapt = jumps_t .- [0; jumps_t[1:end-1]]
    # return new_anw ./gapt, jumps_t
    return cumsum(new_anw, dims=1) ./jumps_t, jumps_t
    # return hcat(anw...)', jumps_t
end


N_trj = 10
Heff = -Delt * a' * a + (U/2) * a' * a' * a * a + F * (a' + a) - 0.5im * sum(op' * op for op in c_ops)
traj_idx = 5
dim = Na
anw, t_arr = Lyap(dim, Heff, c_ops, N_trj, sol, tlist, traj_idx)

f2 = Figure()
ax2 = Axis(f2[1,1])
# CairoMakie.scatter!(ax2, t_arr, cumsum(anw[:, 1]) ./ (1:size(anw,1)))
CairoMakie.scatter!(ax2, t_arr, anw[:, 1])
# CairoMakie.lines!(ax2, t_arr, [tt1], color=:red)
# CairoMakie.scatter!(ax2, tt4)
f2

### convergence estimate
# sequence dynamics
# function conv(data)
#     s1 = data[2:end] 
#     s2 = data[1:end-1]
    
#     @. model(x, p) = p[1] * x + p[2]
#     p0 = [1.0, 0.0]
#     fit = curve_fit(model, s2, s1, p0)
#     params = fit.param
#     return params[2] / (1 - params[1])
# end

function estimate_limit(data::Vector{<:Number}; window::Int=10)
    # 1. Compact Tail Detection
    # Calculate rolling std deviation efficiently using views
    n = length(data)
    stds = [std(@view data[i:i+window-1]) for i in 1:(n-window+1)]
    
    # Define baseline noise from the last 20% of data
    baseline = median(@view stds[end-div(n,5):end])
    
    # Find start of tail: first time std drops below 2x baseline
    # 'something' handles the case where findfirst returns nothing (defaults to n/2)
    start_idx = something(findfirst(<(2 * baseline), stds), div(n, 2))
    
    # 2. AR(1) Fixed-Point Estimation via Linear Algebra
    # Slice the stable tail
    tail = @view data[start_idx:end]
    
    # Solve system: x[t+1] = a * x[t] + b
    # A \ y is the idiomatic way to solve Least Squares in Julia
    a, b = [tail[1:end-1] ones(length(tail)-1)] \ tail[2:end]
    
    # Return Limit L = b / (1 - a)
    return b / (1 - a)
end

lim_anw = [conv_sd(anw[:, i]) for i in axes(anw,2)]
CairoMakie.lines!(ax2, t_arr, [lim_anw], color=:red)
f2


path = "/Users/jinyuanshang/Nutstore Files/file_numerical/Lyap_spec/Julia_workspace/data/trajectory/"
fname = path*"test.csv"
CSV.write(fname,  Tables.table(anw[:,1]), writeheader=false,append=false)