using LinearAlgebra
using QuantumToolbox
using DifferentialEquations
using Statistics, LsqFit
using CSV, DataFrames, Tables

println("Running with $(Threads.nthreads()) threads on $(gethostname())")
flush(stdout)

function qr_pos_log!(StateMat)
    F = qr(StateMat)
    R_diag = @view F.R[diagind(F.R)]
    d = sign.(R_diag)
    replace!(d, 0 => 1)

    # Q absorbs d, R gets multiplied by conj(d) to cancel the phase
    Q_new = Matrix(F.Q) .* d'
    # R_new = conj.(d) .* F.R   #: d -> conj.(d)

    return Q_new, log.(abs.(R_diag))
end

### evolution with matrices sequence
function safe_time(jumps_t, jumps_idx, tend; max_dt=10.0)
    full_times = Float64[]
    full_indices = Int[]

    t_curr = 0.0
    N = length(jumps_t)

    for k in 1:(N+1)
        if k <= N
            t_target = jumps_t[k]
            op_code = jumps_idx[k]
        else
            t_target = tend
            op_code = 0 # 0 means Re-norm only (at tend)
        end
        while (t_target - t_curr) > (max_dt + 1e-9)
            t_curr += max_dt
            push!(full_times, t_curr)
            push!(full_indices, 0) # 0 = Renormalization only
        end
        if t_target > t_curr + 1e-12
            push!(full_times, t_target)
            push!(full_indices, op_code)
            t_curr = t_target
        end
    end

    return full_times, full_indices
end

function Lyap(p_ode, Lm, N_vol, sol, tend, dt_reg, traj_idx)
    dim = size(p_ode, 1)

    jumps_t = copy(sol.col_times[traj_idx])   # Times when jumps occurred
    jumps_idx = copy(sol.col_which[traj_idx])
    cb_times, cb_indices = safe_time(jumps_t, jumps_idx, tend; max_dt=dt_reg)

    function Heff_evo!(du, u, p, t)
        mul!(du, p, u) # du = p * u  (equivalent to du = -iH * u)
    end

    # anw = Vector{Vector{Float64}}()
    # sizehint!(anw, length(cb_times))
    results = Matrix{Float64}(undef, N_vol, length(cb_times))
    u_cache = Matrix{ComplexF64}(undef, dim, N_vol)
    jump_counter = Ref(1)
    function jump_qr_affect!(integrator)
        t = integrator.t
        # k = searchsortedfirst(cb_times, t)
        k = jump_counter[]
        if k <= length(cb_times)
            #  && isapprox(cb_times[k], t, atol=1e-12)
            op_idx = cb_indices[k]
            if op_idx > 0
                # integrator.u .= Lm[op_idx].data * integrator.u
                mul!(u_cache, Lm[op_idx].data, integrator.u)
                integrator.u .= u_cache
            end
        end
        Q, R_log = qr_pos_log!(integrator.u)
        # push!(anw, R_log)
        results[:, k] .= R_log
        integrator.u .= Matrix(Q)
        jump_counter[] += 1
    end

    Q0 = Matrix(qr(randn(ComplexF64, dim, N_vol)).Q)
    jump_counter[] = 1
    cb = PresetTimeCallback(cb_times, jump_qr_affect!)
    prob = ODEProblem(Heff_evo!, Q0, (0.0, cb_times[end]), p_ode)
    sim = solve(prob, Tsit5(), callback=cb, save_everystep=false)

    return results, cb_times
end

function estimate_limit(tarr, data::Vector{<:Number}; fraction=2)
    start_idx = div(length(tarr), fraction)
    x = tarr[start_idx:end]
    y = data[start_idx:end]
    @. model(t, p) = p[1] * t + p[2]
    p0 = [(data[end] - data[1]) / (tarr[end] - tarr[1]), data[1]]
    fit = curve_fit(model, x, y, p0)
    return fit.param[1]
end

### MCWF simulation
tlist = LinRange(0.0, 1000.0, 8000)
Na = 30
v0 = fock(Na, Na - 1)
a = destroy(Na)

Gamma = 1.0
U0, F0 = 10.0 * Gamma, 1.0 * Gamma
N = 10
U, F = U0 / N, F0 * sqrt(N)
Delt = 10 * Gamma

H = -Delt * a' * a + (U / 2) * a' * a' * a * a + F * (a' + a)
c_ops = [sqrt(Gamma) * a]
e_ops = [a' * a]

sol = mcsolve(H, v0, tlist, c_ops, e_ops=e_ops; ntraj=1000, ensemblealg=EnsembleThreads())
flush(stdout)

### Benettin method
N_vol = 10
Heff = -Delt * a' * a + (U / 2) * a' * a' * a * a + F * (a' + a) - 0.5im * sum(op' * op for op in c_ops)
Heff_dense = Matrix(Heff.data)
p_ode = -1im * Heff_dense

path = ""
dt_reg = 10.0
N_trj = 10
Fid = 0
results = Vector{Matrix{Float64}}(undef, N_trj)
times = Vector{Vector{Float64}}(undef, N_trj)
lim_anw = zeros(N_vol, N_trj)
BLAS.set_num_threads(1)
@time Threads.@threads for traj_idx in 1:N_trj
    anw, t_arr = Lyap(p_ode, c_ops, N_vol, sol, tlist[end], dt_reg, traj_idx)
    results[traj_idx] = anw
    times[traj_idx] = t_arr
    temp = cumsum(anw', dims=1)
    lim_anw[:, traj_idx] = [estimate_limit(t_arr, temp[:, i]) for i in axes(temp, 2)]
end

for i in 1:N_trj
    CSV.write(path * "anw_F$(Fid)_T$(i).csv", Tables.table(vec(results[i])), writeheader=false)
    CSV.write(path * "time_F$(Fid)_T$(i).csv", Tables.table(times[i]), writeheader=false)
end

CSV.write(path * "lim_anw_F$(Fid).csv", Tables.table(vec(lim_anw)), writeheader=false)


println("done.")

println("doneeee.")


traj_idx = rand(1:1000)
anw, t_arr = Lyap(p_ode, c_ops, N_vol, sol, tlist[end], dt_reg, traj_idx)
temp = cumsum(anw', dims=1)
[estimate_limit(t_arr, temp[:, i]) for i in axes(temp, 2)]