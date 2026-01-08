using LinearAlgebra
using Yao
using SparseArrays
using Distributions, Statistics, StatsBase
using CairoMakie

const T = ComplexF64

###---positive qr---###
function qr_pos!(StateMat)
    # Decompose
    F = qr(StateMat)
    
    # Extract diagonal signs from R without allocating new full vector
    R_diag = @view F.R[diagind(F.R)]
    d = sign.(R_diag)
    replace!(d, 0 => 1)

    # Optimization: Broadcast scaling
    # Q_new = Q * D (Scale columns)
    # R_new = D * R (Scale rows)
    Q_new = Matrix(F.Q) .* d' 
    R_new = d .* F.R          
    
    return Q_new, R_new
end

###---function for generating Haar-random unitary matrix---###
function haar_unitary(n::Int)
    Z = (randn(T, n, n) + im * randn(T, n, n)) / sqrt(2.0)
    F = qr(Z)
    R_diag = @view F.R[diagind(F.R)]
    signs = sign.(real(R_diag))
    replace!(signs, 0 => 1.0)
    
    # Optimized reconstruction
    return Matrix(F.Q) .* signs'
end
# test
# U = haar_unitary(4)
# norm(U' * U - I)

###---local measurement operator---###
function Mpm(pm, eta)
    val1 = (1 + pm * eta) / sqrt(1 * (1 + eta^2)) 
    val2 = (1 - pm * eta) / sqrt(1 * (1 + eta^2)) 
    # Yao handles Diagonal matrices as sparse operations automatically
    return Diagonal(T[val1, val2]) 
end

###---measurement outcomes---###
function measure_out!(pm, reg, eta, L)
    # Optimization: Pre-calculate the probability operator O = M'M
    # Prob(+) = <psi| M'+ M+ |psi>
    val_plus = (1 + eta)^2 / (2 * (1 + eta^2))
    val_minus = (1 - eta)^2 / (2 * (1 + eta^2))
    prob_op_diagp = Diagonal(T[val_plus, val_minus])
    prob_op_diagm = Diagonal(T[val_minus, val_plus])
    prob_opp = matblock(prob_op_diagp)
    prob_opm = matblock(prob_op_diagm)

    # Create a view of the first trajectory (avoid copying the whole batch)
    # We wrap the view in an ArrayReg to use Yao's dispatch
    reg_view = ArrayReg(view(reg.state, :, 1))

    for n in 1:L
        # Calculate Probability of outcome +1 at site n using optimized kernel
        prob_plus = real(Yao.expect(put(L, n => prob_opp), reg_view))
        prob_minus = real(Yao.expect(put(L, n => prob_opm), reg_view))
        real_pp = prob_plus / (prob_plus + prob_minus)
        
        # Branching logic
        pm[n] = rand(Bernoulli(clamp(real_pp, 0.0, 1.0))) ? 1 : -1
    end

    return pm
end

###---evolution block, q steps---###
function evolution_block!(q, loc_n, L, Q0_reg, eta)
    # Pre-allocate measurement vector
    pm = zeros(Int, L)
    
    # Pre-define locations
    locU1 = [(2*i-1, 2i) for i in 1:div(L,2)]
    locU2 = [(2*i, 2i+1) for i in 1:(div(L,2)-1)]
    locM = 1:L

    for ti in 1:q
        # --- Layer U1 ---
        U1 = [haar_unitary(loc_n^2) for _ in 1:div(L, 2)]
        # layerU1 = chain([put(L, loc => matblock(u)) for (loc, u) in zip(locU1, U1)]...)
        apply!(Q0_reg, chain([put(L, loc => matblock(u)) for (loc, u) in zip(locU1, U1)]...)) # In-place update

        # --- Measure 1 ---
        measure_out!(pm, Q0_reg, eta, L) # Determine outcomes
        M1 = [matblock(Mpm(pm[i], eta)) for i in 1:L]
        # layerM1 = chain([put(L, loc => m) for (loc, m) in zip(locM, M1)]...)
        apply!(Q0_reg, chain([put(L, loc => m) for (loc, m) in zip(locM, M1)]...)) # In-place update

        # --- Layer U2 ---
        U2 = [haar_unitary(loc_n^2) for _ in 1:(div(L,2)-1)]
        # layerU2 = chain([put(L, loc => matblock(u)) for (loc, u) in zip(locU2, U2)]...)
        apply!(Q0_reg, chain([put(L, loc => matblock(u)) for (loc, u) in zip(locU2, U2)]...)) # In-place update

        # --- Measure 2 ---
        measure_out!(pm, Q0_reg, eta, L) # Determine outcomes
        M2 = [matblock(Mpm(pm[i], eta)) for i in 1:L]
        # layerM2 = chain([put(L, loc => m) for (loc, m) in zip(locM, M2)]...)
        apply!(Q0_reg, chain([put(L, loc => m) for (loc, m) in zip(locM, M2)]...)) # In-place update

        Q0_reg.state[:, 1] ./= norm(Q0_reg.state[:, 1])
    end
    
    return Q0_reg
end

# L = 12
# Q0 = rand_state(12;nbatch=10)
# evolution_block!(q, loc_n, L, Q0, eta)
function warm_test(q, n_warm, loc_n, L, eta, N_trj)
    Q0_test = rand_state(L; nbatch = N_trj)

    if n_warm > 0
        for i in 1:n_warm
            evolution_block!(q, loc_n, L, Q0_test, eta)
            Q_new, _ = qr_pos!(Q0_test.state[:, 2:end])
            Q0_test.state[:, 2:end] .= Q_new
        end
    end

    evolution_block!(q, loc_n, L, Q0_test, eta)
    return cond(Q0_test.state[:,2:end]), [norm(Q0_test.state[:,i]) for i in 1:N_trj]
end

function Lyap(q, r, sig0, loc_n, L, eta, N_trj)
    # Initialize register directly
    Q0_reg = rand_state(L; nbatch = N_trj+1) # First column reserved for evolution
    
    sig = 5 * sig0
    s = 0
    D_arr = Vector{Float64}[]
    
    # Pre-allocate storage for Lyapunov exponents
    Rii = zeros(Float64, N_trj, r)
    
    while sig > sig0
        for i in 1:r
            # 1. Evolve (Q0_reg is updated in-place)
            evolution_block!(q, loc_n, L, Q0_reg, eta)

            # 2. QR Decomp (access dense state directly)
            Q_new, R = qr_pos!(Q0_reg.state[:, 2:end])
            
            # 3. Calculate exponents
            Rii[:, i] = -log.(diag(R)) / q
            
            # 4. Update state with orthonormalized Q
            Q0_reg.state[:, 2:end] .= Q_new
        end
        
        push!(D_arr, vec(mean(Rii, dims=2)))
        
        # Convergence check
        if s > 50
            D_stack = stack(D_arr)
            sig = maximum([sem(D_stack[ii, :]) for ii in 1:N_trj])
        end
        s += 1
        println("Step: $s, Sigma: $sig") # Optional progress tracking
    end

    D = stack(D_arr)
    return D, s, sig
end

q, r = 8, 5
loc_n = 2
L = 10 # Optimized code can now handle L=14+ more easily
N_trj = 10
eta = 0.7
sig0 = 0.01

# n_warm = r
# warm_test(q, n_warm, loc_n, L, eta, N_trj)

println("Starting simulation...")
@time lam, sf, sigf = Lyap(q, r, sig0, loc_n, L, eta, N_trj)
println("Total time steps: ", sf*q*r, " Final Sigma: ", sigf)

tt1 = vec(mean(lam, dims=2))
tt = zeros(N_trj,5)
tt[:,1] = tt1 .-minimum(tt1)

f2 = Figure()
ax2 = Axis(f2[1,1])
CairoMakie.scatter!(ax2, cumsum(lam[1, :]) ./ (1:size(lam,2)))
# CairoMakie.hist!(ax2, lam[1, :], bins=20)
f2

xs = vec(ones(size(tt[:,1:4], 1)) .* (1:size(tt[:,1:4], 2))')
ys = vec(tt[:,1:4])

f1 = Figure()
ax1 = Axis(f1[1,1])
CairoMakie.scatter!(ax1,xs,ys)
f1

# gap = lam[2]-lam[1]

