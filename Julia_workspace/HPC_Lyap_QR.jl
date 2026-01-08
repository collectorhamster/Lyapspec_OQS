using LinearAlgebra
using Yao
using Distributions, Statistics, StatsBase
using CSV, Tables, DataFrames

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
        prob_plus = real(expect(put(L, n => prob_opp), reg_view))
        prob_minus = real(expect(put(L, n => prob_opm), reg_view))
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
        apply!(Q0_reg, chain([put(L, loc => matblock(u)) for (loc, u) in zip(locU1, U1)]...)) # In-place update

        # --- Measure 1 ---
        measure_out!(pm, Q0_reg, eta, L) # Determine outcomes
        M1 = [matblock(Mpm(pm[i], eta)) for i in 1:L]
        apply!(Q0_reg, chain([put(L, loc => m) for (loc, m) in zip(locM, M1)]...)) # In-place update

        # --- Layer U2 ---
        U2 = [haar_unitary(loc_n^2) for _ in 1:(div(L,2)-1)]
        apply!(Q0_reg, chain([put(L, loc => matblock(u)) for (loc, u) in zip(locU2, U2)]...)) # In-place update

        # --- Measure 2 ---
        measure_out!(pm, Q0_reg, eta, L) # Determine outcomes
        M2 = [matblock(Mpm(pm[i], eta)) for i in 1:L]
        apply!(Q0_reg, chain([put(L, loc => m) for (loc, m) in zip(locM, M2)]...)) # In-place update
    end
    
    return Q0_reg
end

function Lyap(q, r, s0, sig0, loc_n, L, eta, N_trj)
    # Initialize register directly
    Q0_reg = rand_state(L; nbatch = N_trj)
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
            Q_new, R = qr_pos!(Q0_reg.state)
            
            # 3. Calculate exponents
            Rii[:, i] = -log.(diag(R)) / q
            
            # 4. Update state with orthonormalized Q
            Q0_reg.state .= Q_new
        end
        push!(D_arr, vec(mean(Rii, dims=2)))
        
        # Convergence check
        if s > s0
            D_stack = stack(D_arr)
            sig = maximum([sem(D_stack[ii, :]) for ii in 1:N_trj])
        end
        s += 1
        println("Step: $s, Sigma: $sig") # Optional progress tracking
    end

    D = stack(D_arr)
    return vec(mean(D, dims=2)), s, sig
end

q, r = 8, 10
loc_n = 2
L = 14 # Optimized code can now handle L=14+ more easily
N_trj = 1
eta = 0.3
sig0 = 0.01
s0 = 100

println("Starting simulation...")
@time lam, sf, sigf = Lyap(q, r, s0, sig0, loc_n, L, eta, N_trj)
println("Total time steps: ", sf*q*r, " Final Sigma: ", sigf)

path = ""
fname = path*"eps.csv"
CSV.write(fname,  Tables.table(lam), writeheader=false,append=false)

println("done.")