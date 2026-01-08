using LinearAlgebra
using Yao
using Distributions, Statistics, StatsBase
using CairoMakie

###---positive qr---###
function qr_pos(A)
    F = qr(A)
    # d is a vector of signs (1 or -1)
    # We use Diagonal(d) to perform efficient scaling
    d = sign.(diag(F.R))
    d[d .== 0] .= 1
    D = Diagonal(d)
    
    # Q' = Q * D, R' = D * R
    return F.Q * D, D * F.R
end

###---function for generating Haar-random unitary matrix---###
function haar_unitary(n::Int)
    Z = (randn(n, n) + im * randn(n, n)) / sqrt(2.0)
    F = qr(Z)
    Q = F.Q
    R = F.R
    diagR = diag(R)
    signs = sign.(real(diagR))  # Or: diagR ./ abs.(diagR) for full phase
    signs[signs .== 0] .= 1.0   # Avoid zero
    D = Diagonal(signs)
    U = Q * D
    return Matrix(U)  # Convert to dense matrix if needed
end

# test
# U = haar_unitary(4)
# norm(U' * U - I)

###---local measurement operator---###
function Mpm(pm, eta)
    return Array([(1 + pm * eta) 0; 0 (1 - pm * eta)]) ./ sqrt(2 * (1 + eta^2)) .+0.0im
end

###---measurement outcomes---###
function measure_out(states, eta, loc_n, L)
    pm = zeros(L)
    Mn = Mpm(1, eta)
    states_reg = ArrayReg(states)
    for n in 1:L
        M = put(L, n => matblock(Mn))
        states_reg |> M

        prob = abs2(norm(states_reg))
        pm[n] = rand(Bernoulli(prob)) ? 1 : -1
    end
    
    return pm
end

###---evolution block, q steps---###
function evolution_block!(q, loc_n, L, Q0, eta)
    # dim, N_trj = size(Q0)
    # Q0_temp = copy(Q0)
    # Q0_reg = copy(Q0)

    locU1 = [(2*i-1,2i) for i in 1:div(L,2)]
    locU2 = [(2*i,2i+1) for i in 1:(div(L,2)-1)]
    locM = 1:L
    for ti in 1:q
        U1 = [haar_unitary(loc_n^2) for _ in 1:div(L, 2)]
        layerU1 = chain([put(L, loc => matblock(u)) for (loc, u) in zip(locU1, U1)]...)
        Q0 |> layerU1
        Q0_temp = state(Q0)

        # generate Born rule and measurement operation
        pm1 = measure_out(Q0_temp[:,1], eta, loc_n, L)
        M1 = [Mpm(pm1[i], eta) for i in 1:L]
        layerM1 = chain([put(L, loc => matblock(m)) for (loc, m) in zip(locM, M1)]...)
        Q0 |> layerM1

        U2 = [haar_unitary(loc_n^2) for _ in 1:(div(L,2)-1)]
        layerU2 = chain([put(L, loc => matblock(u)) for (loc, u) in zip(locU2, U2)]...)
        Q0 |> layerU2
        Q0_temp = state(Q0)

        # generate Born rule and measurement operation
        pm2 = measure_out(Q0_temp[:,1], eta, loc_n, L)
        M2 = [Mpm(pm2[i], eta) for i in 1:L]
        layerM2 = chain([put(L, loc => matblock(m)) for (loc, m) in zip(locM, M2)]...)
        Q0 |> layerM2   
    end

    return Q0
end

# L = 12
# Q0 = rand_state(12;nbatch=10)
# evolution_block!(q, loc_n, L, Q0, eta)

function Lyap(q, r, sig0, loc_n, L, eta, N_trj)
    # initial states
    Q0 = rand_state(L,nbatch = N_trj)
    
    sig = 5*sig0
    s = 0
    D_arr = Vector{Float64}[]
    while sig > sig0
        Rii = zeros(N_trj,r)
        for i in 1:r
            Q0_temp,R = qr_pos(state(evolution_block!(q, loc_n, L, Q0, eta)))
            Rii[:,i] = -log.(diag(R))/q
            Q0.state = Q0_temp
        end
        push!(D_arr,vec(mean(Rii,dims=2)))
        if s > 50
            # sig = sem.(eachrow(A))
            D = stack(D_arr)
            sig = sem(D[1,:])
        end
        s += 1
        println("Step: $s, Sigma: $sig")
    end

    D = stack(D_arr)
    return vec(mean(D,dims=2)),s,sig
end

q,r = 8,10
loc_n = 2
L = 12
dim =loc_n^L
N_trj = 10
eta = 0.7

sig0 = 0.01
@time lam, sf,sigf = Lyap(q, r, sig0, loc_n, L, eta, N_trj)
println("t_tot: ", sf*q*r, " sig: ", sigf)

# tt = zeros(N_trj,5)
tt[:,1] = lam .-minimum(lam)

xs = vec(ones(size(tt[:,1:4], 1)) .* (1:size(tt[:,1:4], 2))')
ys = vec(tt[:,1:4])

f1 = Figure()
ax1 = Axis(f1[1,1])
CairoMakie.scatter!(ax1,xs,ys)
f1

# gap = lam[2]-lam[1]

