using LinearAlgebra
using Kronecker
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
    return Array([(1 + pm * eta) 0; 0 (1 - pm * eta)]) ./ sqrt(2 * (1 + eta^2))
end

###---measurement outcomes---###
function measure_out(states, eta, L)
    eye = [1.0 0.0; 0.0 1.0]
    Mn = Mpm(1, eta)
    p = zeros(L)
    for n in 1:L
        temp = kronecker([i == n ? Mn : eye for i in 1:L]...)
        p[n] = real.(dot(temp*states, temp*states))
    end
    
    return p
end

###---evolution block, q steps---###
function evolution_block!(q, loc_n, L, Q0, eta)
    dim, N_trj = size(Q0)
    Q0_temp = copy(Q0)

    for ti in 1:q
        U1 = kronecker([haar_unitary(loc_n^2) for _ in 1:div(L, 2)]...)
        # U1 = foldl(kron, U1_arr)
        Q0_temp[:,:] = U1*Q0_temp

        # generate Born rule and measurement operation
        p1 = measure_out(Q0_temp[:,1], eta, L)
        pm1 = [rand(Bernoulli(prob)) ? 1 : -1 for prob in p1]
        M1 = kronecker([Mpm(pm1[i], eta) for i in 1:L]...)
        Q0_temp[:,:] = M1*Q0_temp

        U2 = kronecker(vcat([Matrix(I,loc_n,loc_n)], [haar_unitary(loc_n^2) for _ in 1:(div(L, 2)-1)], [Matrix(I,loc_n,loc_n)])...)
        # U1 = foldl(kron, U1_arr)
        Q0_temp[:,:] = U2*Q0_temp

        # generate Born rule and measurement operation
        p2 = measure_out(Q0_temp[:,1], eta, L)
        pm2 = [rand(Bernoulli(prob)) ? 1 : -1 for prob in p2]
        M2 = kronecker([Mpm(pm2[i], eta) for i in 1:L]...)
        Q0_temp[:,:] = M2*Q0_temp
    end

    return Q0_temp
end

# tt = vcat([Matrix(I,loc_n,loc_n)], [haar_unitary(loc_n^2) for _ in 1:(div(L, 2)-1)], [Matrix(I,loc_n,loc_n)])
# tt[1]

function Lyap(q, r, sig0, loc_n, L, eta, N_trj)
    dim = loc_n^L

    # initial states
    temp = (randn(dim,dim) + im * randn(dim,dim)) / sqrt(2.0)
    F_temp = qr(temp)
    Q0 = F_temp.Q[:,1:N_trj]
    
    sig = 5*sig0
    s = 0
    D_arr = Vector{Float64}[]
    while sig > sig0
        Rii = zeros(N_trj,r)
        for i in 1:r
            Q0[:,:],R = qr_pos(evolution_block!(q, loc_n, L, Q0, eta))
            Rii[:,i] = -log.(diag(R))/q
        end
        push!(D_arr,vec(mean(Rii,dims=2)))
        if s > 500
            # sig = sem.(eachrow(A))
            D = stack(D_arr)
            sig = sem(D[1,:])
        end
        s += 1
    end

    D = stack(D_arr)
    return vec(mean(D,dims=2)),s,sig
end

q,r = 10,7
loc_n = 2
L = 6
dim =loc_n^L
N_trj = 10
eta = 0.1

sig0 = 0.01
lam, sf,sigf = Lyap(q, r, sig0, loc_n, L, eta, N_trj)
t_tot = q*r*sf

# # tt = zeros(N_trj,5)
# tt[:,1] = lam .-minimum(lam)

# xs = vec(ones(size(tt[:,1:3], 1)) .* (1:size(tt[:,1:3], 2))')
# ys = vec(tt[:,1:3])

# f1 = Figure()
# ax1 = Axis(f1[1,1])
# CairoMakie.scatter!(ax1,xs,ys)
# f1

# gap = lam[2]-lam[1]

