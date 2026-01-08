using LinearAlgebra
using CairoMakie
import DifferentialEquations as DE
using QuantumToolbox

function jump!(state, Lm, gamma, dt)
    p = zeros(length(Lm)-1)
    Heff = copy(Lm[1])

    for (k, i) in enumerate(2:length(Lm))
        jp = sqrt(gamma) * Lm[i]
        p[k] = max(real(dot(state, jp' * jp * state)), 0.0) * dt
        Heff -= (im * jp' * jp) / 2
    end

    r1 = rand()
    totalp = sum(p)
    if r1 > totalp
        state .= state .- im * dt * (Heff * state)
        state ./= norm(state)
    else
        r2 = rand()
        if totalp > 0
            p ./= totalp
        end
        cumsum_p = cumsum(p)
        idx = searchsortedfirst(cumsum_p, r2)
        jp = sqrt(gamma) * Lm[idx + 1]
        state .= jp * state
        state ./= norm(state)
    end
    return state
end

function evolution(state0, ob, Lm, gamma, dt, Nt, N_trj)
    obexp = zeros(Nt, N_trj)
    for tj in 1:N_trj
        state = copy(state0)
        for ti in 1:Nt
            obexp[ti, tj] = real.(state' * ob * state)
            jump!(state, Lm, gamma, dt)
        end
    end
    return obexp
end

sigx = [0.0 1.0; 1.0 0.0]
sigy = [0.0 -im; im 0.0]
sigz = [1.0 0.0; 0.0 -1.0]
sig0 = [1.0 0.0; 0.0 1.0]
sigp = (sigx + im * sigy) / 2
sigm = (sigx - im * sigy) / 2

Omeg, Delt, Gamma = 1.0, 0.0, 1.0/6
H = -(Omeg/2) * sigx - Delt * sigp * sigm
Lm = [H, sigm]

# v0 = rand(ComplexF64, 2)
v0 = complex([1.0, 0.0])
ob = (sig0 - sigz) / 2
dt, Nt = 0.005, 8000
N_trj = 1000
anw = evolution(v0, ob, Lm, Gamma, dt, Nt, N_trj)

t_arr = 0:dt:(Nt-1)*dt
f1 = Figure()
ax1 = Axis(f1[1,1], limits=(nothing, nothing, 0, 1))
CairoMakie.plot!(ax1, 0:dt:(Nt-1)*dt, vec(mean(anw, dims=2)))
f1

### directly integration
function two_level!(du, u, p, t)
    Omeg, Delt, Gamma = p[:]
    du[1] = (im*Delt-Gamma/2)*u[1] - im*(Omeg/2)*u[3] + im*(Omeg/2)*u[4]
    du[2] = (-im*Delt-Gamma/2)*u[2] + im*(Omeg/2)*u[3] - im*(Omeg/2)*u[4]
    du[3] = -im*(Omeg/2)*u[1] + im*(Omeg/2)*u[2] - Gamma*u[3]
    du[4] = im*(Omeg/2)*u[1] - im*(Omeg/2)*u[2] + Gamma*u[3]
end

p = [Omeg, Delt, Gamma]
u0 = complex.([0.0; 0.0; 0.0; 1.0])
tspan = (0.0, 40.0)
prob = DE.ODEProblem(two_level!, u0, tspan, p)
sol = DE.solve(prob)
t2 = sol.t
anw2 = real.(sol[3,:])

CairoMakie.scatter!(ax1, t2, anw2, color=:red)
f1

### QuTip
tlist = LinRange(0.0, 40.0, 2000)

ψ0 = fock(2, 0)
H = -Omeg/2 * sigmax() - Delt * sigmap() * sigmam()
c_ops = [sqrt(Gamma) * sigmam()]
e_ops = [(I(size(sigmaz(),1)) .- sigmaz()) / 2]

sol_500 = mcsolve(H, ψ0, tlist, c_ops, e_ops=e_ops; ntraj=1000)

CairoMakie.lines!(ax1, tlist, real(sol_500.expect)[1,:])
f1