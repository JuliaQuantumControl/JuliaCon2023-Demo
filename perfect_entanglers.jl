# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Julia 1.9.2
#     language: julia
#     name: julia-1.9
# ---

# # Example: Optimization of a Perfectly Entangling Quantum gate

# $
# \newcommand{tr}[0]{\operatorname{tr}}
# \newcommand{diag}[0]{\operatorname{diag}}
# \newcommand{abs}[0]{\operatorname{abs}}
# \newcommand{pop}[0]{\operatorname{pop}}
# \newcommand{aux}[0]{\text{aux}}
# \newcommand{opt}[0]{\text{opt}}
# \newcommand{tgt}[0]{\text{tgt}}
# \newcommand{init}[0]{\text{init}}
# \newcommand{lab}[0]{\text{lab}}
# \newcommand{rwa}[0]{\text{rwa}}
# \newcommand{bra}[1]{\langle#1\vert}
# \newcommand{ket}[1]{\vert#1\rangle}
# \newcommand{Bra}[1]{\left\langle#1\right\vert}
# \newcommand{Ket}[1]{\left\vert#1\right\rangle}
# \newcommand{Braket}[2]{\left\langle #1\vphantom{#2}\mid{#2}\vphantom{#1}\right\rangle}
# \newcommand{op}[1]{\hat{#1}}
# \newcommand{Op}[1]{\hat{#1}}
# \newcommand{dd}[0]{\,\text{d}}
# \newcommand{Liouville}[0]{\mathcal{L}}
# \newcommand{DynMap}[0]{\mathcal{E}}
# \newcommand{identity}[0]{\mathbf{1}}
# \newcommand{Norm}[1]{\lVert#1\rVert}
# \newcommand{Abs}[1]{\left\vert#1\right\vert}
# \newcommand{avg}[1]{\langle#1\rangle}
# \newcommand{Avg}[1]{\left\langle#1\right\rangle}
# \newcommand{AbsSq}[1]{\left\vert#1\right\vert^2}
# \newcommand{Re}[0]{\operatorname{Re}}
# \newcommand{Im}[0]{\operatorname{Im}}
# $

# ## Two Transmon qubits with a shared transmission line

#  ![Two transmons with a shared transmission line](transmon.png)

# ### Hamiltonian

# The energies system energies are on the order of GHz (angular frequency; the factor
# 2π is implicit), with dynamics on the order of ns

const GHz = 2π
const MHz = 0.001GHz
const ns = 1.0
const μs = 1000ns;
⊗ = kron;
const 𝕚 = 1im;

# We truncated the Hamiltonian to $N$ levels

const N = 6;  # levels per transmon

# So the dimension of the total Hilbert space is $N^2 = 36$

# The Hamiltonian and parameters are taken from
# [Goerz *et al.*, Phys. Rev. A 91, 062307 (2015); Table 1](https://michaelgoerz.net/#GoerzPRA2015).

# +
using LinearAlgebra
using SparseArrays
using QuantumControl: hamiltonian

function transmon_hamiltonian(;
    Ωre, Ωim, N=N, ω₁=4.380GHz, ω₂=4.614GHz, ωd=4.498GHz, α₁=-210MHz,
    α₂=-215MHz, J=-3MHz, λ=1.03,
)
    𝟙 = SparseMatrixCSC{ComplexF64,Int64}(sparse(I, N, N))
    b̂₁ = spdiagm(1 => complex.(sqrt.(collect(1:N-1)))) ⊗ 𝟙
    b̂₂ = 𝟙 ⊗ spdiagm(1 => complex.(sqrt.(collect(1:N-1))))
    b̂₁⁺ = sparse(b̂₁'); b̂₂⁺ = sparse(b̂₂')
    n̂₁ = sparse(b̂₁' * b̂₁); n̂₂ = sparse(b̂₂' * b̂₂)
    n̂₁² = sparse(n̂₁ * n̂₁); n̂₂² = sparse(n̂₂ * n̂₂)
    b̂₁⁺_b̂₂ = sparse(b̂₁' * b̂₂); b̂₁_b̂₂⁺ = sparse(b̂₁ * b̂₂')

    # rotating frame: ω₁, ω₂ → detuning; driving field Ω ∈ ℂ
    ω̃₁ = ω₁ - ωd; ω̃₂ = ω₂ - ωd

    Ĥ₀ = sparse(
        (ω̃₁ - α₁ / 2) * n̂₁ +
        (α₁ / 2) * n̂₁² +
        (ω̃₂ - α₂ / 2) * n̂₂ +
        (α₂ / 2) * n̂₂² +
        J * (b̂₁⁺_b̂₂ + b̂₁_b̂₂⁺)
    )
    Ĥ₁re = sparse((1 / 2) * (b̂₁ + b̂₁⁺ + λ * b̂₂ + λ * b̂₂⁺))
    Ĥ₁im = sparse((𝕚 / 2) * (b̂₁⁺ - b̂₁ + λ * b̂₂⁺ - λ * b̂₂))
    return hamiltonian(Ĥ₀, (Ĥ₁re, Ωre), (Ĥ₁im, Ωim))
end;
# -

# ...

# ### Initial driving field

# +
using QuantumControl.Amplitudes: ShapedAmplitude
using QuantumControl.Shapes: flattop

function guess_amplitudes(; T=400ns, E₀=35MHz, dt=0.1ns, t_rise=15ns)
    tlist = collect(range(0, T, step=dt))
    shape(t) = flattop(t, T=T, t_rise=t_rise)
    Ωre = ShapedAmplitude(t -> E₀, tlist; shape)
    Ωim = ShapedAmplitude(t -> 0.0, tlist; shape)
    return tlist, Ωre, Ωim
end

tlist, Ωre_guess, Ωim_guess = guess_amplitudes();
# -

include("includes/plot_complex_pulse.jl")

plot_complex_pulse(tlist, Array(Ωre_guess))

H = transmon_hamiltonian(Ωre=Ωre_guess, Ωim=Ωim_guess);

# ### Logical basis

# +
function ket(i::Int64; N=N)
    Ψ = zeros(ComplexF64, N)
    Ψ[i+1] = 1
    return Ψ
end

function ket(indices::Int64...; N=N)
    Ψ = ket(indices[1]; N=N)
    for i in indices[2:end]
        Ψ = Ψ ⊗ ket(i; N=N)
    end
    return Ψ
end

function ket(label::AbstractString; N=N)
    indices = [parse(Int64, digit) for digit in label]
    return ket(indices...; N=N)
end;
# -

basis = [ket("00"), ket("01"), ket("10"), ket("11")];

ket("01")

# ## Dynamics of the guess field

using QuantumControl: propagate

# ...

logical_overlap = [(Ψ -> Ψ ⋅ ϕ) for ϕ ∈ basis];

dyn00 = propagate(ket("00"), H , tlist; observables=logical_overlap, storage=true)
dyn01 = propagate(ket("01"), H , tlist; observables=logical_overlap, storage=true)
dyn10 = propagate(ket("10"), H , tlist; observables=logical_overlap, storage=true)
dyn11 = propagate(ket("11"), H , tlist; observables=logical_overlap, storage=true)

# We concatenate the columns to get the $4 \times 4$ matrix U that is that "gate" in the two-qubit subspace at each point in time:

U_of_t = [[dyn00[:,n] dyn01[:,n] dyn10[:,n] dyn11[:,n]] for n = 1:length(tlist)];

using TwoQubitWeylChamber: gate_concurrence, unitarity

# The `gate_concurrence` is the amount of entanglement that can be generated by applying the gate to a separable input state.
#
# A well-known perfectly entangling gate is the controlled-NOT gate:

CNOT = [
    1 0 0 0
    0 1 0 0
    0 0 0 1
    0 0 1 0
];

gate_concurrence(CNOT)

plot(tlist, gate_concurrence.(U_of_t), xlabel="time (ns)", ylabel="gate concurrence", label="", ylim=(0, 1))

gate_concurrence(U_of_t[end])

# Our guess pulse does not result in a perfectly entangling gate.
#
# Moreover, there is loss of population from the logical subspace, i.e., the $4 \times 4$ matrices in `U_of_t` are not unitary:

plot(tlist, 1 .- unitarity.(U_of_t), xlabel="time (ns)", ylabel="loss from subspace", label="")

1 - unitarity(U_of_t[end])

# ## Maximization of Gate Concurrence

# +
using QuantumControl: Objective

objectives = [Objective(; initial_state=Ψ, generator=H) for Ψ ∈ basis];
# -

J_T_C = U -> 0.5 * (1 - gate_concurrence(U)) + 0.5 * (1 - unitarity(U));

J_T_C(U_of_t[end])

# +
using QuantumControl.Functionals: gate_functional

J_T = gate_functional(J_T_C);
# -

# $J_T$ is now a function of the propagated states $\ket{\Psi_{00}(T)}$, $\ket{\Psi_{01}(T)}$, $\ket{\Psi_{10}(T)}$, $\ket{\Psi_{11}(T)}$.
#
# ...

# +
using QuantumControl.Functionals: make_gate_chi

chi = make_gate_chi(J_T_C, objectives)

# +
using QuantumControl: ControlProblem

problem = ControlProblem(;
    objectives, tlist, J_T, chi,
    check_convergence=res -> begin
        (
            (res.J_T <= 1e-3) &&
            (res.converged = true) &&
            (res.message = "Found a perfect entangler")
        )
    end,
    use_threads=true,
);

# +
using QuantumControl: optimize

res = optimize(problem; method=:GRAPE)

# +
ϵ_opt = res.optimized_controls[1] + 𝕚 * res.optimized_controls[2]
Ω_opt = ϵ_opt .* discretize(Ωre_guess.shape, tlist)

plot_complex_pulse(tlist, Ω_opt)
# -
# ## Dynamics of the optimized field

# +
using QuantumControl.Controls: get_controls

ϵ_re_guess, ϵ_im_guess = get_controls(H);


# +
using QuantumControl.Controls: substitute

H_opt = substitute(
    H,
    IdDict(
        ϵ_re_guess => res.optimized_controls[1],
        ϵ_im_guess => res.optimized_controls[2]
    )
);
# -

dyn00_opt = propagate(ket("00"), H_opt , tlist; observables=logical_overlap, storage=true)
dyn01_opt = propagate(ket("01"), H_opt , tlist; observables=logical_overlap, storage=true)
dyn10_opt = propagate(ket("10"), H_opt , tlist; observables=logical_overlap, storage=true)
dyn11_opt = propagate(ket("11"), H_opt , tlist; observables=logical_overlap, storage=true)
U_opt_of_t = [[dyn00_opt[:,n] dyn01_opt[:,n] dyn10_opt[:,n] dyn11_opt[:,n]] for n = 1:length(tlist)];

plot(tlist, gate_concurrence.(U_opt_of_t), xlabel="time (ns)", ylabel="gate concurrence", label="")
plot!(tlist, gate_concurrence.(U_of_t), label="guess")

gate_concurrence(U_opt_of_t[end])

plot(tlist, 1 .- unitarity.(U_opt_of_t), xlabel="time (ns)", ylabel="loss from subspace", label="")
plot!(tlist, 1 .- unitarity.(U_of_t), label="guess")

1 - unitarity(U_opt_of_t[end])
