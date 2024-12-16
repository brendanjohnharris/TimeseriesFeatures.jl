using ..DSP

export Analytic_Amplitude, Analytic_Phase, Analytic_Signal, pairwisephaseconsistency,
       phaselockingvalue, PPC_Analytic_Phase, PPC, PLV, PLV_Analytic_Phase

Analytic_Signal = Feature(hilbert, :Analytic_Signal,
                          "Analytic signal of the time series, from the Hilbert Transform",
                          ["transform", "phase", "amplitude", "hilbert"])
Analytic_Phase = Feature(x -> x |> hilbert .|> angle, :Analytic_Phase,
                         "Analytic phase of the time series, from the Hilbert Transform",
                         ["transform", "phase", "hilbert"])
Analytic_Amplitude = Feature(x -> x |> hilbert .|> abs, :Analytic_Amplitude,
                             "Analytic amplitude of the time series, from the Hilbert Transform",
                             ["transform", "amplitude"])

function pairwisephaseconsistency(x::AbstractVector) # Eq. 14 of Vinck 2010
    N = length(x)
    Δ = zeros(N - 1)
    Threads.@threads for i in 1:(N - 1)
        δ = @views x[i] .- x[(i + 1):end]
        Δ[i] = sum(cos.(δ))
    end
    return (2 / (N * (N - 1))) * sum(Δ)
end
function pairwisephaseconsistency(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y)
    return pairwisephaseconsistency(y .- x)
end

PPC = PairwiseFeature(pairwisephaseconsistency, :PPC,
                      "The pairwise-phase consistency, an unbiased estimate of the phase-locking value",
                      ["synchrony", "phase"]) # Assumes phase time series

PPC_Analytic_Phase = SuperFeature(PPC, Analytic_Phase; merge = true)

phaselockingvalue(x::AbstractVector) = exp.(im .* x) |> mean |> abs

function phaselockingvalue(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y)
    return phaselockingvalue(y .- x)
end

PLV = PairwiseFeature(phaselockingvalue, :PLV,
                      "The phase-locking value", ["synchrony", "phase"]) # Assumes phase time series

PLV_Analytic_Phase = SuperFeature(PLV, Analytic_Phase; merge = true)
