using ..DSP

export Analytic_Amplitude, Analytic_Phase, Analytic_Signal, pairwisephaseconsistency,
       phaselockingvalue, PPC_Analytic_Phase, PPC, PLV, PLV_Analytic_Phase

Analytic_Signal = Feature(hilbert, :Analytic_Signal,
                          ["transform", "phase", "amplitude", "hilbert"],
                          "Analytic signal of the time series, from the Hilbert Transform")
Analytic_Phase = Feature(x -> x |> hilbert .|> angle, :Analytic_Phase,
                         ["transform", "phase", "hilbert"],
                         "Analytic phase of the time series, from the Hilbert Transform")
Analytic_Amplitude = Feature(x -> x |> hilbert .|> abs, :Analytic_Amplitude,
                             ["transform", "amplitude"],
                             "Analytic amplitude of the time series, from the Hilbert Transform")

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

PPC = PairwiseFeature(pairwisephaseconsistency, :PPC, ["synchrony", "phase"],
                      "The pairwise-phase consistency, an unbiased estimate of the phase-locking value") # Assumes phase time series

PPC_Analytic_Phase = Super(PPC, Analytic_Phase)

phaselockingvalue(x::AbstractVector) = exp.(im .* x) |> mean |> abs

function phaselockingvalue(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y)
    return phaselockingvalue(y .- x)
end

PLV = PairwiseFeature(phaselockingvalue, :PLV, ["synchrony", "phase"],
                      "The phase-locking value") # Assumes phase time series

PLV_Analytic_Phase = Super(PLV, Analytic_Phase)
