using ..DSP

Analytic_Signal = Feature(
    hilbert, :Analytic_Signal,
    "Analytic signal of the time series, from the Hilbert Transform",
    ["transform", "phase", "amplitude", "hilbert"]
)
Analytic_Phase = Feature(
    x -> x |> hilbert .|> angle, :Analytic_Phase,
    "Analytic phase of the time series, from the Hilbert Transform",
    ["transform", "phase", "hilbert"]
)
Analytic_Amplitude = Feature(
    x -> x |> hilbert .|> abs, :Analytic_Amplitude,
    "Analytic amplitude of the time series, from the Hilbert Transform",
    ["transform", "amplitude"]
)

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

PPC = PairwiseFeature(
    pairwisephaseconsistency, :PPC,
    "The pairwise-phase consistency, an unbiased estimate of the phase-locking value",
    ["synchrony", "phase"]
) # Assumes phase time series

PPC_Analytic_Phase = SuperFeature(PPC, Analytic_Phase; merge = true)

phaselockingvalue(x::AbstractVector) = exp.(im .* x) |> mean |> abs

function phaselockingvalue(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y)
    return phaselockingvalue(y .- x)
end

PLV = PairwiseFeature(
    phaselockingvalue, :PLV,
    "The phase-locking value", ["synchrony", "phase"]
) # Assumes phase time series

PLV_Analytic_Phase = SuperFeature(PLV, Analytic_Phase; merge = true)

"""
    bandpower(p, lo, hi)

Integrate the power spectral density `p` over the half-open band `[lo, hi)`, so that
successive bands do not share a frequency bin. A band containing no bin of `p` integrates to
`0`, so bands should be at least as wide as the frequency resolution of `p`.
"""
function bandpower(p, lo, hi)
    f = DSP.freq(p)
    return sum(DSP.power(p)[lo .≤ f .< hi]) * step(f)
end

_numname(x) = x isa AbstractFloat && isinteger(x) ? Integer(x) : x # 1000 and 1000.0 alike

"""
    BandPower(edges; fs = 1, name = Symbol(:BandPower_, fs), psd = x -> welch_pgram(x; fs, window = hanning))

A `SuperFeatureSet` giving the power in each of the `length(edges) - 1` successive
frequency bands `[edges[i], edges[i + 1])`.
Frequencies are in the same units as `fs`, which defaults to 1 (so that bands are given in
cycles per sample). The power spectral density is estimated by `psd` and shared across all
bands, so it is computed only once per time series.

The default `psd` uses Welch segments of `length(x) ÷ 8`, giving a frequency resolution of
`8fs / length(x)`; bands narrower than this contain no frequency bin and so are `0`. Pass a
`psd` with longer segments (and a matching `name`) to resolve narrow bands.

Features are identified by their names alone, so `name` carries `fs`: two sets differing in
`fs` are named apart and keep their own spectra when combined, while two sets sharing an
`fs` correctly share one spectrum. Names use numeric values rather than types, so `fs = 1000`
and `fs = 1000.0` share a spectrum. A custom `psd` is not reflected in the name and so needs
its own `name`; otherwise its spectrum aliases to that of any set with a matching `fs`.

# Examples
```julia
𝒇 = BandPower(0:5:100; fs = 1000) # 20 linearly spaced bands
𝒈 = BandPower(exp10.(range(0, 2, 21)); fs = 1000) # shares the spectrum of 𝒇
F = (𝒇 + 𝒈)(x)
```
"""
function BandPower(
        edges; fs = 1, name = Symbol(:BandPower_, _numname(fs)),
        psd = x -> welch_pgram(x; fs, window = hanning) # Hann limits leakage between bands
    )
    super = Feature(
        psd, Symbol(name, :_PSD),
        "Power spectral density (fs = $fs)", ["spectral"]
    )
    bands = collect(zip(edges[1:(end - 1)], edges[2:end]))
    return SuperFeatureSet(
        [p -> bandpower(p, lo, hi) for (lo, hi) in bands],
        [Symbol(name, :_, _numname(lo), :_, _numname(hi)) for (lo, hi) in bands],
        ["Power in the frequency band [$lo, $hi)" for (lo, hi) in bands],
        [["spectral", "power"] for _ in bands],
        super
    )
end

export Analytic_Amplitude, Analytic_Phase, Analytic_Signal, pairwisephaseconsistency,
    phaselockingvalue, PPC_Analytic_Phase, PPC, PLV, PLV_Analytic_Phase,
    bandpower, BandPower
