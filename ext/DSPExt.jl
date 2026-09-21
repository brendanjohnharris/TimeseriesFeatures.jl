module DSPExt
using DSP
using TimeseriesFeatures

TimeseriesFeatures.maybe_hilbert(x::AbstractVector) = hilbert(x)

function TimeseriesFeatures.maybe_welch_pgram(x::AbstractVector; kwargs...)
    return welch_pgram(x; window = hanning, kwargs...) # Hann limits leakage between bands
end

function TimeseriesFeatures.bandpower(p, lo, hi)
    f = freq(p)
    return sum(power(p)[lo .≤ f .< hi]) * step(f)
end

end
