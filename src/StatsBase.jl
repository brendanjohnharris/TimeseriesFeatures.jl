using Statistics
function maybe_autocor end
function maybe_pacf end
function maybe_median end

ac_lags = 1:40
ACF = Feature(x -> maybe_autocor(x, ac_lags; demean = true), :ACF,
              "Autocorrelation function to lag $(maximum(ac_lags))", ["autocorrelation"])

AC = SuperFeatureSet([x -> x[ℓ] for ℓ in eachindex(ac_lags)],
                     Symbol.(["AC_$ℓ" for ℓ in ac_lags]),
                     ["Autocorrelation at lag $ℓ" for ℓ in ac_lags],
                     [["correlation"] for ℓ in ac_lags],
                     ACF) # We compute the ACF just once, and pick off results for each feature
export AC

PACF = Feature(x -> maybe_pacf(x, ac_lags; method = :regression), :PACF,
               "Partial autocorrelation function to lag $(maximum(ac_lags))",
               ["autocorrelation"])

Partial_AC = SuperFeatureSet([x -> x[ℓ] for ℓ in eachindex(ac_lags)],
                             Symbol.(["Partial_AC_$ℓ" for ℓ in ac_lags]),
                             ["Partial autocorrelation at lag $ℓ (regression method)"
                              for ℓ in ac_lags],
                             [["correlation"] for ℓ in ac_lags],
                             PACF)
export Partial_AC

function firstcrossing(r, threshold = 0)
    if first(r) < threshold
        idx = findfirst(r .> threshold)
    elseif first(r) > threshold
        idx = findfirst(r .< threshold)
    elseif first(r) == threshold
        return 1
    elseif all(r) .> threshold || all(r) .< threshold
        return nothing
    end
    b = r[idx]
    a = r[idx - 1]
    return idx - 1 + (threshold - a) / (b - a)
end

function firstcrossingacf(x, threshold = 0)
    lagchunks = min(100, length(x) - 1)
    lags = 1:lagchunks
    i = 1
    r1 = sign(maybe_autocor(x, [1]; demean = true) |> first) # If the time series is anticorrelated with itself, we look for the first upward crossing over the threshold
    threshold = threshold * r1
    while i * lagchunks < length(x)
        r = maybe_autocor(x, lags; demean = true) .* r1
        lastr = r[end]
        if any(r .< threshold)
            idx = findfirst(r .< threshold)
            b = r[idx]
            a = idx == 1 ? lastr : r[idx - 1]
            idx += (i - 1) * lagchunks
            return idx - 1 + (threshold - a) / (b - a)
        else
            lags = lags .+ lagchunks
            i += 1
        end
    end
end

"""
    RAD(x, τ=1, doAbs=true)
Compute the rescaled auto-density, a metric for inferring the
distance to criticality that is insensitive to uncertainty in the noise strength.
Calibrated to experiments on the Hopf bifurcation with variable and unknown
measurement noise.

Inputs:
    x:      The input time series (vector).
    doAbs:  Whether to centre the time series at 0 then take absolute values (logical flag)
    τ:      The embedding and differencing delay in units of the timestep (integer), or :τ

Outputs:
    f:      The RAD feature value
"""
function RAD(z, τ = 1, doAbs = true)
    if doAbs
        z = z .- maybe_median(z)
        z = abs.(z)
    end
    if τ === :τ
        # Make τ the first zero crossing of the autocorrelation function
        τ = round(Int, firstcrossingacf(z, 0))
    elseif !isinteger(τ)
        error("τ must be an integer or :τ")
    end

    y = @view z[(τ + 1):end]
    x = @view z[1:(end - τ)]

    # Median split
    subMedians = x .< maybe_median(x)
    superMedianSD = std(x[.!subMedians])
    subMedianSD = std(x[subMedians])

    # Properties of the auto-density
    sigma_dx = std(y - x)
    densityDifference = 1 / superMedianSD - 1 / subMedianSD

    f = sigma_dx * densityDifference
end
RAD(z::AbstractDimArray, args...; kwargs...) = RAD(parent(z), args...; kwargs...)

CR_RAD = Feature(x -> RAD(x, 1, true), :CR_RAD,
                 "Rescaled Auto-Density criticality metric (centered)",
                 ["criticality"])
CR_RAD_raw = Feature(x -> RAD(x, 1, false), :CR_RAD_raw,
                     "Rescaled Auto-Density criticality metric (uncentered)",
                     ["criticality"])
export RAD, CR_RAD, CR_RAD_raw, firstcrossingacf, firstcrossing
