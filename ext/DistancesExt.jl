using .TimeseriesFeatures
using .Distances

_distances = [
    Euclidean,
    SqEuclidean,
    # PeriodicEuclidean,
    Cityblock,
    TotalVariation,
    Chebyshev,
    # Minkowski,
    Jaccard,
    BrayCurtis,
    # RogersTanimoto,
    Hamming,
    CosineDist,
    CorrDist,
    ChiSqDist,
    KLDivergence,
    GenKLDivergence,
    JSDivergence,
    # RenyiDivergence,
    SpanNormDist,
    # WeightedEuclidean,
    # WeightedSqEuclidean,
    # WeightedCityblock,
    # WeightedMinkowski,
    # WeightedHamming,
    # SqMahalanobis,
    # Mahalanobis,
    BhattacharyyaDist,
    HellingerDist,
    # Haversine,
    # SphericalAngle,
    MeanAbsDeviation,
    MeanSqDeviation,
    RMSDeviation,
    # Bregman,
    NormRMSDeviation]

distances = map(_distances) do d
    kws = ["distance"]
    if d <: PreMetric
        push!(kws, "premetric")
    end
    if d <: SemiMetric
        push!(kws, "semimetric")
    end
    if d <: Metric
        push!(kws, "metric")
    end
    f(x, y) = begin
        try
            return evaluate(d(), x, y)
        catch e
            @warn e
            return NaN
        end
    end
    SPI(f, Symbol(d), "$d distance", kws)
end

DistanceSPIs = FeatureSet(distances)
export DistanceSPIs
