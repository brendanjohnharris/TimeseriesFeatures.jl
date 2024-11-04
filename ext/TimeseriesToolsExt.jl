module TimeseriesToolsExt
import TimeseriesTools: SpikeTrain, times
import TimeseriesFeatures: AbstractPairwiseFeature
using DimensionalData
function (𝑓::AbstractPairwiseFeature)(X::AbstractVector{<:SpikeTrain})
    X = times.(X)
    𝑓(X)
end
function (𝑓::AbstractPairwiseFeature)(X::AbstractArray{<:SpikeTrain})
    X = times.(X)
    𝑓(X)
end

end # module
