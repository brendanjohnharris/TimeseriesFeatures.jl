module StatsBaseExt
using StatsBase
using TimeseriesFeatures

TimeseriesFeatures.maybe_autocor(args...; kwargs...) = StatsBase.autocor(args...; kwargs...)
TimeseriesFeatures.maybe_pacf(args...; kwargs...) = StatsBase.pacf(args...; kwargs...)
TimeseriesFeatures.maybe_median(args...; kwargs...) = StatsBase.median(args...; kwargs...)

end
