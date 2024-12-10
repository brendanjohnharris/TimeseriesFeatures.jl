module TimeseriesToolsExt
using TimeseriesTools
import TimeseriesFeatures: firstcrossingacf
import TimeseriesTools: timescale, UnivariateRegular

function timescale(x::UnivariateRegular, ::Val{:ac_crossing}; threshold = 0)
    τ = firstcrossingacf(parent(x), threshold)
    return τ * step(x)
end

end
