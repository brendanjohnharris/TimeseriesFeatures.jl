module AssociationsExt
using Associations
using TimeseriesFeatures

function TimeseriesFeatures.maybe_association(x::AbstractVector, y::AbstractVector)
    return association(KSG1(; k = 20), x, y)
end

end
