function maybe_association(::Any...)
    return error("Associations.jl must be loaded to use this feature; run `using Associations`")
end

MI_Kraskov_NN_20 = PairwiseFeature(
    (x, y) -> maybe_association(x, y),
    :MI_Kraskov_NN_20,
    "Mutual Information using the Kraskov-1 estimator",
    ["information_theory", "mutual_information"]
)
export MI_Kraskov_NN_20
