# module AssociationsExt
using .Associations
# using TimeseriesFeatures

MI_Kraskov_NN_20 = PairwiseFeature((x, y) -> association(KSG1(; k = 20), x, y),
                                   :MI_Kraskov_NN_20,
                                   "Mutual Information using the Kraskov-1 estimator",
                                   ["information_theory", "mutual_information"])

export MI_Kraskov_NN_20

# end
