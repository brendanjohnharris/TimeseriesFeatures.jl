# module CausalityToolsExt
using .CausalityTools
# using TimeseriesFeatures

MI_Lord_NN_20 = SPI((x, y) -> mutualinfo(Lord(k = 20), x, y),
                    :MI_Lord_NN_20,
                    "Mutual Information using the Lord Estimator, 4 nearest neighbours",
                    ["information_theory", "mutual_information"])

export MI_Lord_NN_20

# end
