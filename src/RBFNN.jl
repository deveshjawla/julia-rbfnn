module RBFNN
using Clustering
using Statistics
using Flux, ParameterSchedulers
using ParameterSchedulers: Scheduler
using Distances

include("computeRBFBetas.jl")
include("computeCentroids.jl")
include("computeRBFActivations.jl")
include("computeWeights.jl")
include("trainRBF.jl")
include("testRBF.jl")

export trainRbf, testRbf
end