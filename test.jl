using Clustering
using Plots
using CSV, DataFrames, Printf
using Statistics
using Flux, ParameterSchedulers
using ParameterSchedulers: Scheduler
using Distances

include("computeRBFBetas.jl")
include("computeCentroids.jl")
include("computeRBFActivations.jl")
include("computeWeights.jl")
include("trainRbf.jl")

data = CSV.read("dataset.csv", DataFrame, header=false)
numdims = size(data, 2) - 1
X = Matrix{Float32}(data[:, 1:numdims])
y = data[:, numdims+1]
# @printf("\n ========================= \n")
# @printf("\n X \n")
# show(size(X))
# @printf("\n")
# @printf("\n y \n")
# show(size(y))
# @printf("\n")

gridSize = 100
u = range(-2.0f0, 2.0f0, length=gridSize)
v = range(-2.0f0, 2.0f0, length=gridSize)

#count = 100
#t = zeros(count, 1)
#for timingIndex = 1:count
#	tic();
(Centers, betas, Theta, nn) = trainRbf(X, y, 20, false)
#	t[timingIndex] = toc();
#end

########################################
# Let's test
model = nn(Theta)

# # ================================================
# #       Evaluate the RBFNN over the grid.
# # ================================================
# # Define a grid over which to evaluate the RBFN.
# # We'll store the scores for each category as well as the 'prediction' for
# # each point on the grid.
# scores1 = zeros(lastindex(u), lastindex(v))
# scores2 = zeros(lastindex(u), lastindex(v))
# p = zeros(lastindex(u), lastindex(v))
# z = rand(gridSize, gridSize)
# for i = 1:gridSize, j = 1:gridSize
#     scores =  softmax(model(getRBFActivations(Centers, betas, hcat(u[i], v[j]))))
#     z[i, j] = scores[1, 1]
# end

########################################
# Draw contour
contour(u, v, (x, y) -> softmax(model(getRBFActivations(Centers, betas, hcat(x, y))))[1])
scatter!(Centers[:, 1], Centers[:, 2], color=:blue, label="Centers")
scatter!(X[y.==1, 1], X[y.==1, 2], color=:red, label="1")
scatter!(X[y.==2, 1], X[y.==2, 2], color=:green, label="2")
savefig("./example.pdf")