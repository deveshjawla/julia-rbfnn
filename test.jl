using RBFNN
using Plots
# using CSV, DataFrames
using Flux: onecold
using Random
using StatsBase: countmap

"""
Generates three Gaussian(0,1) Distributed blobs around (0,0), (2,2) and (-2,2)
and labels for each blob

Input: n = the number of points needed per blob(class)
Output: Tuple{Matrix, Matrix} = Input features X (coordinates in 2d space), Onehot labels of the blobs
"""
function gen_3_clusters(n; cluster_centers=[[0, 0], [2, 2], [-2, 2]])
    x1 = randn(Xoshiro(1234), Float32, 2, n) .+ cluster_centers[1]
    x2 = randn(Xoshiro(1234), Float32, 2, n) .+ cluster_centers[2]
    x3 = randn(Xoshiro(1234), Float32, 2, n) .+ cluster_centers[3]
    y1 = vcat(ones(Float32, n), zeros(Float32, 2 * n))
    y2 = vcat(zeros(Float32, n), ones(Float32, n), zeros(Float32, n))
    y3 = vcat(zeros(Float32, n), zeros(Float32, n), ones(Float32, n))
    return hcat(x1, x2, x3), permutedims(hcat(y1, y2, y3))
end

# Generate data
n = 200
X, y = gen_3_clusters(n)
X = permutedims(X)
y = onecold(y)

gridSize = 100
u = range(-10.0f0, 10.0f0, length=gridSize)
v = range(-10.0f0, 10.0f0, length=gridSize)

balance_of_training_data = countmap(Int.(y))

n_output = length(unique(y))

sample_weights = similar(y, Float32)

nos_training = lastindex(y)

for i ∈ 1:nos_training
    sample_weights[i] = nos_training / balance_of_training_data[y[i]]
end
sample_weights ./= n_output


(Centers, betas, Theta, nn) = trainRbf(X, y, sample_weights, 10, true, 3)

# # ================================================
# #       Evaluate the RBFNN over the grid.
# # ================================================
# # Define a grid over which to evaluate the RBFN.
# # We'll store the scores for each category as well as the 'prediction' for
# # each point on the grid.
########################################
# Draw contour
heatmap(u, v, (x, y) -> testRbf(Centers, betas, Theta, nn, hcat(x, y))[3])
scatter!(Centers[:, 1], Centers[:, 2], color=:blue, label="Centers")
scatter!(X[y.==1, 1], X[y.==1, 2], color=:red, label="1")
scatter!(X[y.==2, 1], X[y.==2, 2], color=:green, label="2")
scatter!(X[y.==3, 1], X[y.==3, 2], color=:blue, label="2")
savefig("./example.pdf")