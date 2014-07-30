using Clustering
using Plotly

include("computeRBFBetas.jl")
include("rbfActivations.jl")
include("costFunctionRBFN.jl")
include("trainRbf.jl")
include("evaluateRbf.jl")
include("fmincg.jl")

data = readcsv("dataset.csv")
numdims = size(data,2) - 1
X = data[:, 1 : numdims]
y = data[:, numdims + 1]
@printf("\n ========================= \n")
@printf("\n X \n")
show(size(X))
@printf("\n")
@printf("\n y \n")
show(size(y))
@printf("\n")

gridSize = 100
u = linspace(-2.0, 2.0, gridSize)
v = linspace(-2.0, 2.0, gridSize)

#count = 100
#t = zeros(count, 1)
#for timingIndex = 1:count
#	tic();
	(Centers, betas, Theta) = trainRbf(X, y, 5, false)
#	t[timingIndex] = toc();
#end

#@printf("\n Timings: \n")
#show(t)
#@printf("\n")
	########################################
	# Let's test

	# Define a grid over which to evaluate the RBFN.

	# We'll store the scores for each category as well as the 'prediction' for
	# each point on the grid.
	scores1 = zeros(length(u), length(v))
	scores2 = zeros(length(u), length(v))
	p = zeros(length(u), length(v))

	# Evaluate the RBFNN over the grid.
	z = rand(gridSize, gridSize)
	for i = 1:gridSize, j = 1:gridSize
	    scores = evaluateRbf(Centers, betas, Theta, hcat(u[i], v[j]))
	    z[i,j] = scores[1,1]
	end
	#t[timingIndex] = toc();
#end

#@printf("\n Timings: \n")
#show(t)
#@printf("\n")


########################################
# Draw contour

#data = [["z" => z, "x" => u, "y" => v, "type" => "contour"]]
#Plotly.signin("secret", "secret")
#response = Plotly.plot([data], ["filename" => "simple-contour", "fileopt" => "overwrite"])
#plot_url = response["url"]
#@printf("\n ========================= \n")
#@printf("\n URL \n")
#show(plot_url)
#@printf("\n")
