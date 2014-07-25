using Clustering

type Rbfnn
        indim::Int64
        numCenters::Int64
        outdim::Int64
        Centers::Array{Float64,2}
        betas::Array{Float64,1}
        W::Array{Float64,1}
        G::Array{Float64,2}
end

#=rbfnn(indim::Uint8, numCenters::Uint8, outdim::Uint8) = rbfnn (
#    indim,
#    numCenters,
#    outdim,
#    (rand(indim, numCenters) * 2) - 1,
#    8,
#    rand(numCenters, outdim))=#

function _basisfunc(c::Array{Float64, 2}, d::Array{Float64, 2}, b::Float64)
    #@printf("\n c \n")
    #show(c)
    #@printf("\n d \n")
    #show(d)
    #@printf("\n b \n")
    #show(b)
    #@printf("\n")
    return exp(-b * (norm(c-d)*norm(c-d)))
end

function _calcAct(X::Matrix{Float64}, centers::Matrix{Float64}, beta::Vector{Float64})
    # calculate activations of RBFs
    numCenters = size(centers, 1)
    G = zeros((size(X)[1], numCenters))
    #for ci, c in enumerate(centers) {
    #@printf("betas \n")
    for ci = 1:size(centers)[1]
        for xi = 1:size(X)[1]
            c = centers[:,ci]'
            d = X[xi, :]
            #@printf("\n c \n")
            #show(c)
            #@printf("\n d \n")
            #show(d)
            #@printf("\n beta \n")
            #show(beta)
            #@printf(" \n calling basisfunc... \n")
            G[xi,ci] = _basisfunc(c', d', beta[ci])
        end
    end
    return G
end

#function [Centers, betas, Theta] = trainRBFN(X_train, y_train, centersPerCategory, verbose)
function train(X::Array{Float64, 2}, y::Array{Float64, 1}, centersPerCategory::Int64)
    verbose = true

    # X: matrix of dimensions n x indim
    # y: column vector of dimension n x 1

    # choose random center vectors from training set
    #rnd_idx = rand random.permutation(X.shape[0])[:self.numCenters]

    # Get the number of unique categories in the dataset.
    numCats = size(unique(y), 1)

    # Set 'm' to the number of data points.
    m = size(X, 1)

    # Ensure category values are non-zero and continuous.
    # This allows the index of the output node to equal its category (e.g.,
    # the first output node is category 1).
    if (any(y .== 0) || any(y .> numCats))
        error("Category values must be non-zero and continuous.")
    end

    # ================================================
    #       Select RBF Centers and Parameters
    # ================================================
    # Here I am selecting the cluster centers using k-Means clustering.
    # I've chosen to separate the data by category and cluster each category
    # separately, though I've read that this step is often done over the full
    # unlabeled dataset. I haven't compared the accuracy of the two approaches.

    if (verbose)
        @printf("1. Selecting centers through k-Means.\n")
    end

    Centers = zeros(size(X,2), 0)
    betas = zeros(centersPerCategory*2);

    # For each of the categories...
    for c = 1:numCats

        if (verbose)
            @printf("  Category %d centers...\n", c);
        end

        # Select the training vectors for category 'c'.
        Xc = X[y .== c, :]

        # ================================
        #      Find cluster centers
        # ================================

        # Run k-means clustering
        @printf("  Running kmeans with %d centers...\n", centersPerCategory);
        result = kmeans(Xc', centersPerCategory)
        #show(typeof(result))
        #@printf("\n")
        #show(size(Centers))
        #@printf("\n")
        Centroids_c = result.centers
        #show(typeof(Centroids_c))
        #@printf("\n")
        #show(size(Centroids_c))
        #@printf("\n")

        Centers = [Centers Centroids_c];
    end

    #print "center", centers
    # calculate activations of RBFs

    #@printf("X \n")
    #show(size(X))
    @printf("\n Centers \n")
    show(Centers)
    @printf("\n betas \n")
    show(betas)
    @printf("\n")

    G = _calcAct(X, Centers, betas)
    @printf("\n G \n")
    show(G)
    @printf("\n")

    @printf("\n y \n")
    show(y)
    @printf("\n")

    # calculate output weights (pseudoinverse)
    pinvG = pinv(G)

    @printf("\n pinvG \n")
    show(pinvG)
    @printf("\n")

    #W = dot(pinvG, y)
    W = pinvG * y
    @printf("\n W \n")
    show(W)
    @printf("\n")

    return Rbfnn(size(X, 2), #indim =
            centersPerCategory * 2, #numCenters =
            numCats, #outdim =
            Centers,
            betas,
            W,
            G)
end

function test(nn::Rbfnn, X::Matrix{Float64})
    # X: matrix of dimensions n x indim

    @printf("\n running test \n")

    G = _calcAct(X, nn.Centers, nn.betas)

    @printf("\n G \n")
    show(G)
    @printf("\n")

    @printf("\n W \n")
    show(nn.W)
    @printf("\n")

    Y = G * nn.W

    @printf("\n Y \n")
    show(Y)
    @printf("\n")

    #Y = dot(nn.G, nn.W)
    return Y
end

function meshgrid{T}(vx::AbstractVector{T}, vy::AbstractVector{T})
    m, n = length(vx), length(vy)
    vx = reshape(vx, 1, n)
    vy = reshape(vy, m, 1)
    (repmat(vx, m, 1), repmat(vy, 1, n))
end


#data = readcsv("dataset.csv")
data = [ 0. 0. 1.; 0.1 0.1 1.; 1. 1. 1.; 0.9 0.9 1; 0. 1. 2.; 0.1 1.0 2.; 1. 0. 2.; 1.0 0.1 2.0]
numdims = size(data,2) - 1
X = data[:, 1 : numdims]
y = data[:, numdims + 1]
@printf("\n ========================= \n")
@printf("\n X \n")
show(X)
@printf("\n")
@printf("\n y \n")
show(y)
@printf("\n")

rbfNet = train(X, y, 2)
yTest = test(rbfNet, X)

@printf("\n ========================= \n")
@printf("\n yTest \n")
show(yTest)

@printf("\n ========================= \n")
#show(y .== yTest)
