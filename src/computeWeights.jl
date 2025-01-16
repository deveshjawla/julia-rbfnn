"""
Compute the optimum weights of the RBF NN using Stochastic Gradient Descent method AdaBelief. Uses CosAnneal for scheduling the learning rate η and automatically saves the best model obtained during the epochs.

Parameters
X       - The training set inputs.
y       - The training set labels.

Keyword Arguements
lambda  - The L2 regularization parameter
n_epochs - Number of Epochs

Returns
optim_theta - The learned weights of the RBF-NN
RBFNN - The RBF-NN architecture
"""
function computeWeights(X::Array{Float32,2}, y::Vector{Int64}, sample_weights, numRBFNeurons, numCats, verbose; lambda=0.0, n_epochs=500)
    if (verbose)
        println("3. Learn output weights.\n")
    end
    nn = Dense(numRBFNeurons, numCats; bias=false)

    y = Flux.onehotbatch(vec(y), 1:numCats)

    opt = OptimiserChain(WeightDecay(lambda), AdaBelief())
    # s = ParameterSchedulers.Stateful(CosAnneal(0.1, 1e-6, 100))
    opt_state = Flux.setup(opt, nn)

    # Train it
    least_loss = Inf32
    last_improvement = 0
    optim_theta = 0
    re = 0

    train_loader = nothing
    sample_weights_loader = nothing
    if size(y, 2) > 1000
        @info "Using Minibatch Training" batchsize = 1000
        train_loader = Flux.DataLoader((X, y), batchsize=1000)
        sample_weights_loader = Flux.DataLoader(sample_weights, batchsize=1000)
    end

    trnlosses = zeros(n_epochs)
    for e in 1:n_epochs
        local loss = 0.0f0

        # global opt_state, nn
        # Flux.adjust!(opt_state, ParameterSchedulers.next!(s))
        # loss, grad = Flux.withgradient(m -> Flux.Losses.logitcrossentropy(m(X), y), nn)
        if size(y, 2) > 1000
            for ((x_, y_), sample_weights_) in zip(train_loader, sample_weights_loader)
                # counter += 1
                # @info "Minibatch no. $(counter) and Epoch no. $(e)"
                local l = 0.0
                l, grad = Flux.withgradient(m -> logitcrossentropyweighted(m(x_), y_, sample_weights_), nn)
                Flux.update!(opt_state, nn, grad[1])
                loss += l / length(train_loader)
            end
        else
            loss, grad = Flux.withgradient(m -> logitcrossentropyweighted(m(X), y, sample_weights), nn)
            trnlosses[e] = loss
            Flux.update!(opt_state, nn, grad[1])
        end
        # if mod(e, 2) == 1
        # 	# Report on train and test, only every 2nd epoch_idx:
        # 	@info "After Epoch $e" loss
        # @warn("After Epoch $e -> $(opt_state.weight.rule)\n $(opt_state.bias.rule)!")
        # end

        if abs(loss) < abs(least_loss)
            # @info("After Epoch $e -> New minimum loss $loss. Saving model weights.\n $(opt_state.weight.rule)")
            optim_theta, re = Flux.destructure(nn)
            least_loss = loss
            last_improvement = e
        end

        if e - last_improvement >= 100
            @warn("After Epoch $e -> We're calling this converged.")
            break
        end

    end
    # scatter(1:n_epochs, trnlosses, width=80, height=30)
    # savefig("./$(nn_arch)_loss.pdf")
    @info "Finished training" last(trnlosses)
    # optim_params, re = Flux.destructure(nn)

    return optim_theta, re
end

function logitcrossentropyweighted(ŷ::AbstractArray, y::AbstractArray, sample_weights::AbstractArray; dims=1)
    if size(ŷ) != size(y)
        error("logitcrossentropyweighted(ŷ, y), sizes of (ŷ, y) are not the same")
    end
    if size(y, 2) != size(sample_weights, 1)
        error("logitcrossentropyweighted(ŷ, y), size y = $(size(y)) and size of sample_weights = $(size(sample_weights)) not the same")
    end

    mean(permutedims(sample_weights) .* -sum((y .* logsoftmax(ŷ; dims=dims)); dims=dims))
end
