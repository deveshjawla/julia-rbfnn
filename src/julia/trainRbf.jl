function trainRbf(X_train::Array{Float64, 2}, y_train::Array{Float64, 1}, centersPerCategory::Int64, verbose::Bool)
#(X_train, y_train, centersPerCategory, verbose)#
# TRAINRBFN Builds an RBF Network from the provided training set.
#   [Centers, betas, Theta] = trainRBFN(X_train, y_train, centersPerCategory, verbose)
#    
#   There are three main steps to the training process:
#     1. Prototype selection through k-means clustering.
#     2. Calculation of beta coefficient (which controls the width of the 
#        RBF neuron activation function) for each RBF neuron.
#     3. Training of output weights for each category using gradient descent.
#
#   Parameters
#     X_train  - The training vectors, one per row
#     y_train  - The category values for the corresponding training vector.
#                Category values should be continuous starting from 1. (e.g.,
#                1, 2, 3, ...)
#     centersPerCategory - How many RBF centers to select per category. k-Means
#                          requires that you specify 'k', the number of 
#                          clusters to look for.
#     verbose  - Whether to print out messages about the training status.
#
#   Returns
#     Centers  - The prototype vectors stored in the RBF neurons.
#     betas    - The beta coefficient for each coressponding RBF neuron.
#     Theta    - The weights for the output layer. There is one row per neuron
#                and one column per output node / category.

# $Author: ChrisMcCormick $    $Date: 2014/04/08 22:00:00 $    $Revision: 1.2 $

    # Get the number of unique categories in the dataset.
    numCats = size(unique(y_train), 1)
    
    # Set 'm' to the number of data points.
    m = size(X_train, 1)
    
    # Ensure category values are non-zero and continuous.
    # This allows the index of the output node to equal its category (e.g.,
    # the first output node is category 1).
    if (any(y_train .== 0) || any(y_train .> numCats))
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
        println("1. Selecting centers through k-Means.\n")
    end    
    
    numDims = size(X_train, 2)
    Centers = rand(0, numDims)#[];
    betas = rand(0, 1)#[]    
    
    # For each of the categories...
    for (c = 1 : numCats)

        if (verbose)
	    @printf("  Category %d centers...\n", c)
        end
        
        # Select the training vectors for category 'c'.
        Xc = X_train[(y_train .== c), :]

        # ================================
        #      Find cluster centers
        # ================================
        
        # Pick the first 'centersPerCategory' samples to use as the initial centers.
        init_Centroids = Xc[1:centersPerCategory, :]

        # Run k-means clustering, with at most 100 iterations.        
	@printf("  Running kmeans with %d centers...\n", centersPerCategory)
        result = kmeans(Xc', centersPerCategory)
        Centroids_c = result.centers'
	memberships_c = result.assignments    
        
        # Remove any empty clusters.
        #toRemove = [];
        # 
        # For each of the centroids...
        #for (i = 1 : size(Centroids_c, 1))
        #    # If this centroid has no members, mark it for removal.
        #    if (sum(memberships_c .== i) == 0)        
        #        toRemove = [toRemove; i];
        #    end
        #end
        #
        # If there were empty clusters...
        #if (~isempty(toRemove))
        #    # Remove the centroids of the empty clusters.
        #    Centroids_c(toRemove, :) = [];
        #    
        #    # Reassign the memberships (index values will have changed).
        #    memberships_c = findClosestCentroids(Xc, Centroids_c);
        #end
        
        # ================================
        #    Compute Beta Coefficients
        # ================================
        if (verbose)
	    @printf("  Category %d betas...\n", c)
        end

        # Compute betas for all the clusters.
        betas_c = computeRBFBetas(Xc, Centroids_c, memberships_c)
        
        # Add the centroids and their beta values to the network.

	#@printf("\n betas \n")
	#show(betas)
	#@printf("\n betas_c \n")
	#show(betas_c)
	#@printf("\n")        
	#@printf("\n Centers \n")
	#show(Centers)
	#@printf("\n Centroids_c \n")
	#show(Centroids_c)
	#@printf("\n")        
	#Centers = [Centers; Centroids_c]
	Centers = vcat(Centers, Centroids_c)
        betas = vcat(betas, betas_c)
    end

    #Centers = [    -0.4528   -0.7259;
#	    1.1624   -0.2929;
#	    0.8586   -1.3844;
#	   -0.8102    0.3913;
#	    0.4574    0.3437;
#	   -0.2140    1.2991;
#	   -1.4650    0.7938;
#	   -1.3791   -0.8765;
#	    1.0840    1.0717;
#	    0.8385   -1.1455]
#
#    betas =
#    [2.2308;
#    3.7955;
#    3.0337;
#    3.3659;
#    3.3811;
#    2.3574;
#    2.7513;
#    2.3913;
#    2.4260;
#    0.8062]'
#    betas = betas'

    # Get the final number of RBF neurons.
    numRBFNeurons = size(Centers, 1)
    
    # ===================================
    #        Train Output Weights
    # ===================================

    # ==========================================================
    #       Compute RBF Activations Over The Training Set
    # ===========================================================
    if (verbose)
        print("\n2. Calculate RBF neuron activations over full training set.\n")
    end

    # First, compute the RBF neuron activations for all training examples.

    # The X_activ matrix stores the RBF neuron activation values for each training 
    # example: one row per training example and one column per RBF neuron.
    X_activ = zeros(m, numRBFNeurons)

    #@printf("\n X_activ (type,size) \n")	  
    #show(typeof(X_activ));
    #show(size(X_activ));	     	
    #@printf("\n ") 

    #@printf("\n m \n")	  
    #show(typeof(m));
    #show(m);	     	
    #@printf("\n ")

    # For each training example...
    for (i = 1 : m)
       
        input = X_train[i, :]
       
	#@printf("\n input ==-- \n")
	#show(input);	     	
	#@printf("\n ")

       # Get the activation for all RBF neurons for this input.
        z = getRBFActivations(Centers, betas, input)
       
	#@printf("\n calcualted Z \n")
	#show(z);	     	
	#@printf("\n ")

        # Store the activation values 'z' for training example 'i'.
        X_activ[i, :] = z #z'
    end

    # Add a column of 1s for the bias term.
    biasColumn = ones(m, 1)
    X_activ = hcat(biasColumn, X_activ)

    #@printf("\n X_activ \n")
    #show(X_activ)
    #@printf("\n") 

    # =============================================
    #        Perform Gradient Descent
    # =============================================

    if (verbose)
        println("3. Learn output weights.\n")
    end

    # Don't perform any regularization (lambda = 0).
    lambda = 1.0

    # Perform 200 iterations of gradient descent.
    maxIter = 200;

    # Initialize theta to 0 for all RBF neurons. Add another zero to theta for 
    # the bias term.
    initial_theta = zeros(numRBFNeurons + 1, 1)

    # Set the options for fmincg.
    #options = optimset('GradObj', 'on', 'MaxIter', maxIter);

    # Create a matrix to hold all of the output weights.
    # There is one column per category / output neuron.
    Theta = zeros(numRBFNeurons + 1, numCats)

    # For each category...
    for (c = 1 : numCats)

        # Make the y values binary--1 for category 'c' and 0 for all other categories.
        #y_c = (y_train .== c)
	y_c = map(b -> convert(Int64, b), (y_train .== c))	
            
        #  ================================
        #      Manual Gradient Descent
        #  ================================
        #  You can alternatively use the below code to run gradient descent manually,
        #  and plot the cost over each iteration.
        #
        #  # Set the learning rate
          alpha = 1
          J_history = zeros(maxIter, 1)
          Theta[:, c] = initial_theta
          for (i = 1 : maxIter)
              # Compute the cost and gradients.    
              (J, grad) = costFunctionRBFN(Theta[:, c]'', X_activ, y_c, lambda)
   	      #@printf("  %4i | Cost: %4.6e \r\n", i, J)	

              # Record the cost for this iteration.
              J_history[i] = J
              # Simultaneously update all values of theta.
              #Theta(:, c) -= grad * alpha
	      Theta[:, c] = Theta[:, c] .- grad * alpha
          end
        #  
        #  # Plot the cost
        #  figure(c);
        #  hold off;
        #  plot(J_history);
        #  title('Cost At Each Iteration');

    end
    
    Centers, betas, Theta	

end

