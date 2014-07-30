julia-rbfnn
===========

Julia package for Radial Basis Function Neural Networks (RBFNN)

This is translation of Chris McCormick RBFNN (http://chrisjmccormick.wordpress.com/2013/08/16/rbf-network-matlab-code/) Octave/Matlab code https://dl.dropboxusercontent.com/u/94180423/RBFN_Example_v2014_04_08.zip  
Julia version for now is not optimized, but runs 60% faster.

For allocation of RBF neurons kMeans package is used, for allocation of beta parameters (neuron widths) computeRBFBetas.jl routine is used.
Once al RBF neurons are found, their parameters remain fixed and Gradient Descent is used to find weights between RBF neurons and output neurons.
Amount of output neurons is determined based on count of categories/labels (they should be represented in separate integer vector). Labels should start from 1 and monotonically increase. 

Please refer to test.jl file to see how stuff works. It uses Plot.ly to do graphing - so make sure you provide your credentials for logging into plot'ly (see the very end of file).

Two main functions are trainRbf and evaluateRbf - they build model and utilize it respectively. 
