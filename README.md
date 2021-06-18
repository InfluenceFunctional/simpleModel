# simpleModel
for instructional purposes
 
By: Michael Kilgour

## Setup
* Setup python environment including:
  * numpy, glob, matplotlib, argparse
  * AND pytorch - get a link from [Here](https://pytorch.org/get-started/locally/). We will not need GPU for this tutorial
    
## Outline
This code does regression using a basic feedforward neural network. It guesses at the value of some function given the inputs, computes a loss (error) based on that guess, and backpropagates this error, updating the parameters of the network to hopefully do better next time.

The code is set-up for users to play with different objective functions, or load existing datasets. One can also play with the size and shape of the network, train an ensemble of networks, and the details of the training protocol, and visualize the results.

The basic object in the code is the simpleNet, which has only one subclass, simpleNet.model, which contains everything of interest here. 