# Experimental results of VeriNet and extensions

This folder contains all the results with which I was working in my bachelor's thesis in a non-aggregated form. These are the default logs created by VeriNet.

## Structure
Each folder contains the results of a specific strategy. Details of how they work can be found in the design chapter of my bachelor's thesis.

The strategies are:

* Default VeriNet in the default_results folder.
* Semi-hierarchical strategy named as best_by_layer. This strategy has achieved the best results compared to default results.
* Memory sorted strategies with mandatory parameter *N*, which describes size of used memory.
  * Simple memory strategy
  * Sorted memory strategy
  * Reversed sorted memory strategy (significantly worst results)
* Alternating memory strategy

## Experimental seting

The results of the experiments strongly depend on the HW used. Specifically, these results come from a computer with components:

  * CPU: Ryzen 3 1300X
  * RAM: 16 GB DDR4
  * GPU: GTX 1660 with 6 GB GDDR5

## Source of NN models

All networks were taken from the default VeriNet (see https://github.com/vas-group-imperial/VeriNet-OpenSource) and Marabou (see https://github.com/NeuralNetworkVerification/Marabou).
