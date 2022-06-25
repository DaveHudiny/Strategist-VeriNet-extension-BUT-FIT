# VeriNet extension

This folder contains strategist extension created while working on bachelor thesis at BUT FIT. We have not removed any content created by original authors. 

After extension text you can find original text readme.md created by VeriNet authors.


## Codes and models

We created extension for original VeriNet project from: https://github.com/vas-group-imperial/VeriNet-OpenSource

We took few NNet models from Marabou repository: https://github.com/NeuralNetworkVerification/Marabou

The extension contains new files: 
$ ./src/algorithm/splitmans.py 
$ ./src/algorithm/strategist.py 
$ ./script.sh.

We also created few benchmark scripts in this folder, which are strongly inspired by default benchmarks.

## Usage

You can use this code in same was as original project. However, you can use any benchmark by:

$ scriph.sh <benchmark_name>

You can change strategy in file ./src/algorithm/verinet_worker.py in function _branch() at line 343.
Simply uncomment the strategy you want to use and comment the strategy you do not want.

## Extension authors

David Hudák: xhudak03@vutbr.cz
Milan Češka (bachelor thesis supervisor)

# VeriNet Open Source

This repository contains the open source version of VeriNet VeriNet toolkit for local robustness verification of feed-forward neural networks.  

# Important Notice

This version of VeriNet is outdated and should **not be used for benchmarking**; 
for benchmarking purposes see https://github.com/vas-group-imperial/VeriNet. 

## Installation

### Pipenv

Most dependencies can be installed via pipenv:

$ cd <your_verinet_path>/VeriNet/src
$ pipenv install

(If pipenv install fails, try pipenv install torch==1.1.0 and rerun pipenv install)

### Gurobi

VeriNet uses the Gurobi LP-solver which has a free academic license.  

1) Go to https://www.gurobi.com, download Gurobi and get the license.  
2) Follow the install instructions from http://abelsiqueira.github.io/blog/installing-gurobi-7-on-linux/  
3) Activate pipenv by cd'ing into your VeriNet/src and typing $pipenv shell
4) Find your python path by typing $which python
5) cd into your Gurobi installation and run $<your python path> setup.py install

### Numpy with OpenBLAS

For optimal performance, we recommend compiling Numpy from source with OpenBLAS.

Install instruction can be found at: 
https://hunseblog.wordpress.com/2014/09/15/installing-numpy-and-openblas/  

## IMPORTANT NOTES:

### Cuda devices

Pythons multiprocessing does not work well with Cuda. To avoid any problems 
we hide all Cuda devices using environment variables. This is done in 
VeriNet/src/.env which is run every time you enter pipenv shell. 
If you have a Cuda device and do not use the pipenv environment, you have to 
manually enter:

$export CUDA_DEVICE_ORDER="PCI_BUS_ID"  
$export CUDA_VISIBLE_DEVICES=""

### OpenBLAS threads

Since our algorithm has a highly efficient parallel implementation, OpenBLAS 
should be limited to 1 thread. This can not be done in runtime after Numpy is 
loaded, so we use an environment variable instead. 
The variable is automatically set when using pipenv and can be found in 
VeriNet/src/.env. If you do not use the pipenv environment, this has to be done 
manually with the command:

$export OMP_NUM_THREADS=1

## Usage

All of the experiments used in the paper can be run with the scripts in
VeriNet/src/scripts. The file VeriNet/examples/examples.py contains several
examples of how to run the algorithm using networks loaded from the nnet
format and custom networks.  More information about the nnet format can be found
in VeriNet/data/models_nnet.

## Authors

Patrick Henriksen: ph818@ic.ac.uk  
Alessio Lomuscio


