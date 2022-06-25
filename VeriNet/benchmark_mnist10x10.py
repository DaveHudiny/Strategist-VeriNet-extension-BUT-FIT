"""
Small script for benchmarking the MNIST 10x10 network

Original author: Patrick Henriksen <patrick@henriksen.as>
Taken by: David Hudák
"""


import os

from src.scripts.benchmark import run_benchmark
from src.data_loader.input_data_loader import load_mnist_human_readable
import numpy as np

if __name__ == "__main__":
    # 1, 2, 5, 10, 15 - missing epsilons
    epsilons = [ 10, 15 ] 
    timeout = 900

    num_images = 100
    img_dir: str = f"./data/mnist_neurify/test_images_100/"

    if not os.path.isdir("./benchmark_results"):
        os.mkdir("./benchmark_results")

    run_benchmark(images=load_mnist_human_readable(img_dir, list(range(num_images))).reshape(num_images, -1),
                  epsilons=epsilons,
                  timeout=timeout,
                  conv=False,
                  model_path="./data/marabou/mnist10x10.nnet",
                  result_path=f"./benchmark_results/mnist_{num_images}_imgs_10x10_relu.txt",
                  memory=1)
