"""
Small script for benchmarking the MNIST 1024 network

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os

from src.scripts.benchmark import run_benchmark
from src.data_loader.input_data_loader import load_mnist_human_readable

if __name__ == "__main__":
	# 1, 2, 5, 10, 15
    epsilons = [ 1, 2, 5 ]
    timeout = 900

    num_images = 100
    img_dir: str = f"./data/mnist_neurify/test_images_100/"

    if not os.path.isdir("./benchmark_results"):
        os.mkdir("./benchmark_results")

    run_benchmark(images=load_mnist_human_readable(img_dir, list(range(num_images))).reshape(num_images, -1),
                  epsilons=epsilons,
                  timeout=timeout,
                  conv=False,
                  model_path="./data/models_nnet/neurify/mnist512.nnet",
                  result_path=f"./benchmark_results/mnist_{num_images}_imgs_1024_relu.txt")
