#!/usr/bin/python3.8
# coding=utf-8
"""Script for converting mnist dataset text images to .png images.

Part of bachelor thesis at BUT FIT.

Keyword arguments:
author -- David Hudák
login -- xhudak03
year -- 2022
"""

import numpy as np
from PIL import Image
import argparse

def parse_args():
    """
    Function reads arguments from command line.

    -i file -- should be file with 784 numbers of MNIST image.
    -o namefile.image -- should be file with correct extension as .png
        output file for plot of image.
        
    Returns:
        (input, output) where input is legal file and output name of new file for creation. 
    """
    parser = argparse.ArgumentParser(description='Parser of arguments.')
    parser.add_argument('-i', '--input', dest='input', metavar='i', type=argparse.FileType('r'),
                        help='Name of input text file')
    parser.add_argument('-o', '--output', dest='output', metavar='o', type=str,
                        help='Name of output image file')

    args = parser.parse_args()
    return args.input, args.output

if __name__ == "__main__":
    inp, out = parse_args()
    data = inp.read()
    string_list = np.array(data.split(","))
    int_list = [np.uint8(float(i)) for i in string_list if i != ""]
    shaped = np.reshape(int_list, (28, 28))
    image = Image.fromarray(shaped)
    image.save(out)

