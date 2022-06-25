"""
The supporting class for new strategies implementation.

Author: David Hudak <xhudak03@vutbr.cz>
"""


import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Splitmans:
    def __init__(self, start_index=0, memory_size=0, memory: np.array = np.empty((0,)), layer=0):
        """
        Args:
            start_index                 : starting index for memory-based and alternating strategies 
            memory_size                 : size of memory for all strategies.
            memory                      : starting memory for memory-based strategies
            layer                       : starting layer for semi-hierachical strategy 
        """
        self._index = start_index
        self._memory_size = memory_size
        self._memory = memory.copy()
        if self.memory_size <= 0:
            self._memory_size = 1
        self._layer = layer
        self._structure = np.empty((0,))
        self._list_layers = None

    @property
    def structure(self):
        return self._structure

    @property
    def layer(self):
        return self._layer

    @property
    def index(self):
        return self._index

    @property
    def memory_size(self):
        return self._memory_size

    @property
    def memory(self):
        return self._memory

    def set_structure(self, structure):
        """
        Semi-hierarchical strategy function. Creates list of unique layers 
        and update current layer index if the number of layers decrease.

        Args:
            structure : list of node indices.
        """
        self._structure = structure[-1][:, 0]
        self._list_layers = np.unique(self._structure)
        self._layer %= self._list_layers.size

    def raise_layer(self) -> int:
        """
        Semi-hierarchical strategy function. Increments current layer.
        """
        self._layer = (self.layer + 1) % self._list_layers.size

    def get_current_layer_indices(self):
        """
        Semi-hierarchical strategy function. Returns list of current layer indices
        """
        return np.where(self._structure == self._list_layers[self.layer])[0]

    def raise_index(self):
        """
        Memory-based strategy function. Increments stack index.
        """
        self._index = (self.index + 1) % self.memory_size

    def set_memory(self, memory):
        """
        Memory-based strategy function. Sets new stack.
        """
        self._memory = memory

    def sort_by_layer(self):
        """
        Sorted and reverse sorted memory strategies function. Sorts stack.
        """
        self._memory = self.memory[self.memory[:, 0].argsort()]

    def null_index(self):
        """
        Memory-based and alternating strategies function.
        """
        self._index = 0

    def alternate_index(self):
        """
        Alternating memory strategy function. Alternates index.
        """
        self._index = (self._index + 1) % 2

    def is_odd(self):
        """
        Alternating memory strategy function.
        """
        return self._index % 2 > 0
