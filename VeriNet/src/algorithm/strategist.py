"""
The main code for new strategies implementation.

Functions:
    largest_error_split_node()
    largest_error_by_layer()
    largest_error_split_alternate()

are strongly inspired by largest_error_split_node() from
    src.algorithm.esip
    created by Patrick Henriksen

Author: David Hudak <xhudak03@vutbr.cz>
"""


import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.algorithm.esip import ESIP
from src.algorithm.verification_objectives import VerificationObjective
from src.algorithm.splitmans import Splitmans


class Strategist:
    """ 
    Static class, where we define our strategies.
    """

    def load_new_set(bounds: ESIP, verification_objective: VerificationObjective, splitmans: Splitmans):
        """ 
        Function implements load of a list of nodes by memory-based strategies.
        Args:
            bounds          : Neural network representation.

            Splitmans       : Structure of the current strategy data.

            verification_objective: The verification objective

        Returns:
            list[splitmans.memory_size] of the nodes (layer_num, node_num) with largest error effect on the output
        """
        refine_output_weights = verification_objective.output_refinement_weights(
            bounds)
        return Strategist.largest_error_split_node(bounds, splitmans.memory_size, output_weights=refine_output_weights)

    def pop_first(bounds: ESIP, verification_objective: VerificationObjective, splitmans: Splitmans):
        """ 
        Function implements simple memory strategy.
        Args:
            bounds          : Neural network representation.

            Splitmans       : Structure of the current strategy data.

            verification_objective: The verification objective

        Returns:
            (layer_num, node_num) of the node with largest error effect via simple memory strategy.
        """
        if splitmans.memory.size == 0:
            splitmans.set_memory(Strategist.load_new_set(
                bounds, verification_objective, splitmans))

        split = splitmans.memory[splitmans.index]
        splitmans.raise_index()
        if splitmans.index == 0 or splitmans.index >= splitmans.memory.size:
            splitmans.set_memory(Strategist.load_new_set(
                bounds, verification_objective, splitmans))
            splitmans.null_index()

        return split

    def pop_first_layer(bounds: ESIP, verification_objective: VerificationObjective, splitmans: Splitmans):
        """ 
        Function implements sorted memory strategy.
        Args:
            bounds          : Neural network representation.

            Splitmans       : Structure of the current strategy data.

            verification_objective: The verification objective

        Returns:
            (layer_num, node_num) of the node with largest error effect on the output via sorted memory strategy.
        """
        if splitmans.memory.size == 0:
            splitmans.set_memory(Strategist.load_new_set(
                bounds, verification_objective, splitmans))
            splitmans.sort_by_layer()

        split = splitmans.memory[splitmans.index]
        splitmans.raise_index()
        if splitmans.index == 0 or splitmans.index >= splitmans.memory.size:
            splitmans.set_memory(Strategist.load_new_set(
                bounds, verification_objective, splitmans))
            splitmans.null_index()
            splitmans.sort_by_layer()

        return split

    def pop_last_layer(bounds: ESIP, verification_objective: VerificationObjective, splitmans: Splitmans):
        """ 
        Function implements reverse-sorted memory strategy.
        Args:
            bounds          : Neural network representation.

            Splitmans       : Structure of the current strategy data.

            verification_objective: The verification objective

        Returns:
            (layer_num, node_num) of the node with largest error effect on the output via reverse-sorted memory strategy.
        """
        if splitmans.memory.size == 0:
            splitmans.set_memory(Strategist.load_new_set(
                bounds, verification_objective, splitmans))
            splitmans.sort_by_layer()

        split = splitmans.memory[-(1+splitmans.index)]
        splitmans.raise_index()
        if splitmans.index == 0 or splitmans.index >= splitmans.memory.size:
            splitmans.set_memory(Strategist.load_new_set(
                bounds, verification_objective, splitmans))
            splitmans.null_index()
            splitmans.sort_by_layer()

        return split

    def load_set_by_layer(bounds: ESIP, verification_objective: VerificationObjective, splitmans: Splitmans):
        """ 
        Function implements load of a node by semi-hierarchical strategy.
        Args:
            bounds          : Neural network representation.

            Splitmans       : Structure of the current strategy data.

            verification_objective: The verification objective

        Returns:
            (layer_num, node_num) of the node with largest error effect on the output from splitmans layer.
        """
        refine_output_weights = verification_objective.output_refinement_weights(
            bounds)
        return Strategist.largest_error_by_layer(bounds, splitmans, output_weights=refine_output_weights)

    def get_best_by_layer(bounds: ESIP, verification_objective: VerificationObjective, splitmans: Splitmans):
        """ 
        Function implements semi-hierarchical strategy.
        Args:
            bounds          : Neural network representation.

            Splitmans       : Structure of the current strategy data.

            verification_objective: The verification objective

        Returns:
            (layer_num, node_num) of the node with largest error effect on the output via sorted memory strategy.
        """
        splitmans.set_structure(bounds._error_matrix_to_node_indices)
        split = Strategist.load_set_by_layer(
            bounds, verification_objective, splitmans)
        return split

    def load_by_alternation(bounds: ESIP, verification_objective: VerificationObjective, splitmans: Splitmans):
        """ 
        Function implements load of a node by alternating heuristic strategy.
        Args:
            bounds          : Neural network representation.

            Splitmans       : Structure of the current strategy data.

            verification_objective: The verification objective

        Returns:
            (layer_num, node_num) of the node with largest error effect on the output by alternating heuristic.
        """
        refine_output_weights = verification_objective.output_refinement_weights(
            bounds)
        return Strategist.largest_error_split_alternate(bounds, splitmans, output_weights=refine_output_weights)

    def get_alternate_weights(bounds: ESIP, verification_objective: VerificationObjective, splitmans: Splitmans):
        split = Strategist.load_by_alternation(
            bounds, verification_objective, splitmans)
        splitmans.alternate_index()
        if not split.any():
            return Strategist.load_by_alternation(bounds, verification_objective, splitmans)
        return split

    def largest_error_split_alternate(bounds, splitmans: Splitmans, output_weights: np.array = None):
        """
        Returns the node with the largest weighted error effect on the output by alternating heuristic.

        The error from overestimation is calculated for each output node with respect to each hidden node.
        This value is weighted using the given output_weights and the index of the node with largest effect on the
        output is returned.

        Args:
            bounds          : Neural network representation

            Splitmans       : Structure of the current strategy data.

            output_weights  : A Nx2 array with the weights for the lower bounds in column 1 and the upper bounds
                              in column 2. All weights should be >= 0.
        Returns:
              (layer_num, node_num) of the node with largest error effect on the output
        """
        if bounds._error_matrix[-1].shape[1] == 0:
            return None

        output_weights = np.ones(
            (bounds.layer_sizes[-1], 2)) if output_weights is None else output_weights
        output_weights[output_weights <= 0] = 0.01

        if splitmans.is_odd():
            err_matrix_alt = bounds._error_matrix[-1].copy()
            err_matrix_alt[err_matrix_alt > 0] = 0
            err_matrix_alt = - err_matrix_alt * output_weights[:, 0:1]
        else:
            err_matrix_alt = bounds._error_matrix[-1].copy()
            err_matrix_alt[err_matrix_alt < 0] = 0
            err_matrix_alt = err_matrix_alt * output_weights[:, 1:2]

        weighted_error = (err_matrix_alt).sum(axis=0)
        max_err_idx = weighted_error.argsort()[-splitmans.memory_size:][::-1]

        max_err_idx = np.argmax(weighted_error)
        if weighted_error[max_err_idx] <= 0:
            return None
        else:
            return bounds._error_matrix_to_node_indices[-1][max_err_idx]

    def largest_error_by_layer(bounds, splitmans: Splitmans, output_weights: np.array = None):
        """
        Returns the node with the largest weighted error effect on the output at layer by semi-hierarchical strategy. 

        The error from overestimation is calculated for each output node with respect to each hidden node.
        This value is weighted using the given output_weights and the index of the node with largest effect on the
        output is returned.

        Args:
            bounds          : Neural network representation

            Splitmans       : Structure of the current strategy data.

            output_weights  : A Nx2 array with the weights for the lower bounds in column 1 and the upper bounds
                              in column 2. All weights should be >= 0.
        Returns:
              (layer_num, node_num) of the node with largest error effect on the output
        """
        if bounds._error_matrix[-1].shape[1] == 0:
            return None
        indices = splitmans.get_current_layer_indices()
        splitmans.raise_layer()

        output_weights = np.ones(
            (bounds.layer_sizes[-1], 2)) if output_weights is None else output_weights
        output_weights[output_weights <= 0] = 0.01

        err_matrix_neg = bounds._error_matrix[-1].copy()
        err_matrix_neg[err_matrix_neg > 0] = 0

        err_matrix_pos = bounds._error_matrix[-1].copy()
        err_matrix_pos[err_matrix_pos < 0] = 0

        err_matrix_neg = - err_matrix_neg * output_weights[:, 0:1]
        err_matrix_pos = err_matrix_pos * output_weights[:, 1:2]

        weighted_error = (err_matrix_neg + err_matrix_pos).sum(axis=0)
        weighted_error[indices] *= 10
        max_err_idx = weighted_error.argsort()[-splitmans.memory_size:][::-1]

        max_err_idx = np.argmax(weighted_error)
        if weighted_error[max_err_idx] <= 0:
            return None
        else:
            return bounds._error_matrix_to_node_indices[-1][max_err_idx]

    def largest_error_split_node(bounds, memory, output_weights: np.array = None):
        """
        Returns the list of nodes with the largest weighted error effect on the output for memory-based strategies.

        The error from overestimation is calculated for each output node with respect to each hidden node.
        This value is weighted using the given output_weights and the index of the node with largest effect on the
        output is returned.

        Args:
            bounds          : Neural network representation

            memory          : Size of returned memory

            output_weights  : A Nx2 array with the weights for the lower bounds in column 1 and the upper bounds
                              in column 2. All weights should be >= 0.
        Returns:
              list[splitmans.memory_size] of the nodes (layer_num, node_num) with largest error effect on the output
        """

        if bounds._error_matrix[-1].shape[1] == 0:
            return None

        output_weights = np.ones(
            (bounds.layer_sizes[-1], 2)) if output_weights is None else output_weights
        output_weights[output_weights <= 0] = 0.01

        err_matrix_neg = bounds._error_matrix[-1].copy()
        err_matrix_neg[err_matrix_neg > 0] = 0
        err_matrix_pos = bounds._error_matrix[-1].copy()
        err_matrix_pos[err_matrix_pos < 0] = 0

        err_matrix_neg = - err_matrix_neg * output_weights[:, 0:1]
        err_matrix_pos = err_matrix_pos * output_weights[:, 1:2]

        weighted_error = (err_matrix_neg + err_matrix_pos).sum(axis=0)
        max_err_idx = weighted_error.argsort()[-memory:][::-1]

        if weighted_error[max_err_idx[-1]] <= 0:
            return None
        else:
            return bounds._error_matrix_to_node_indices[-1][max_err_idx]
