import numpy as np
import pandas as pd
from mtuq.stochastic_sampling.cmaes_utils import array_in_bounds, in_bounds, Repair


def generate_mutants(cmaes_instance):
    """
    Generates all mutants from a Gaussian distribution.
    """
    for i in range(cmaes_instance.lmbda):
        mutant = draw_single_mutant(cmaes_instance)
        cmaes_instance.mutants[:, i] = mutant.flatten()

def draw_single_mutant(cmaes_instance):
    """
    Draws a single mutant from the Gaussian distribution.
    """
    return cmaes_instance.xmean + cmaes_instance.sigma * cmaes_instance.B @ (cmaes_instance.D * np.random.randn(cmaes_instance.n, 1))

def draw_mutants(cmaes_instance):
    """
    Draws mutants from a Gaussian distribution and scatters them across MPI processes.
    """
    if cmaes_instance.rank == 0:
        generate_mutants(cmaes_instance)
        repair_and_redraw_mutants(cmaes_instance)
        scatter_mutants(cmaes_instance)
    else:
        receive_mutants(cmaes_instance)
    
    cmaes_instance.scattered_mutants = cmaes_instance.mutant_slice
    cmaes_instance.counteval += cmaes_instance.lmbda

def repair_and_redraw_mutants(cmaes_instance):
    """
    Applies repair methods and redraws to all mutants, parameter by parameter.
    """
    bounds = [0, 10]  # Define the hardcoded bounds
    redraw_counter = 0
    for param_idx in range(cmaes_instance.n):
        param_values = cmaes_instance.mutants[param_idx, :]
        if array_in_bounds(param_values, bounds[0], bounds[1]):
            continue
        if cmaes_instance._parameters[param_idx].repair is None:
            cmaes_instance.mutants[param_idx, :], was_redrawn = redraw_param_until_valid(cmaes_instance, param_values, bounds)
            if was_redrawn:
                redraw_counter += 1
            print(f'Redrawn {redraw_counter} out-of-bounds mutants for parameter {cmaes_instance._parameters[param_idx].name}')
        else:
            cmaes_instance.mutants[param_idx, :] = apply_repair_to_param(cmaes_instance, param_values, bounds, param_idx)

def redraw_param_until_valid(cmaes_instance, param_values, bounds):
    """
    Redraws the out-of-bounds values of a parameter array until they are within bounds.
    """
    was_redrawn = False
    for i in range(len(param_values)):
        while not in_bounds(param_values[i], bounds[0], bounds[1]):
            param_values[i] = np.random.randn()
            was_redrawn = True
    return param_values, was_redrawn

def apply_repair_to_param(cmaes_instance, param_values, bounds, param_idx):
    """
    Applies a repair method to the full array of parameter values if defined.
    """
    param = cmaes_instance._parameters[param_idx]
    printed_repair = False
    while not array_in_bounds(param_values, bounds[0], bounds[1]):
        if cmaes_instance.verbose_level >= 0 and not printed_repair:
            print(f'Repairing parameter {param.name} using method {param.repair}')
            printed_repair = True
        Repair(param.repair, param_values, cmaes_instance.xmean[param_idx])
    return param_values

def scatter_mutants(cmaes_instance):
    """
    Splits and scatters the mutants across processes.
    """
    cmaes_instance.mutant_lists = np.array_split(cmaes_instance.mutants, cmaes_instance.size, axis=1)
    cmaes_instance.mutant_slice = cmaes_instance.comm.scatter(cmaes_instance.mutant_lists, root=0)

def receive_mutants(cmaes_instance):
    """
    Receives scattered mutants on non-root processes.
    """
    cmaes_instance.mutant_slice = cmaes_instance.comm.scatter(None, root=0)

def gather_mutants(cmaes_instance):
    """
    Gathers mutants from all processes into the root process. It also uses the datalogger to construct the mutants_logger_list.

    Attributes
    ----------
    cmaes_instance.mutants : array
        The gathered and concatenated mutants. This attribute is set to None for non-root processes after gathering.
    cmaes_instance.transformed_mutants : array
        The gathered and concatenated transformed mutants. This attribute is set to None for non-root processes after gathering.
    cmaes_instance.mutants_logger_list : list
        The list to which the datalogger is appended.
    """
    # Printing the mutants on each process, their shapes and types for debugging purposes
    if cmaes_instance.verbose_level >= 2:
        print(cmaes_instance.scattered_mutants, '\n', 'shape is', np.shape(cmaes_instance.scattered_mutants), '\n', 'type is', type(cmaes_instance.scattered_mutants))

    cmaes_instance.mutants = cmaes_instance.comm.gather(cmaes_instance.scattered_mutants, root=0)
    if cmaes_instance.rank == 0:
        cmaes_instance.mutants = np.concatenate(cmaes_instance.mutants, axis=1)
        if cmaes_instance.verbose_level >= 2:
            print(cmaes_instance.mutants, '\n', 'shape is', np.shape(cmaes_instance.mutants), '\n', 'type is', type(cmaes_instance.mutants))  # DEBUG PRINT
    else:
        cmaes_instance.mutants = None

    cmaes_instance.transformed_mutants = cmaes_instance.comm.gather(cmaes_instance.transformed_mutants, root=0)
    if cmaes_instance.rank == 0:
        cmaes_instance.transformed_mutants = np.concatenate(cmaes_instance.transformed_mutants, axis=1)
        if cmaes_instance.verbose_level >= 2:
            print(cmaes_instance.transformed_mutants, '\n', 'shape is', np.shape(cmaes_instance.transformed_mutants), '\n', 'type is', type(cmaes_instance.transformed_mutants))  # DEBUG PRINT
    else:
        cmaes_instance.transformed_mutants = None

    if cmaes_instance.comm.rank == 0:
        current_df = cmaes_instance._datalogger(mean=False)
    # Log the mutants from _datalogger object
    # If cmaes_instance.mutants_logger_list is empty, initialize it with the current DataFrame
        if cmaes_instance.mutants_logger_list.empty:
            cmaes_instance.mutants_logger_list = current_df
        else:
            # Concatenate the current DataFrame to the logger list
            cmaes_instance.mutants_logger_list = pd.concat([cmaes_instance.mutants_logger_list, current_df], ignore_index=True)
