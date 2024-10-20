"""
This module provides classes and functions for stochastic sampling in the context of seismic inversion.
It includes implementations of the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) algorithm
and variable encoders for moment tensor and force source inversion.

Classes:
    - CMA_ES: Implementation of the CMA-ES algorithm for seismic inversion.
    - CMAESParameters: Variable encoder for the CMA-ES class.
    - initialize_mt: Initializes the CMA-ES parameters for moment tensor inversion.
    - initialize_force: Initializes the CMA-ES parameters for force source inversion.

Usage:
    from mtuq.stochastic_sampling import CMA_ES, CMAESParameters, initialize_mt, initialize_force

    # Example usage for moment tensor inversion
    parameters_list = initialize_mt(Mw_range=[4, 5], depth_range=[0, 10000])
    cma_es = CMA_ES(parameters_list, origin)
    cma_es.Solve(data_list, stations, misfit_list, process_list, db_or_greens_list, max_iter=100, wavelet=wavelet)

    # Example usage for force source inversion
    parameters_list = initialize_force(F0_range=[1e11, 1e14], depth=[0, 10000])
    cma_es = CMA_ES(parameters_list, origin)
    cma_es.Solve(data_list, stations, misfit_list, process_list, db_or_greens_list, max_iter=100, wavelet=wavelet)
"""

from mtuq.stochastic_sampling.cmaes import CMA_ES
from mtuq.stochastic_sampling.variable_encoder import CMAESParameters, initialize_mt, initialize_force
