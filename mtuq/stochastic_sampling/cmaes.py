import numpy as np
import pandas as pd
from mpi4py import MPI

import matplotlib
matplotlib.use('Agg')

from mtuq.stochastic_sampling.cmaes_utils import *
from mtuq.stochastic_sampling.cmaes_initialization import CMAESInitialization
from mtuq.stochastic_sampling.cmaes_mutant_generation import CMAESMutantGeneration
from mtuq.stochastic_sampling.cmaes_fitness_evaluation import CMAESFitnessEvaluation
from mtuq.stochastic_sampling.cmaes_plotting import CMAESPlotting
from mtuq.util.math import to_mij, to_rtp, to_gamma, to_delta, wrap_180
from mtuq import MTUQDataFrame
from mtuq.graphics import plot_combined, plot_misfit_force
from mtuq.graphics.uq._matplotlib import _plot_force_matplotlib
from mtuq.misfit import PolarityMisfit, Misfit
from mtuq.misfit.waveform.level2 import _get_time_sampling, _get_stations, _get_components, _get_weights, _get_groups, _get_data, _get_greens, _get_padding, _autocorr_1, _autocorr_2, _corr_1_2
from mtuq.util import Null
from mtuq.misfit.waveform import c_ext_L2
from mtuq.process_data import ProcessData
from mtuq.dataset import Dataset
from mtuq.event import Origin, MomentTensor, Force
from mtuq.grid import UnstructuredGrid
from mtuq.graphics import plot_data_greens2, plot_data_greens1
from mtuq.graphics.uq._matplotlib import _generate_lune, _generate_sphere, _hammer_projection
from mtuq.misfit.waveform import calculate_norm_data
from mtuq.io.clients.AxiSEM_NetCDF import AxiSEM_Client
from mtuq.greens_tensor.base import GreensTensorList
import copy


class CMA_ES(object):

    def __init__(self, parameters_list: list, origin: Origin, lmbda: int = None, callback_function=None, event_id: str = '', verbose_level: int = 0):
        '''
        parallel_CMA_ES class

        CMA-ES class for moment tensor and force inversion. The class accept a list of `CMAESParameters` objects containing the options and tunning of each of the inverted parameters. CMA-ES will be carried automatically based of the content of the `CMAESParameters` list.

        .. rubric :: Usage

        The inversion is carried in a two step procedure

        .. code::

        cma_es = parallel_CMA_ES(**parameters)
        cma_es.solve(data, process, misfit, stations, db, wavelet, iterations=10)

        .. note ::
        In the first step, the user supplies parameters such as the number of mutants, the list of inverted parameters, the catalog origin, etc. (see below for detailed argument descriptions).

        In the second step, the user supplies data, data process, misfit type, stations list, an Axisem Green's function database, a source wavelet and a number of iterations on which to carry the CMA-ES inversion (number of generations).

        .. rubric:: Parameters

        ``parameters_list`` (`list`): A list of `CMAESParameters` objects containing the options and tunning of each of the inverted parameters.

        ``lmbda`` (`int`): The number of mutants to be generated. If None, the default value is set to 4 + np.floor(3 * np.log(len(parameters_list))).

        ``origin`` (`mtuq.Event`): The origin of the event to be inverted.

        '''

        # Initialize basic properties
        self.initialization = CMAESInitialization(parameters_list, lmbda, origin, callback_function, event_id, verbose_level)
        self.mutant_generation = CMAESMutantGeneration(self.initialization._parameters, self.initialization.xmean, self.initialization.sigma, self.initialization.B, self.initialization.D, self.initialization.n, self.initialization.lmbda, self.initialization.size, self.initialization.rank, self.initialization.comm, self.initialization.verbose_level)
        self.fitness_evaluation = CMAESFitnessEvaluation(self.initialization._parameters, self.initialization.xmean, self.initialization.sigma, self.initialization.B, self.initialization.D, self.initialization.n, self.initialization.lmbda, self.initialization.size, self.initialization.rank, self.initialization.comm, self.initialization.verbose_level, self.initialization.callback, self.initialization.catalog_origin, self.initialization._greens_tensors_cache)
        self.plotting = CMAESPlotting(self.initialization._parameters, self.initialization.xmean, self.initialization.sigma, self.initialization.B, self.initialization.D, self.initialization.n, self.initialization.lmbda, self.initialization.size, self.initialization.rank, self.initialization.comm, self.initialization.verbose_level, self.initialization.callback, self.initialization.catalog_origin, self.initialization._greens_tensors_cache)

        # Set up caches and storage for logging
        self._setup_caches()

    def _setup_caches(self):
        """ Initializes caches and logging variables for performance tracking. """
        self.cache_size = 10  # Number of iterations to cache
        self.cache_counter = 0  # To keep track of cached iterations
        self.mutants_cache = np.zeros((self.initialization.n + 1, self.initialization.lmbda * self.cache_size))  # For storing [parameters + misfit]
        self.mutants_logger_list = pd.DataFrame()
        self.mean_logger_list = pd.DataFrame()

        # Define holder variables for post-processing and plotting
        self._misfit_holder = np.zeros((int(self.initialization.lmbda), 1))
        self.fig = None
        self.ax = None

    # Where the mutants are generated ... --------------------------------------------------------------
    def draw_mutants(self):
        """
        Draws mutants from a Gaussian distribution and scatters them across MPI processes.
        
        This function generates `self.initialization.lmbda` mutants from a multivariate normal distribution, applies bounds
        and repair methods where necessary, and scatters the mutants to all processes for parallel evaluation.

        Returns
        -------
        None
        """
        self.mutant_generation.draw_mutants()
        self.scattered_mutants = self.mutant_generation.scattered_mutants
        self.initialization.counteval += self.initialization.lmbda

    # Where the mutants are evaluated ... --------------------------------------------------------------
    def eval_fitness(self, data, stations, misfit, db_or_greens_list, process=None, wavelet=None, verbose=False):
        """
        eval_fitness method

        This method evaluates the misfit for each mutant of the population.

        .. rubric :: Usage

        The usage is as follows:

        .. code::

            if mode == 'db':
                eval_fitness(data, stations, misfit, db, process, wavelet)
            elif mode == 'greens':
                eval_fitness(data, stations, misfit, greens, process = None, wavelet = None)

        .. note ::
        The ordering of the CMA_ES parameters should follow the ordering of the input variables of the callback function, but this is dealt with internally if using the initialize_mt() and initialize_force() functions.

        .. rubric:: Parameters

        data (mtuq.Dataset): the data to fit (body waves, surface waves).
        stations (list): the list of stations.
        misfit (mtuq.WaveformMisfit): the associated mtuq.Misfit object.
        db (mtuq.AxiSEM_Client or mtuq.GreensTensorList): Preprocessed Greens functions or local database (for origin search).
        process (mtuq.ProcessData, optional): the processing function to apply to the Greens functions.
        wavelet (mtuq.wavelet, optional): the wavelet to convolve with the Greens functions.
        verbose (bool, optional): whether to print debug information.

        .. rubric:: Returns

        The misfit values for each mutant of the population.

        """
        return self.fitness_evaluation.eval_fitness(data, stations, misfit, db_or_greens_list, process, wavelet, verbose)
    
    # Where the mutants are gathered ... --------------------------------------------------------------
    def gather_mutants(self, verbose=False):
        '''
        gather_mutants method

        This function gathers mutants from all processes into the root process. It also uses the datalogger to construct the mutants_logger_list.

        .. rubric :: Usage

        The method is used as follows:

        .. code::

            gather_mutants(verbose=False)

        .. rubric:: Parameters

        verbose (bool):
            If set to True, prints the concatenated mutants, their shapes, and types. Default is False.

        .. rubric:: Attributes

        self.mutants (array):
            The gathered and concatenated mutants. This attribute is set to None for non-root processes after gathering.
        self.transformed_mutants (array):
            The gathered and concatenated transformed mutants. This attribute is set to None for non-root processes after gathering.
        self.mutants_logger_list (list):
            The list to which the datalogger is appended.
        '''

        # Printing the mutants on each process, their shapes and types for debugging purposes
        if self.initialization.verbose_level >= 2:
            print(self.scattered_mutants, '\n', 'shape is', np.shape(self.scattered_mutants), '\n', 'type is', type(self.scattered_mutants))


        self.mutants = self.initialization.comm.gather(self.scattered_mutants, root=0)
        if self.initialization.rank == 0:
            self.mutants = np.concatenate(self.mutants, axis=1)
            if self.initialization.verbose_level >= 2:
                print(self.mutants, '\n', 'shape is', np.shape(self.mutants), '\n', 'type is', type(self.mutants)) # - DEBUG PRINT
        else:
            self.mutants = None


        self.transformed_mutants = self.initialization.comm.gather(self.fitness_evaluation.transformed_mutants, root=0)
        if self.initialization.rank == 0:
            self.transformed_mutants = np.concatenate(self.transformed_mutants, axis=1)
            if self.initialization.verbose_level >= 2:
                print(self.transformed_mutants, '\n', 'shape is', np.shape(self.transformed_mutants), '\n', 'type is', type(self.transformed_mutants)) # - DEBUG PRINT
        else:
            self.transformed_mutants = None


        if self.initialization.comm.rank == 0:
            current_df = self._datalogger(mean=False)
        # Log the mutants from _datalogger object
        # If self.mutants_logger_list is empty, initialize it with the current DataFrame
            if self.mutants_logger_list.empty:
                self.mutants_logger_list = current_df
            else:
                # Concatenate the current DataFrame to the logger list
                self.mutants_logger_list = pd.concat([self.mutants_logger_list, current_df], ignore_index=True)


    # Sort the mutants by fitness
    def fitness_sort(self, misfit):
        """
        fitness_sort method

        This function sorts the mutants by fitness, and updates the misfit_holder.

        .. rubric :: Usage

        The method is used as follows:

        .. code::

            fitness_sort(misfit)

        .. rubric:: Parameters

        misfit (array):

            The misfit array to sort the mutants by. Can be the sum of body and surface wave misfits, or the misfit of a single wave type.

        .. rubric:: Attributes

        self.mutants (array):
            The sorted mutants.
        self.transformed_mutants (array):
            The sorted transformed mutants.
        self._misfit_holder (array):
            The updated misfit_holder. Reset to 0 after sorting.

        """
        
        if self.initialization.rank == 0:
            self.mutants = self.mutants[:,np.argsort(misfit.T)[0]]
            self.transformed_mutants = self.transformed_mutants[:,np.argsort(misfit.T)[0]]
        self._misfit_holder *= 0
    # Update step size
    def update_step_size(self):
        # Step size control
        if self.initialization.rank == 0:
            self.initialization.ps = (1-self.initialization.cs)*self.initialization.ps + np.sqrt(self.initialization.cs*(2-self.initialization.cs)*self.initialization.mueff) * self.initialization.invsqrtC @ (self.mean_diff(self.initialization.xmean, self.initialization.xold) / self.initialization.sigma)
    # Update covariance matrix
    def update_covariance(self):
        # Covariance matrix adaptation
        if self.initialization.rank == 0:
            ps_norm = np.linalg.norm(self.initialization.ps)
            condition = ps_norm / np.sqrt(1 - (1 - self.initialization.cs) ** (2 * (self.initialization.counteval // self.initialization.lmbda + 1))) / self.initialization.chin
            threshold = 1.4 + 2 / (self.initialization.n + 1)

            if condition < threshold:
                self.hsig = 1
            else:
                self.hsig = 0

            self.dhsig = (1 - self.hsig)*self.initialization.cc*(2-self.hsig)

            self.initialization.pc = (1 - self.initialization.cc) * self.initialization.pc + self.hsig * np.sqrt(self.initialization.cc * (2 - self.initialization.cc) * self.initialization.mueff) * self.mean_diff(self.initialization.xmean, self.initialization.xold) / self.initialization.sigma

            artmp = (1/self.initialization.sigma) * self.mean_diff(self.mutants[:,0:int(self.initialization.mu)], self.initialization.xold)
            # Old version - from the pureCMA Matlab implementation on Wikipedia
            # self.C = (1-self.c1-self.cmu) * self.C + self.c1 * (self.pc@self.pc.T + (1-self.hsig) * self.cc*(2-self.cc) * self.C) + self.cmu * artmp @ np.diag(self.weights.T[0]) @ artmp.T
            # Old version - from CMA-ES tutorial by Hansen et al. (2016) - https://arxiv.org/pdf/1604.00772.pdf
            # self.C = (1 + self.c1*self.dhsig - self.c1 - self.cmu*np.sum(self.weights)) * self.C + self.c1 * self.pc@self.pc.T + self.cmu * artmp @ np.diag(self.weights.T[0]) @ artmp.T

            # New version - from the purecma python implementation on GitHub - September, 2017, version 3.0.0
            # https://github.com/CMA-ES/pycma/blob/development/cma/purecma.py
            self.initialization.C *= 1 + self.initialization.c1*self.dhsig - self.initialization.c1 - self.initialization.cmu * sum(self.initialization.weights) # Discount factor
            self.initialization.C += self.initialization.c1 * self.initialization.pc @ self.initialization.pc.T # Rank one update (pc.pc^T is a matrix of rank 1) 
            self.initialization.C += self.initialization.cmu * artmp @ np.diag(self.initialization.weights.T[0]) @ artmp.T # Rank mu update, supported by the mu best individuals

            # Adapt step size
            # We do sigma_i+1 = sigma_i * exp((cs/damps)*(||ps||/E[N(0,I)]) - 1) only now as artmp needs sigma_i
            self.initialization.sigma = self.initialization.sigma * np.exp((self.initialization.cs/self.initialization.damps)*(np.linalg.norm(self.initialization.ps)/self.initialization.chin - 1))

            if self.initialization.counteval - self.initialization.eigeneval > self.initialization.lmbda/(self.initialization.c1+self.initialization.cmu)/self.initialization.n/10:
                self.initialization.eigeneval = self.initialization.counteval
                self.initialization.C = np.triu(self.initialization.C) + np.triu(self.initialization.C,1).T
                self.initialization.D,self.initialization.B = np.linalg.eig(self.initialization.C)
                self.initialization.D = np.array([self.initialization.D]).T
                self.initialization.D = np.sqrt(self.initialization.D)
                self.initialization.invsqrtC = self.initialization.B @ np.diag(self.initialization.D[:,0]**-1) @ self.initialization.B.T
        
        self.initialization.iteration = self.initialization.counteval//self.initialization.lmbda
    # Update mean
    def update_mean(self):
        # Update the mean
        if self.initialization.rank == 0:
            self.initialization.xold = self.initialization.xmean.copy()
            self.initialization.xmean = np.dot(self.mutants[:, 0:len(self.initialization.weights)], self.initialization.weights)
            for _i, param in enumerate(self.initialization._parameters):
                if param.repair == 'wrapping':
                    print('computing wrapped mean for parameter:', param.name)
                    self.initialization.xmean[_i] = self.circular_mean(_i)
            
            # Update the mean datalogger
            current_mean_df = self._datalogger(mean=True)

            # If self.mean_logger_list is empty, initialize it with the current DataFrame
            if self.mean_logger_list.empty:
                self.mean_logger_list = current_mean_df
            else:
                # Concatenate the current DataFrame to the logger list
                self.mean_logger_list = pd.concat([self.mean_logger_list, current_mean_df], ignore_index=True)

    # Utility functions --------------------------------------------------------------
    def circular_mean(self, id):
        '''
        Compute the circular mean on the "id"th parameter. Ciruclar mean allows to compute mean of the samples on a periodic space.
        '''
        param = self.mutants[id]
        a = linear_transform(param, 0, 360)-180
        mean = np.rad2deg(np.arctan2(np.sum(np.sin(np.deg2rad(a[range(int(self.initialization.mu))]))*self.initialization.weights.T), np.sum(np.cos(np.deg2rad(a[range(int(self.initialization.mu))]))*self.initialization.weights.T)))+180
        mean = inverse_linear_transform(mean, 0, 360)
        return mean

    def smallestAngle(self, targetAngles, currentAngles) -> np.ndarray:
        """
        smallestAngle method

        This function calculates the smallest angle (in degrees) between two given sets of angles. It computes the difference between the target and current angles, making sure the result stays within the range [0, 360). If the resulting difference is more than 180, it is adjusted to go in the shorter, negative direction.

        .. rubric :: Usage

        The method is used as follows:

        .. code::

            smallest_diff = smallestAngle(targetAngles, currentAngles)

        .. rubric:: Parameters

        targetAngles (np.ndarray):
            An array containing the target angles in degrees.
        currentAngles (np.ndarray):
            An array containing the current angles in degrees.

        .. rubric:: Returns

        diffs (np.ndarray):
            An array containing the smallest difference in degrees between the target and current angles.

        """

        # Subtract the angles, constraining the value to [0, 360)
        diffs = (targetAngles - currentAngles) % 360

        # If we are more than 180 we're taking the long way around.
        # Let's instead go in the shorter, negative direction
        diffs[diffs > 180] = -(360 - diffs[diffs > 180])
        return diffs

    def mean_diff(self, new, old):
        # Compute mean change, and apply circular difference for wrapped repair methods (implying periodic parameters)
        diff = new-old
        for _i, param in enumerate(self.initialization._parameters):
            if param.repair == 'wrapping':
                angular_diff = self.smallestAngle(linear_transform(new[_i], 0, 360), linear_transform(old[_i], 0, 360))
                angular_diff = inverse_linear_transform(angular_diff, 0, 360)
                diff[_i] = angular_diff
        return diff

    def create_origins(self):
        
        # Check which of the three origin modifiers are in the parameters
        if 'depth' in self.initialization._parameters_names:
            depth = self.fitness_evaluation.transformed_mutants[self.initialization._parameters_names.index('depth')]
        else:
            depth = self.initialization.catalog_origin.depth_in_m
        if 'latitude' in self.initialization._parameters_names:
            latitude = self.fitness_evaluation.transformed_mutants[self.initialization._parameters_names.index('latitude')]
        else:
            latitude = self.initialization.catalog_origin.latitude
        if 'longitude' in self.initialization._parameters_names:
            longitude = self.fitness_evaluation.transformed_mutants[self.initialization._parameters_names.index('longitude')]
        else:
            longitude = self.initialization.catalog_origin.longitude
        
        self.origins = []
        for i in range(len(self.scattered_mutants[0])):
            self.origins += [self.initialization.catalog_origin.copy()]
            if 'depth' in self.initialization._parameters_names:
                setattr(self.origins[-1], 'depth_in_m', depth[i])
            if 'latitude' in self.initialization._parameters_names:
                setattr(self.origins[-1], 'latitude', latitude[i])
            if 'longitude' in self.initialization._parameters_names:
                setattr(self.origins[-1], 'longitude', longitude[i])

    def return_candidate_solution(self, id=None):
        # Only required on rank 0
        if self.initialization.rank == 0:
            if not id == None:
                return_x = np.array([self.mutants[:,id]]).T
            else:
                return_x = self.initialization.xmean
            self.transformed_mean = np.zeros_like(return_x)
            for _i, param in enumerate(self.initialization._parameters):
                # Print paramter scaling if verbose
                if param.scaling == 'linear':
                    self.transformed_mean[_i] = linear_transform(return_x[_i], param.lower_bound, param.upper_bound)
                elif param.scaling == 'log':
                    self.transformed_mean[_i] = logarithmic_transform(return_x[_i], param.lower_bound, param.upper_bound)
                else:
                    raise ValueError("Unrecognized scaling, must be linear or log")
                # Apply optional projection operator to each parameter
                if not param.projection is None:
                    self.transformed_mean[_i] = np.asarray(list(map(param.projection, self.transformed_mean[_i])))


            # Check which of the three origin modifiers are in the parameters
            if 'depth' in self.initialization._parameters_names:
                depth = self.transformed_mean[self.initialization._parameters_names.index('depth')]
            else:
                depth = self.initialization.catalog_origin.depth_in_m
            if 'latitude' in self.initialization._parameters_names:
                latitude = self.transformed_mean[self.initialization._parameters_names.index('latitude')]
            else:
                latitude = self.initialization.catalog_origin.latitude
            if 'longitude' in self.initialization._parameters_names:
                longitude = self.transformed_mean[self.initialization._parameters_names.index('longitude')]
            else:
                longitude = self.initialization.catalog_origin.longitude
            
            self.origins = []
            for i in range(len(self.transformed_mean[0])):
                self.origins += [self.initialization.catalog_origin.copy()]
                if 'depth' in self.initialization._parameters_names:
                    setattr(self.origins[-1], 'depth_in_m', depth[i])
                if 'latitude' in self.initialization._parameters_names:
                    setattr(self.origins[-1], 'latitude', latitude[i])
                if 'longitude' in self.initialization._parameters_names:
                    setattr(self.origins[-1], 'longitude', longitude[i])

            return(self.transformed_mean, self.origins)

    def _datalogger(self, mean=False):
        """
        _datalogger method

        This method saves in memory all of the CMA-ES mutants drawn and evaluated during the inversion. This allows quick access to the inversion records in order to plot the misfit. The data is stored within a pandas.DataFrame().

        Note
        ----------
        When mean=False, the datalogger stores the coordinates of each mutant (Mw, v, w, kappa, sigma,...) and misfit at the current iteration.

        When mean=True, the datalogger stores the coordinates of the mean mutant at the current iteration. The mean mutant's misfit is not evaluated, thus only its coordinates are returned.
            """
        if self.initialization.rank == 0:
            if mean==False:
                coordinates = self.transformed_mutants.T
                misfit_values = self._misfit_holder
                results = np.hstack((coordinates, misfit_values))
                columns_labels=self.initialization._parameters_names+["misfit"]

            if mean==True:
                self.transformed_mean = np.zeros_like(self.initialization.xmean)
                for _i, param in enumerate(self.initialization._parameters):
                    if param.scaling == 'linear':
                        self.transformed_mean[_i] = linear_transform(self.initialization.xmean[_i], param.lower_bound, param.upper_bound)
                    elif param.scaling == 'log':
                        self.transformed_mean[_i] = logarithmic_transform(self.initialization.xmean[_i], param.lower_bound, param.upper_bound)
                    else:
                        raise ValueError("Unrecognized scaling, must be linear or log")
                    # Apply optional projection operator to each parameter
                    if not param.projection is None:
                        self.transformed_mean[_i] = np.asarray(list(map(param.projection, self.transformed_mean[_i])))

                results = self.transformed_mean.T
                columns_labels=self.initialization._parameters_names

            da = pd.DataFrame(
            data=results,
            columns=columns_labels
            )
            return(MTUQDataFrame(da))

    def _prep_and_cache_C_arrays(self, data, greens, misfit, stations):
        """
        Helper function to prepare and cache C compatible arrays for the misfit function evaluation. 

        It is responsible for preparing the data arrays for the inversion, in a format expected by the lower-level 
        c-code for misfit evaluation. Mostly copy-pasted from mtuq.misfit.waveform.level2

        Only used when the mode is 'greens'.
        """

        from mtuq.misfit.waveform.level2 import _get_time_sampling, _get_stations, \
        _get_components, _get_weights, _get_groups, _get_data, _get_greens, \
        _get_padding, _autocorr_1, _autocorr_2, _corr_1_2
        from mtuq.util import Null

        msg_handle = Null()
        # If no attributes are present, create the dictionaries
        if not hasattr(self, 'data_cache'):
            self.data_cache = {}

        # Use the misfit object __hash__ method to create a unique key for the data_cache dictionary.
        key = hash(misfit)

        # If the key is not present in the data_cache, prepare the data arrays for the inversion before caching them.
        if key not in self.data_cache:

            # Precompute the data arrays for the inversion before caching them.
            nt, dt = _get_time_sampling(data)
            stations = _get_stations(data)
            components = _get_components(data)

            weights = _get_weights(data, stations, components)

            # which components will be used to determine time shifts (boolean array)?
            groups = _get_groups(misfit.time_shift_groups, components)

            # Set include_mt and include_force based on the mode
            if self.initialization.mode in ['mt', 'mt_dc', 'mt_dev']:
                for g in greens:
                    g.include_mt = True
                    g.include_force = False
            elif self.initialization.mode == 'force':
                for g in greens:
                    g.include_mt = False
                    g.include_force = True
            else:
                raise ValueError("Invalid mode. Supported modes: 'mt', 'mt_dc', 'mt_dev', 'force'.")

            #
            # collapse main structures into NumPy arrays
            #
            data = _get_data(data, stations, components)
            greens = _get_greens(greens, stations, components)


            #
            # cross-correlate data and synthetics
            #
            padding = _get_padding(misfit.time_shift_min, misfit.time_shift_max, dt)
            data_data = _autocorr_1(data)
            greens_greens = _autocorr_2(greens, padding)
            greens_data = _corr_1_2(data, greens, padding)

            if misfit.norm=='hybrid':
                hybrid_norm = 1
            else:
                hybrid_norm = 0

            #
            # collect message attributes
            #
            try:
                msg_args = [getattr(msg_handle, attrib) for attrib in 
                    ['start', 'stop', 'percent']]
            except:
                msg_args = [0, 0, 0]

            # Cache the data arrays for the inversion to be called by c_ext_L2.misfit(data_data, greens_data, greens_greens, sources, groups, weights, hybrid_norm, dt, padding[0], padding[1], debug_level, *msg_args)
            self.data_cache[key] = {
                'data_data': data_data,
                'greens_data': greens_data,
                'greens_greens': greens_greens,
                'groups': groups,
                'weights': weights,
                'hybrid_norm': hybrid_norm,
                'dt': dt,
                'padding': padding,
                'msg_args': msg_args
            }
        
        elif key in self.data_cache:
            if self.initialization.verbose_level >= 2:
                print('Data arrays already cached. Nothing to do here.')
            pass
    
    # def _prepare_and_cache_green_green(self):

    # Main method responsible for the inversion ----------------------------------------
    def Solve(self, data_list, stations, misfit_list, process_list, db_or_greens_list, max_iter=100, wavelet=None, plot_interval=10, iter_count=0, misfit_weights=None, **kwargs):
        """
        Solves for the best-fitting source model using the CMA-ES algorithm. This is the master method used in inversions. 

        This method iteratively draws mutants, evaluates their fitness based on misfits between synthetic and observed data, and updates the mean and covariance of the CMA-ES distribution. At specified intervals, it also plots mean waveforms and results for visualization.

        Parameters
        ----------
        data_list : list
            List of observed data sets. (e.g. [data_sw] or [data_bw, data_sw])
        stations : list
            List of stations (generally obtained from mtuq method data.get_stations())
        misfit_list : list
            List of mtuq misfit objects (e.g. [misfit_sw] or [misfit_bw, misfit_sw]).
        process_list : list
            List of mtuq ProcessData objects to apply to data (e.g. [process_sw] or [process_bw, process_sw]).
        db_or_greens_list : list or AxiSEM_Client object
            Either an AxiSEM database client or a mtuq GreensTensorList.
        max_iter : int, optional
            Maximum number of iterations to perform. Default is 100. A stoping criterion will be implemented in the future.
        wavelet : object, optional
            Wavelet for source time function. Default is None. Required when db_or_greens_list is an AxiSEM database client.
        plot_interval : int, optional
            Interval at which plots of mean waveforms and results should be generated. Default is 10.
        iter_count : int, optional
            Current iteration count, should be useful for resuming. Default is 0.
        src_type : str, optional
            Type of source model, one of ['full', 'deviatoric', 'dc']. Default is full.
        misfit_weights : list, optional
            List of misfit weights. Default is None for equal weights.
        **kwargs
            Additional keyword arguments passed to eval_fitness method.

        Returns
        -------
        None

        Note
        ----
        This method is the wrapper that automate the execution of the CMA-ES algorithm. It is the default workflow for Moment tensor and Force inversion and should not work with a "custom" inversion (multiple-sub events, source time function, etc.). It interacts with the  `draw_mutants`, `eval_fitness`, `gather_mutants`, `fitness_sort`, `update_mean`, `update_step_size` and `update_covariance`. 
        """

        greens_cache = {}
        data_cache = {}

        if self.initialization.rank == 0:
            # Check Solve inputs
            self._check_Solve_inputs(data_list, stations, misfit_list, process_list, db_or_greens_list, max_iter, wavelet, plot_interval, iter_count)

        # Handling of the misfit weights. If not provided, equal weights are used, otherwise the weights are used to derive percentages.
        if misfit_weights is None:
            misfit_weights = [1.0] * len(data_list)
        elif len(misfit_weights) != len(data_list):
            raise ValueError("Length of misfit_weights must match the length of data_list.")

        total_weight = sum(misfit_weights)
        if total_weight == 0:
            raise ValueError("Sum of weights cannot be zero.")

        misfit_weights = [w/total_weight for w in misfit_weights]


        for i in range(max_iter):
            if self.initialization.rank == 0:
                print('Iteration %d\n' % (i + iter_count))
            
            self.draw_mutants()

            misfits = []
            for j, (current_data, current_misfit, process) in enumerate(zip(data_list, misfit_list, process_list)):
                mode = 'db' if isinstance(db_or_greens_list, AxiSEM_Client) else 'greens'
                # get greens[j] or db depending on mode from db_or_greens_list
                greens = db_or_greens_list[j] if mode == 'greens' else None
                db = db_or_greens_list if mode == 'db' else None

                if mode == 'db':
                    misfit_values = self.eval_fitness(current_data, stations, current_misfit, db, process_list[j], wavelet, **kwargs)
                elif mode == 'greens':
                    if i == 1: 
                        raw_greens_to_cache = copy.deepcopy(greens)
                        raw_data_to_cache = copy.deepcopy(current_data)
                        self._prep_and_cache_C_arrays(raw_data_to_cache, raw_greens_to_cache, current_misfit, stations)
                    misfit_values = self.eval_fitness(current_data, stations, current_misfit, greens,  **kwargs)

                norm = self._get_data_norm(current_data, current_misfit)
                misfit_values /= norm
                misfits.append(misfit_values)


            weighted_misfits = [w * m for w, m in zip(misfit_weights, misfits)]
            total_missfit = sum(weighted_misfits)
            self._misfit_holder += total_missfit
            self.gather_mutants()
            self.fitness_sort(total_missfit)
            self.update_mean()
            self.update_step_size()
            self.update_covariance()

            if i != 0 and i % plot_interval == 0 or i == max_iter - 1:
                if self.initialization.rank == 0:
                    self.plotting.plot_mean_waveforms(data_list, process_list, misfit_list, stations, db_or_greens_list)
                    if self.initialization.mode in ['mt', 'mt_dc', 'mt_dev']:
                        print('Plotting results for iteration %d\n' % (i + iter_count))
                        result = self.mutants_logger_list

                        # Handling the mean solution
                        if self.initialization.mode == 'mt':
                            V,W = self._datalogger(mean=True)['v'], self._datalogger(mean=True)['w']
                        elif self.initialization.mode == 'mt_dev':
                            V = self._datalogger(mean=True)['v']
                            W = 0
                        elif self.initialization.mode == 'mt_dc':
                            V = W = 0

                    # If mode is mt, mt_dev or mt_dc, plot the misfit map
                    if self.initialization.mode in ['mt', 'mt_dev', 'mt_dc']:
                        plot_combined(self.initialization.event_id+'_combined_misfit_map.png', result, colormap='viridis', best_vw = (V,W))
                    elif self.initialization.mode == 'force':
                        print('Plotting results for iteration %d\n' % (i + iter_count))
                        result = self.mutants_logger_list
                        plot_misfit_force(self.initialization.event_id+'_misfit_map.png', result, colormap='viridis', backend=_plot_force_matplotlib, plot_type='colormesh', best_force=self.return_candidate_solution()[0][1::])

    def _check_Solve_inputs(self, data_list, stations, misfit_list, process_list, db_or_greens_list, max_iter=100, wavelet=None, plot_interval=10, iter_count=0, **kwargs):
        """
        Checks the validity of input arguments for the Solve method.
        
        Raises
        ------
        ValueError : If any of the inputs are invalid.
        """

        if not isinstance(data_list, list):
            if isinstance(data_list, Dataset):
                data_list = [data_list]
            else:
                raise ValueError("`data_list` should be a list of mtuq Dataset or an array containing polarities.")
        if not isinstance(stations, list):
            raise ValueError("`stations` should be a list of mtuq Station objects.")
        if not isinstance(misfit_list, list):
            if isinstance(misfit_list, PolarityMisfit) or isinstance(misfit_list, Misfit):
                misfit_list = [misfit_list]
            else:
                raise ValueError("`misfit_list` should be a list of mtuq Misfit objects.")
        if not isinstance(process_list, list):
            if isinstance(process_list, ProcessData):
                process_list = [process_list]
            else:
                raise ValueError("`process_list` should be a list of mtuq ProcessData objects.")
        if not isinstance(db_or_greens_list, list):
            if isinstance(db_or_greens_list, AxiSEM_Client) or isinstance(db_or_greens_list, GreensTensorList):
                db_or_greens_list = [db_or_greens_list]
            else:
                raise ValueError("`db_or_greens_list` should be a list of either mtuq AxiSEM_Client or GreensTensorList objects.")
        if not isinstance(max_iter, int):
            raise ValueError("`max_iter` should be an integer.")
        if any(isinstance(db, AxiSEM_Client) for db in db_or_greens_list) and wavelet is None:
            raise ValueError("wavelet must be specified if database is an AxiSEM_Client")
        if not isinstance(plot_interval, int):
            raise ValueError("`plot_interval` should be an integer.")
        if iter_count is not None and not isinstance(iter_count, int):
            raise ValueError("`iter_count` should be an integer or None.")            

    def _get_data_norm(self, data, misfit):
        """
        Compute the norm of the data using the calculate_norm_data function.

        Arguments
        ----------
            data: The evaluated processed data.
            misfit: The misfit object used to evaluate the data object

        """

        # If misfit type is Polarity misfit, use the sum of the absolute values of the data as number of used stations.
        if isinstance(misfit, PolarityMisfit):
            return np.sum(np.abs(data))
        # Else, use the calculate_norm_data function.
        else:
            if isinstance(misfit.time_shift_groups, str):
                components = list(misfit.time_shift_groups)
            elif isinstance(misfit.time_shift_groups, list):
                components = list("".join(misfit.time_shift_groups))
            
            return calculate_norm_data(data, misfit.norm, components)
