import numpy as np
import pandas as pd
from mpi4py import MPI
import matplotlib
from mtuq.stochastic_sampling.cmaes_utils import linear_transform, logarithmic_transform, in_bounds, array_in_bounds, Repair, inverse_linear_transform
from mtuq.util.math import to_mij, to_rtp, to_gamma, to_delta
from mtuq import MTUQDataFrame
from mtuq.grid.moment_tensor import UnstructuredGrid
from mtuq.grid.force import UnstructuredGrid
from mtuq.graphics import plot_data_greens2, plot_data_greens1
from mtuq.io.clients.AxiSEM_NetCDF import Client as AxiSEM_Client
from mtuq.greens_tensor.base import GreensTensorList
from mtuq.dataset import Dataset
from mtuq.event import MomentTensor, Force, Origin
from mtuq.graphics.uq._matplotlib import _hammer_projection, _generate_lune, _generate_sphere
from mtuq.graphics import plot_combined, plot_misfit_force
from mtuq.graphics.uq._matplotlib import _plot_force_matplotlib
from mtuq.misfit import Misfit, PolarityMisfit, WaveformMisfit
from mtuq.misfit.waveform import c_ext_L2, calculate_norm_data
from mtuq.process_data import ProcessData
import copy
from mtuq.stochastic_sampling.cmaes_plotting import plot_mean_waveforms


matplotlib.use('Agg')


class CMA_ES(object):
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES) class for moment tensor and force inversion.

    This class implements the CMA-ES algorithm for seismic inversion. It accepts a list of `CMAESParameters` objects containing the options and tuning of each of the inverted parameters. The inversion is carried out automatically based on the content of the `CMAESParameters` list.

    Attributes
    ----------
    rank : int
        The rank of the current MPI process.
    size : int
        The total number of MPI processes.
    comm : MPI.Comm
        The MPI communicator.
    event_id : str
        The event ID for the inversion.
    verbose_level : int
        The verbosity level for logging.
    _parameters : list
        A list of `CMAESParameters` objects.
    _parameters_names : list
        A list of parameter names.
    n : int
        The number of parameters.
    lmbda : int
        The number of mutants to be generated.
    catalog_origin : Origin
        The origin of the event to be inverted.
    callback : function
        The callback function used for the inversion.
    xmean : numpy.ndarray
        The mean of the parameter distribution.
    sigma : float
        The step size for the CMA-ES algorithm.
    iteration : int
        The current iteration number.
    counteval : int
        The number of function evaluations.
    _greens_tensors_cache : dict
        A cache for Green's tensors.
    mu : int
        The number of top-performing mutants used for recombination.
    weights : numpy.ndarray
        The weights for recombination.
    mueff : float
        The effective number of top-performing mutants.
    cs : float
        The learning rate for step-size control.
    damps : float
        The damping parameter for step-size control.
    cc : float
        The learning rate for covariance matrix adaptation.
    acov : float
        The covariance matrix adaptation parameter.
    c1 : float
        The learning rate for rank-one update of the covariance matrix.
    cmu : float
        The learning rate for rank-mu update of the covariance matrix.
    ps : numpy.ndarray
        The evolution path for step-size control.
    pc : numpy.ndarray
        The evolution path for covariance matrix adaptation.
    B : numpy.ndarray
        The matrix of eigenvectors of the covariance matrix.
    D : numpy.ndarray
        The matrix of eigenvalues of the covariance matrix.
    C : numpy.ndarray
        The covariance matrix.
    invsqrtC : numpy.ndarray
        The inverse square root of the covariance matrix.
    eigeneval : int
        The number of eigenvalue evaluations.
    chin : float
        The expected length of the evolution path.
    mutants : numpy.ndarray
        The array of mutants.
    cache_size : int
        The number of iterations to cache.
    cache_counter : int
        The counter for cached iterations.
    mutants_cache : numpy.ndarray
        The cache for mutants.
    mutants_logger_list : pandas.DataFrame
        The logger for mutants.
    mean_logger_list : pandas.DataFrame
        The logger for the mean solution.
    _misfit_holder : numpy.ndarray
        The holder for misfit values.
    fig : matplotlib.figure.Figure
        The figure object for plotting.
    ax : matplotlib.axes.Axes
        The axis object for plotting.

    Methods
    -------
    __init__(self, parameters_list, origin, lmbda=None, callback_function=None, event_id='', verbose_level=0)
        Initializes the CMA-ES class with the given parameters.
    _initialize_mpi_communicator(self)
        Initializes the MPI communicator and sets the process rank and size.
    _initialize_logging(self, event_id, verbose_level)
        Sets up logging properties like event_id and verbosity level.
    _initialize_parameters(self, parameters_list, lmbda, origin, callback_function)
        Initializes the parameters for the CMA-ES algorithm.
    _set_default_callback(self)
        Sets the default callback function based on the parameter names.
    _setup_caches(self)
        Initializes caches and logging variables for performance tracking.
    draw_mutants(self)
        Draws mutants from a Gaussian distribution and scatters them across MPI processes.
    _generate_mutants(self)
        Generates all `self.lmbda` mutants from a Gaussian distribution.
    _draw_single_mutant(self)
        Draws a single mutant from the Gaussian distribution.
    _repair_and_redraw_mutants(self)
        Applies repair methods and redraws to all mutants, parameter by parameter.
    _redraw_param_until_valid(self, param_values, bounds)
        Redraws the out-of-bounds values of a parameter array until they are within bounds.
    _apply_repair_to_param(self, param_values, bounds, param_idx)
        Applies a repair method to the full array of parameter values if defined.
    _scatter_mutants(self)
        Splits and scatters the mutants across processes.
    _receive_mutants(self)
        Receives scattered mutants on non-root processes.
    eval_fitness(self, data, stations, misfit, db_or_greens_list, process=None, wavelet=None, verbose=False)
        Evaluates the misfit for each mutant of the population.
    _eval_fitness_db(self, data, stations, misfit, db_or_greens_list, process, wavelet)
        Helper function to evaluate fitness for 'db' mode.
    _eval_fitness_greens(self, data, stations, misfit, db_or_greens_list)
        Helper function to evaluate fitness for 'greens' mode.
    gather_mutants(self, verbose=False)
        Gathers mutants from all processes into the root process.
    fitness_sort(self, misfit)
        Sorts the mutants by fitness and updates the misfit_holder.
    update_step_size(self)
        Updates the step size for the CMA-ES algorithm.
    update_covariance(self)
        Updates the covariance matrix for the CMA-ES algorithm.
    update_mean(self)
        Updates the mean of the parameter distribution.
    circular_mean(self, id)
        Computes the circular mean for a given parameter.
    smallestAngle(self, targetAngles, currentAngles)
        Calculates the smallest angle (in degrees) between two given sets of angles.
    mean_diff(self, new, old)
        Computes the mean change and applies circular difference for wrapped repair methods.
    create_origins(self)
        Creates a list of origins for each mutant.
    return_candidate_solution(self, id=None)
        Returns the candidate solution for a given mutant.
    _datalogger(self, mean=False)
        Logs the coordinates and misfit values of the mutants.
    _prep_and_cache_C_arrays(self, data, greens, misfit, stations)
        Prepares and caches C compatible arrays for the misfit function evaluation.
    Solve(self, data_list, stations, misfit_list, process_list, db_or_greens_list, max_iter=100, wavelet=None, plot_interval=10, iter_count=0, misfit_weights=None, **kwargs)
        Solves for the best-fitting source model using the CMA-ES algorithm.
    _transform_mutants(self)
        Transforms local mutants on each process based on the parameters scaling and projection settings.
    _generate_sources(self)
        Generates sources by calling the callback function on transformed data according to the set mode.
    _get_greens_tensors_key(self, process)
        Gets the body-wave or surface-wave key for the GreensTensors object from the ProcessData object.
    _check_greens_input_combination(self, db, process, wavelet)
        Checks the validity of the given parameters.
    _check_Solve_inputs(self, data_list, stations, misfit_list, process_list, db_or_greens_list, max_iter=100, wavelet=None, plot_interval=10, iter_count=0, **kwargs)
        Checks the validity of input arguments for the Solve method.
    _get_data_norm(self, data, misfit)
        Computes the norm of the data using the calculate_norm_data function.
    """

    def __init__(self, parameters_list: list, origin: Origin, lmbda: int = None, callback_function=None, event_id: str = '', verbose_level: int = 0):
        """
        Initializes the CMA-ES class with the given parameters.

        Parameters
        ----------
        parameters_list : list
            A list of `CMAESParameters` objects containing the options and tuning of each of the inverted parameters.
        origin : Origin
            The origin of the event to be inverted.
        lmbda : int, optional
            The number of mutants to be generated. If None, the default value is set to 4 + np.floor(3 * np.log(len(parameters_list))).
        callback_function : function, optional
            The callback function used for the inversion.
        event_id : str, optional
            The event ID for the inversion.
        verbose_level : int, optional
            The verbosity level for logging.
        """

        # Initialize basic properties
        self._initialize_mpi_communicator()
        self._initialize_logging(event_id, verbose_level)
        self._initialize_parameters(parameters_list, lmbda, origin, callback_function)
        
        # Set up caches and storage for logging
        self._setup_caches()

    def _initialize_mpi_communicator(self):
        """Initializes the MPI communicator and sets the process rank and size."""
        self.rank = 0
        self.size = 1
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
    
    def _initialize_logging(self, event_id, verbose_level):
        """Sets up logging properties like event_id and verbosity level."""
        self.event_id = event_id
        self.verbose_level = verbose_level
        if self.rank == 0:
            print(f'Initializing CMA-ES inversion for event {self.event_id}')

    def _initialize_parameters(self, parameters_list, lmbda, origin, callback_function):
        """
        Initializes the parameters for the CMA-ES algorithm, including population size, callback function,
        and variables related to step-size control and covariance matrix adaptation.

        Parameters
        ----------
        parameters_list : list
            A list of `CMAESParameters` objects containing the options and tuning of each of the inverted parameters.
        lmbda : int, optional
            The number of mutants to be generated. If None, the default value is set to 4 + np.floor(3 * np.log(len(parameters_list))).
        origin : Origin
            The origin of the event to be inverted.
        callback_function : function, optional
            The callback function used for the inversion.
        """
        self._parameters = parameters_list
        self._parameters_names = [parameter.name for parameter in parameters_list]
        self.n = len(self._parameters)

        # Set the default number of mutants (lambda) if not specified
        if lmbda is None:
            self.lmbda = int(4 + np.floor(3 * np.log(self.n)))
        else:
            self.lmbda = lmbda

        # Ensure that the number of MPI processes is not greater than the population size
        if self.size > self.lmbda:
            raise ValueError(f'Number of MPI processes ({self.size}) exceeds population size ({self.lmbda})')

        # Validate the origin parameter
        if not isinstance(origin, Origin):
            raise ValueError('The "origin" parameter must be an instance of mtuq.event.Origin. Please provide a valid object to be used as catalog origin.')

        # Set the origin for the inversion
        self.catalog_origin = origin

        # Handle callback function initialization
        self.callback = callback_function
        if self.callback is None:
            self._set_default_callback()

        # Initialize parameters tied to the CMA-ES algorithm
        self.xmean = np.asarray([[param.initial for param in self._parameters]]).T
        self.sigma = 0.5  # Default initial Gaussian variance for all parameters
        self.iteration = 0
        self.counteval = 0
        self._greens_tensors_cache = {}

        # Weight initialization for recombination
        self.mu = np.floor(self.lmbda / 2)
        a = 1  # Use 1/2 in tutorial and 1 in publication
        self.weights = np.array([np.log(self.mu + a) - np.log(np.arange(1, self.mu + 1))]).T
        self.weights /= sum(self.weights)
        self.mueff = sum(self.weights)**2 / sum(self.weights**2)

        # Step-size control parameters
        self.cs = (self.mueff + 2) / (self.n + self.mueff + 5)
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.n + 1)) - 1) + self.cs

        # Covariance matrix adaptation parameters
        self.cc = (4 + self.mueff / self.n) / (self.n + 4 + 2 * self.mueff / self.n)
        self.acov = 2
        self.c1 = self.acov / ((self.n + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, self.acov * (self.mueff - 2 + 1 / self.mueff) / ((self.n + 2)**2 + self.acov * self.mueff / 2))

        # Initialize step-size and covariance-related variables
        self.ps = np.zeros_like(self.xmean)
        self.pc = np.zeros_like(self.xmean)
        self.B = np.eye(self.n, self.n)
        self.D = np.ones((self.n, 1))
        self.C = self.B @ np.diag(self.D[:, 0]**2) @ self.B.T
        self.invsqrtC = self.B @ np.diag(self.D[:, 0]**-1) @ self.B.T
        self.eigeneval = 0
        self.chin = self.n**0.5 * (1 - 1 / (4 * self.n) + 1 / (21 * self.n**2))
        self.mutants = np.zeros((self.n, self.lmbda))

    def _set_default_callback(self):
        """Sets the default callback function based on the parameter names."""
        if 'Mw' in self._parameters_names or 'kappa' in self._parameters_names:
            self.callback = to_mij
            self.mij_args = ['rho', 'v', 'w', 'kappa', 'sigma', 'h']
            self.mode = 'mt'
            if 'w' not in self._parameters_names and 'v' in self._parameters_names:
                self.callback = to_mij
                self.mode = 'mt_dev'
                self.mij_args = ['rho', 'w', 'kappa', 'sigma', 'h']
            elif 'v' not in self._parameters_names and 'w' not in self._parameters_names:
                self.callback = to_mij
                self.mode = 'mt_dc'
                self.mij_args = ['rho', 'kappa', 'sigma', 'h']
        elif 'F0' in self._parameters_names:
            self.callback = to_rtp
            self.mode = 'force'

    def _setup_caches(self):
        """Initializes caches and logging variables for performance tracking."""
        self.cache_size = 10  # Number of iterations to cache
        self.cache_counter = 0  # To keep track of cached iterations
        self.mutants_cache = np.zeros((self.n + 1, self.lmbda * self.cache_size))  # For storing [parameters + misfit]
        self.mutants_logger_list = pd.DataFrame()
        self.mean_logger_list = pd.DataFrame()

        # Define holder variables for post-processing and plotting
        self._misfit_holder = np.zeros((int(self.lmbda), 1))
        self.fig = None
        self.ax = None

    # Where the mutants are generated ... --------------------------------------------------------------
    def draw_mutants(self):
        """
        Draws mutants from a Gaussian distribution and scatters them across MPI processes.
        
        This function generates `self.lmbda` mutants from a multivariate normal distribution, applies bounds
        and repair methods where necessary, and scatters the mutants to all processes for parallel evaluation.

        Returns
        -------
        None
        """
        if self.rank == 0:
            # Generate mutants on the root process
            self._generate_mutants()

            # Apply repairs and redraws parameter by parameter
            self._repair_and_redraw_mutants()

            # Scatter mutants to all processes
            self._scatter_mutants()
        else:
            # Receive mutants on non-root processes
            self._receive_mutants()

        # Slice the data for the current process
        self.scattered_mutants = self.mutant_slice

        # Increase the counter for misfit evaluations (each mutant will be evaluated once)
        self.counteval += self.lmbda

    def _generate_mutants(self):
        """Generates all `self.lmbda` mutants from a Gaussian distribution."""
        for i in range(self.lmbda):
            mutant = self._draw_single_mutant()
            self.mutants[:, i] = mutant.flatten()

    def _draw_single_mutant(self):
        """Draws a single mutant from the Gaussian distribution."""
        return self.xmean + self.sigma * self.B @ (self.D * np.random.randn(self.n, 1))

    def _repair_and_redraw_mutants(self):
        """Applies repair methods and redraws to all mutants, parameter by parameter."""
        bounds = [0, 10]  # Define the hardcoded bounds
        redraw_counter = 0

        # Iterate over each parameter (column)
        for param_idx in range(self.n):
            # Extract the full array of parameter values for this parameter across all mutants
            param_values = self.mutants[param_idx, :]

            # First, check if any repair or redraw is needed
            if array_in_bounds(param_values, bounds[0], bounds[1]):
                continue  # Skip if all values are already in bounds

            # Check if repair is defined for this parameter
            if self._parameters[param_idx].repair is None:
                # Redraw only the out-of-bounds parameter values across all mutants
                self.mutants[param_idx, :], was_redrawn = self._redraw_param_until_valid(param_values, bounds)
                if was_redrawn:
                    redraw_counter += 1
                print(f'Redrawn {redraw_counter} out-of-bounds mutants for parameter {self._parameters[param_idx].name}')
            else:
                # Apply repair method to the entire array of parameter values across all mutants
                self.mutants[param_idx, :] = self._apply_repair_to_param(param_values, bounds, param_idx)

    def _redraw_param_until_valid(self, param_values, bounds):
        """Redraws the out-of-bounds values of a parameter array until they are within bounds."""
        was_redrawn = False
        for i in range(len(param_values)):
            while not in_bounds(param_values[i], bounds[0], bounds[1]):
                param_values[i] = np.random.randn()  # Redraw only for the specific out-of-bounds value
                was_redrawn = True
        return param_values, was_redrawn

    def _apply_repair_to_param(self, param_values, bounds, param_idx):
        """Applies a repair method to the full array of parameter values if defined."""
        param = self._parameters[param_idx]
        printed_repair = False
        # Keep applying repair until all values in the array are within bounds
        while not array_in_bounds(param_values, bounds[0], bounds[1]):
            if self.verbose_level >= 0 and not printed_repair:
                print(f'Repairing parameter {param.name} using method {param.repair}')
                printed_repair = True
            Repair(param.repair, param_values, self.xmean[param_idx])
        
        return param_values

    def _scatter_mutants(self):
        """Splits and scatters the mutants across processes."""
        self.mutant_lists = np.array_split(self.mutants, self.size, axis=1)
        self.mutant_slice = self.comm.scatter(self.mutant_lists, root=0)

    def _receive_mutants(self):
        """Receives scattered mutants on non-root processes."""
        self.mutant_slice = self.comm.scatter(None, root=0)


    # Where the mutants are evaluated ... --------------------------------------------------------------
    def eval_fitness(self, data, stations, misfit, db_or_greens_list, process=None, wavelet=None, verbose=False):
        """
        Evaluates the misfit for each mutant of the population.

        Parameters
        ----------
        data : mtuq.Dataset
            The data to fit (body waves, surface waves).
        stations : list
            The list of stations.
        misfit : mtuq.WaveformMisfit
            The associated mtuq.Misfit object.
        db_or_greens_list : mtuq.AxiSEM_Client or mtuq.GreensTensorList
            Preprocessed Greens functions or local database (for origin search).
        process : mtuq.ProcessData, optional
            The processing function to apply to the Greens functions.
        wavelet : mtuq.wavelet, optional
            The wavelet to convolve with the Greens functions.
        verbose : bool, optional
            Whether to print debug information.

        Returns
        -------
        numpy.ndarray
            The misfit values for each mutant of the population.
        """
        # Check if the input parameters are valid
        self._check_greens_input_combination(db_or_greens_list, process, wavelet)

        # Use consistent coding style and formatting
        mode = 'db' if isinstance(db_or_greens_list, AxiSEM_Client) else 'greens'

        self._transform_mutants()

        self._generate_sources()

        if mode == 'db':
            return self._eval_fitness_db(data, stations, misfit, db_or_greens_list, process, wavelet)
        elif mode == 'greens':
            return self._eval_fitness_greens(data, stations, misfit, db_or_greens_list)

    def _eval_fitness_db(self, data, stations, misfit, db_or_greens_list, process, wavelet):
        """
        Helper function to evaluate fitness for 'db' mode.

        Parameters
        ----------
        data : mtuq.Dataset
            The data to fit (body waves, surface waves).
        stations : list
            The list of stations.
        misfit : mtuq.WaveformMisfit
            The associated mtuq.Misfit object.
        db_or_greens_list : mtuq.AxiSEM_Client
            Preprocessed Greens functions or local database (for origin search).
        process : mtuq.ProcessData
            The processing function to apply to the Greens functions.
        wavelet : mtuq.wavelet
            The wavelet to convolve with the Greens functions.

        Returns
        -------
        numpy.ndarray
            The misfit values for each mutant of the population.
        """
        # Check if latitude longitude AND depth are absent from the parameters list
        if not any(x in self._parameters_names for x in ['depth', 'latitude', 'longitude']):
            # If so, use the catalog origin, and make one copy per mutant to match the number of mutants.
            if self.rank == 0 and self.verbose_level >= 1:
                print('using catalog origin')
            self.origins = [self.catalog_origin]

            key = self._get_greens_tensors_key(process)

            # Only rank 0 fetches the data from the database
            if self.rank == 0:
                if key not in self._greens_tensors_cache:
                    self._greens_tensors_cache[key] = db_or_greens_list.get_greens_tensors(stations, self.origins)
                    self._greens_tensors_cache[key].convolve(wavelet)
                    self._greens_tensors_cache[key] = self._greens_tensors_cache[key].map(process)
            else:
                self._greens_tensors_cache[key] = None

            # Rank 0 broadcasts the data to the others
            self.local_greens = self.comm.bcast(self._greens_tensors_cache[key], root=0)
            
            self.local_misfit_val = misfit(data, self.local_greens, self.sources)
            self.local_misfit_val = np.asarray(self.local_misfit_val).T
            if self.verbose_level >= 2:
                print('local misfit is :', self.local_misfit_val)  # DEBUG PRINT

            # Gather local misfit values
            self.misfit_val = self.comm.gather(self.local_misfit_val.T, root=0)
            # Broadcast the gathered values and concatenate to return across processes.
            self.misfit_val = self.comm.bcast(self.misfit_val, root=0)
            self.misfit_val = np.asarray(np.concatenate(self.misfit_val)).T
            return self.misfit_val.T
        # If one of the three is present, create a list of origins (one for each mutant), and load the corresponding local greens functions.
        else:
            if self.rank == 0 and self.verbose_level >= 1:
                print('creating new origins list')
            self.create_origins()
            if self.verbose_level >= 2:
                for X in self.origins:
                    print(X)
            # Load, convolve and process local greens function
            start_time = MPI.Wtime()
            self.local_greens = db_or_greens_list.get_greens_tensors(stations, self.origins)
            end_time = MPI.Wtime()
            if self.rank == 0:
                print('Fetching greens tensor: ' + str(end_time-start_time))
            start_time = MPI.Wtime()
            self.local_greens.convolve(wavelet)
            end_time = MPI.Wtime()
            if self.rank == 0:
                print('Convolution: ' + str(end_time-start_time))
            start_time = MPI.Wtime()
            self.local_greens = self.local_greens.map(process)
            end_time = MPI.Wtime()
            if self.rank == 0:
                print('Processing: ' + str(end_time-start_time))
            # DEBUG PRINT to check what is happening on each process: print the number of greens functions loaded on each process
            if self.verbose_level >= 2:
                print('Number of greens functions loaded on process', self.rank, ':', len(self.local_greens))

            # Compute misfit
            start_time = MPI.Wtime()
            self.local_misfit_val = [misfit(data, self.local_greens.select(origin), np.array([self.sources[_i]])) for _i, origin in enumerate(self.origins)]
            self.local_misfit_val = np.asarray(self.local_misfit_val).T[0]
            end_time = MPI.Wtime()

            if self.verbose_level >= 2:
                print('local misfit is :', self.local_misfit_val)  # DEBUG PRINT

            if self.rank == 0:
                print('Misfit: ' + str(end_time-start_time))
            # Gather local misfit values
            self.misfit_val = self.comm.gather(self.local_misfit_val.T, root=0)
            # Broadcast the gathered values and concatenate to return across processes.
            self.misfit_val = self.comm.bcast(self.misfit_val, root=0)
            self.misfit_val = np.asarray(np.concatenate(self.misfit_val))
            return self.misfit_val

    def _eval_fitness_greens(self, data, stations, misfit, db_or_greens_list):
        """
        Helper function to evaluate fitness for 'greens' mode.

        Parameters
        ----------
        data : mtuq.Dataset
            The data to fit (body waves, surface waves).
        stations : list
            The list of stations.
        misfit : mtuq.WaveformMisfit
            The associated mtuq.Misfit object.
        db_or_greens_list : mtuq.GreensTensorList
            Preprocessed Greens functions or local database (for origin search).

        Returns
        -------
        numpy.ndarray
            The misfit values for each mutant of the population.
        """
        # Check if latitude longitude AND depth are absent from the parameters list
        if not any(x in self._parameters_names for x in ['depth', 'latitude', 'longitude']):
            # If so, use the catalog origin, and make one copy per mutant to match the number of mutants.
            if self.rank == 0 and self.verbose_level >= 1:
                print('using catalog origin')
            self.local_greens = db_or_greens_list

            # Get cached arrays using misfit hash and run c_ext_L2.misfit using values from the cache
            # For some reason, the results are not the same if we use the cache from iteration 0. 
            # This is why the cache is create and used from iteration 1 onwards.
            if hasattr(self, 'data_cache'):
                if self.verbose_level >= 1:
                    print('Using cached data')
                hashkey = hash(misfit)
                cache = self.data_cache.get(hashkey)
                data_data = cache['data_data']
                greens_data = cache['greens_data']
                greens_greens = cache['greens_greens']
                groups = cache['groups']
                weights = cache['weights']
                hybrid_norm = cache['hybrid_norm']
                dt = cache['dt']
                padding = cache['padding']
                debug_level = 0
                msg_args = [0, 0, 0]
                self.local_misfit_val = c_ext_L2.misfit(data_data, greens_data, greens_greens, self.sources, groups, weights, hybrid_norm, dt, padding[0], padding[1], debug_level, *msg_args)
            else:
                if self.verbose_level >= 1:
                    print('First iteration, calculating misfit using misfit object')
                self.local_misfit_val = misfit(data, self.local_greens, self.sources)

            self.local_misfit_val = np.asarray(self.local_misfit_val).T
            if self.verbose_level >= 2:
                print('local misfit is :', self.local_misfit_val)

            # Gather local misfit values
            self.misfit_val = self.comm.gather(self.local_misfit_val.T, root=0)
            # Broadcast the gathered values and concatenate to return across processes.
            self.misfit_val = self.comm.bcast(self.misfit_val, root=0)
            self.misfit_val = np.asarray(np.concatenate(self.misfit_val)).T
            return self.misfit_val.T
        # If one of the three is present, issue a warning and break.
        else:
            if self.rank == 0:
                print('WARNING: Greens mode is not compatible with latitude, longitude or depth parameters. Consider using a local Axisem database instead.')
            return None

    def gather_mutants(self, verbose=False):
        """
        Gathers mutants from all processes into the root process. It also uses the datalogger to construct the mutants_logger_list.

        Parameters
        ----------
        verbose : bool, optional
            If set to True, prints the concatenated mutants, their shapes, and types. Default is False.

        Attributes
        ----------
        self.mutants : array
            The gathered and concatenated mutants. This attribute is set to None for non-root processes after gathering.
        self.transformed_mutants : array
            The gathered and concatenated transformed mutants. This attribute is set to None for non-root processes after gathering.
        self.mutants_logger_list : list
            The list to which the datalogger is appended.
        """
        # Printing the mutants on each process, their shapes and types for debugging purposes
        if self.verbose_level >= 2:
            print(self.scattered_mutants, '\n', 'shape is', np.shape(self.scattered_mutants), '\n', 'type is', type(self.scattered_mutants))

        self.mutants = self.comm.gather(self.scattered_mutants, root=0)
        if self.rank == 0:
            self.mutants = np.concatenate(self.mutants, axis=1)
            if self.verbose_level >= 2:
                print(self.mutants, '\n', 'shape is', np.shape(self.mutants), '\n', 'type is', type(self.mutants))  # DEBUG PRINT
        else:
            self.mutants = None

        self.transformed_mutants = self.comm.gather(self.transformed_mutants, root=0)
        if self.rank == 0:
            self.transformed_mutants = np.concatenate(self.transformed_mutants, axis=1)
            if self.verbose_level >= 2:
                print(self.transformed_mutants, '\n', 'shape is', np.shape(self.transformed_mutants), '\n', 'type is', type(self.transformed_mutants))  # DEBUG PRINT
        else:
            self.transformed_mutants = None

        if self.comm.rank == 0:
            current_df = self._datalogger(mean=False)
        # Log the mutants from _datalogger object
        # If self.mutants_logger_list is empty, initialize it with the current DataFrame
            if self.mutants_logger_list.empty:
                self.mutants_logger_list = current_df
            else:
                # Concatenate the current DataFrame to the logger list
                self.mutants_logger_list = pd.concat([self.mutants_logger_list, current_df], ignore_index=True)

    def fitness_sort(self, misfit):
        """
        Sorts the mutants by fitness, and updates the misfit_holder.

        Parameters
        ----------
        misfit : array
            The misfit array to sort the mutants by. Can be the sum of body and surface wave misfits, or the misfit of a single wave type.

        Attributes
        ----------
        self.mutants : array
            The sorted mutants.
        self.transformed_mutants : array
            The sorted transformed mutants.
        self._misfit_holder : array
            The updated misfit_holder. Reset to 0 after sorting.
        """
        if self.rank == 0:
            self.mutants = self.mutants[:, np.argsort(misfit.T)[0]]
            self.transformed_mutants = self.transformed_mutants[:, np.argsort(misfit.T)[0]]
        self._misfit_holder *= 0

    def update_step_size(self):
        """Updates the step size for the CMA-ES algorithm."""
        if self.rank == 0:
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * self.invsqrtC @ (self.mean_diff(self.xmean, self.xold) / self.sigma)

    def update_covariance(self):
        """Updates the covariance matrix for the CMA-ES algorithm."""
        if self.rank == 0:
            ps_norm = np.linalg.norm(self.ps)
            condition = ps_norm / np.sqrt(1 - (1 - self.cs) ** (2 * (self.counteval // self.lmbda + 1))) / self.chin
            threshold = 1.4 + 2 / (self.n + 1)

            if condition < threshold:
                self.hsig = 1
            else:
                self.hsig = 0

            self.dhsig = (1 - self.hsig) * self.cc * (2 - self.cc)

            self.pc = (1 - self.cc) * self.pc + self.hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * self.mean_diff(self.xmean, self.xold) / self.sigma

            artmp = (1 / self.sigma) * self.mean_diff(self.mutants[:, 0:int(self.mu)], self.xold)
            # Old version - from the pureCMA Matlab implementation on Wikipedia
            # self.C = (1-self.c1-self.cmu) * self.C + self.c1 * (self.pc@self.pc.T + (1-self.hsig) * self.cc*(2-self.cc) * self.C) + self.cmu * artmp @ np.diag(self.weights.T[0]) @ artmp.T
            # Old version - from CMA-ES tutorial by Hansen et al. (2016) - https://arxiv.org/pdf/1604.00772.pdf
            # self.C = (1 + self.c1*self.dhsig - self.c1 - self.cmu*np.sum(self.weights)) * self.C + self.c1 * self.pc@self.pc.T + self.cmu * artmp @ np.diag(self.weights.T[0]) @ artmp.T

            # New version - from the purecma python implementation on GitHub - September, 2017, version 3.0.0
            # https://github.com/CMA-ES/pycma/blob/development/cma/purecma.py
            self.C *= 1 + self.c1 * self.dhsig - self.c1 - self.cmu * sum(self.weights)  # Discount factor
            self.C += self.c1 * self.pc @ self.pc.T  # Rank one update (pc.pc^T is a matrix of rank 1)
            self.C += self.cmu * artmp @ np.diag(self.weights.T[0]) @ artmp.T  # Rank mu update, supported by the mu best individuals

            # Adapt step size
            # We do sigma_i+1 = sigma_i * exp((cs/damps)*(||ps||/E[N(0,I)]) - 1) only now as artmp needs sigma_i
            self.sigma = self.sigma * np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chin - 1))

            if self.counteval - self.eigeneval > self.lmbda / (self.c1 + self.cmu) / self.n / 10:
                self.eigeneval = self.counteval
                self.C = np.triu(self.C) + np.triu(self.C, 1).T
                self.D, self.B = np.linalg.eig(self.C)
                self.D = np.array([self.D]).T
                self.D = np.sqrt(self.D)
                self.invsqrtC = self.B @ np.diag(self.D[:, 0]**-1) @ self.B.T

        self.iteration = self.counteval // self.lmbda

    def update_mean(self):
        """Updates the mean of the parameter distribution."""
        if self.rank == 0:
            self.xold = self.xmean.copy()
            self.xmean = np.dot(self.mutants[:, 0:len(self.weights)], self.weights)
            for _i, param in enumerate(self._parameters):
                if param.repair == 'wrapping':
                    print('computing wrapped mean for parameter:', param.name)
                    self.xmean[_i] = self.circular_mean(_i)
            
            # Update the mean datalogger
            current_mean_df = self._datalogger(mean=True)

            # If self.mean_logger_list is empty, initialize it with the current DataFrame
            if self.mean_logger_list.empty:
                self.mean_logger_list = current_mean_df
            else:
                # Concatenate the current DataFrame to the logger list
                self.mean_logger_list = pd.concat([self.mean_logger_list, current_mean_df], ignore_index=True)

    def circular_mean(self, id):
        """
        Compute the circular mean on the "id"th parameter. Ciruclar mean allows to compute mean of the samples on a periodic space.

        Parameters
        ----------
        id : int
            The index of the parameter.

        Returns
        -------
        float
            The circular mean of the parameter.
        """
        param = self.mutants[id]
        a = linear_transform(param, 0, 360) - 180
        mean = np.rad2deg(np.arctan2(np.sum(np.sin(np.deg2rad(a[range(int(self.mu))])) * self.weights.T), np.sum(np.cos(np.deg2rad(a[range(int(self.mu))])) * self.weights.T))) + 180
        mean = inverse_linear_transform(mean, 0, 360)
        return mean

    def smallestAngle(self, targetAngles, currentAngles) -> np.ndarray:
        """
        Calculates the smallest angle (in degrees) between two given sets of angles. It computes the difference between the target and current angles, making sure the result stays within the range [0, 360). If the resulting difference is more than 180, it is adjusted to go in the shorter, negative direction.

        Parameters
        ----------
        targetAngles : np.ndarray
            An array containing the target angles in degrees.
        currentAngles : np.ndarray
            An array containing the current angles in degrees.

        Returns
        -------
        np.ndarray
            An array containing the smallest difference in degrees between the target and current angles.
        """
        # Subtract the angles, constraining the value to [0, 360)
        diffs = (targetAngles - currentAngles) % 360

        # If we are more than 180 we're taking the long way around.
        # Let's instead go in the shorter, negative direction
        diffs[diffs > 180] = -(360 - diffs[diffs > 180])
        return diffs

    def mean_diff(self, new, old):
        """
        Computes the mean change and applies circular difference for wrapped repair methods.

        Parameters
        ----------
        new : np.ndarray
            The new mean values.
        old : np.ndarray
            The old mean values.

        Returns
        -------
        np.ndarray
            The mean difference.
        """
        diff = new - old
        for _i, param in enumerate(self._parameters):
            if param.repair == 'wrapping':
                angular_diff = self.smallestAngle(linear_transform(new[_i], 0, 360), linear_transform(old[_i], 0, 360))
                angular_diff = inverse_linear_transform(angular_diff, 0, 360)
                diff[_i] = angular_diff
        return diff

    def create_origins(self):
        """Creates a list of origins for each mutant."""
        # Check which of the three origin modifiers are in the parameters
        if 'depth' in self._parameters_names:
            depth = self.transformed_mutants[self._parameters_names.index('depth')]
        else:
            depth = self.catalog_origin.depth_in_m
        if 'latitude' in self._parameters_names:
            latitude = self.transformed_mutants[self._parameters_names.index('latitude')]
        else:
            latitude = self.catalog_origin.latitude
        if 'longitude' in self._parameters_names:
            longitude = self.transformed_mutants[self._parameters_names.index('longitude')]
        else:
            longitude = self.catalog_origin.longitude
        
        self.origins = []
        for i in range(len(self.scattered_mutants[0])):
            self.origins += [self.catalog_origin.copy()]
            if 'depth' in self._parameters_names:
                setattr(self.origins[-1], 'depth_in_m', depth[i])
            if 'latitude' in self._parameters_names:
                setattr(self.origins[-1], 'latitude', latitude[i])
            if 'longitude' in self._parameters_names:
                setattr(self.origins[-1], 'longitude', longitude[i])

    def return_candidate_solution(self, id=None):
        """
        Returns the candidate solution for a given mutant.

        Parameters
        ----------
        id : int, optional
            The index of the mutant. If None, returns the mean solution.

        Returns
        -------
        tuple
            The transformed mean and the origins.
        """
        # Only required on rank 0
        if self.rank == 0:
            if not id == None:
                return_x = np.array([self.mutants[:, id]]).T
            else:
                return_x = self.xmean
            self.transformed_mean = np.zeros_like(return_x)
            for _i, param in enumerate(self._parameters):
                # Print paramter scaling if verbose
                if param.scaling == 'linear':
                    self.transformed_mean[_i] = linear_transform(return_x[_i], param.lower_bound, param.upper_bound)
                elif param.scaling == 'log':
                    self.transformed_mean[_i] = logarithmic_transform(return_x[_i], param.lower_bound, param.upper_bound)
                else:
                    raise ValueError('Unrecognized scaling, must be linear or log')
                # Apply optional projection operator to each parameter
                if not param.projection is None:
                    self.transformed_mean[_i] = np.asarray(list(map(param.projection, self.transformed_mean[_i])))

            # Check which of the three origin modifiers are in the parameters
            if 'depth' in self._parameters_names:
                depth = self.transformed_mean[self._parameters_names.index('depth')]
            else:
                depth = self.catalog_origin.depth_in_m
            if 'latitude' in self._parameters_names:
                latitude = self.transformed_mean[self._parameters_names.index('latitude')]
            else:
                latitude = self.catalog_origin.latitude
            if 'longitude' in self._parameters_names:
                longitude = self.transformed_mean[self._parameters_names.index('longitude')]
            else:
                longitude = self.catalog_origin.longitude
            
            self.origins = []
            for i in range(len(self.transformed_mean[0])):
                self.origins += [self.catalog_origin.copy()]
                if 'depth' in self._parameters_names:
                    setattr(self.origins[-1], 'depth_in_m', depth[i])
                if 'latitude' in self._parameters_names:
                    setattr(self.origins[-1], 'latitude', latitude[i])
                if 'longitude' in self._parameters_names:
                    setattr(self.origins[-1], 'longitude', longitude[i])

            return self.transformed_mean, self.origins

    def _datalogger(self, mean=False):
        """
        Logs the coordinates and misfit values of the mutants.

        Parameters
        ----------
        mean : bool, optional
            If True, logs the mean coordinates. Default is False.

        Returns
        -------
        MTUQDataFrame
            The logged data.
        """
        if self.rank == 0:
            if mean == False:
                coordinates = self.transformed_mutants.T
                misfit_values = self._misfit_holder
                results = np.hstack((coordinates, misfit_values))
                columns_labels = self._parameters_names + ['misfit']

            if mean == True:
                self.transformed_mean = np.zeros_like(self.xmean)
                for _i, param in enumerate(self._parameters):
                    if param.scaling == 'linear':
                        self.transformed_mean[_i] = linear_transform(self.xmean[_i], param.lower_bound, param.upper_bound)
                    elif param.scaling == 'log':
                        self.transformed_mean[_i] = logarithmic_transform(self.xmean[_i], param.lower_bound, param.upper_bound)
                    else:
                        raise ValueError('Unrecognized scaling, must be linear or log')
                    # Apply optional projection operator to each parameter
                    if not param.projection is None:
                        self.transformed_mean[_i] = np.asarray(list(map(param.projection, self.transformed_mean[_i])))

                results = self.transformed_mean.T
                columns_labels = self._parameters_names

            da = pd.DataFrame(
                data=results,
                columns=columns_labels
            )
            return MTUQDataFrame(da)

    def _prep_and_cache_C_arrays(self, data, greens, misfit, stations):
        """
        Prepares and caches C compatible arrays for the misfit function evaluation.

        It is responsible for preparing the data arrays for the inversion, in a format expected by the lower-level 
        c-code for misfit evaluation. Mostly copy-pasted from mtuq.misfit.waveform.level2

        Only used when the mode is 'greens'.

        Parameters
        ----------
        data : mtuq.Dataset
            The data to fit (body waves, surface waves).
        greens : mtuq.GreensTensorList
            Preprocessed Greens functions or local database (for origin search).
        misfit : mtuq.WaveformMisfit
            The associated mtuq.Misfit object.
        stations : list
            The list of stations.
        """
        from mtuq.misfit.waveform.level2 import _get_time_sampling, _get_stations, _get_components, _get_weights, _get_groups, _get_data, _get_greens, _get_padding, _autocorr_1, _autocorr_2, _corr_1_2
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
            if self.mode in ['mt', 'mt_dc', 'mt_dev']:
                for g in greens:
                    g.include_mt = True
                    g.include_force = False
            elif self.mode == 'force':
                for g in greens:
                    g.include_mt = False
                    g.include_force = True
            else:
                raise ValueError('Invalid mode. Supported modes: "mt", "mt_dc", "mt_dev", "force".')

            # collapse main structures into NumPy arrays
            data = _get_data(data, stations, components)
            greens = _get_greens(greens, stations, components)

            # cross-correlate data and synthetics
            padding = _get_padding(misfit.time_shift_min, misfit.time_shift_max, dt)
            data_data = _autocorr_1(data)
            greens_greens = _autocorr_2(greens, padding)
            greens_data = _corr_1_2(data, greens, padding)

            if misfit.norm == 'hybrid':
                hybrid_norm = 1
            else:
                hybrid_norm = 0

            # collect message attributes
            try:
                msg_args = [getattr(msg_handle, attrib) for attrib in ['start', 'stop', 'percent']]
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
            if self.verbose_level >= 2:
                print('Data arrays already cached. Nothing to do here.')
            pass

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

        if self.rank == 0:
            # Check Solve inputs
            self._check_Solve_inputs(data_list, stations, misfit_list, process_list, db_or_greens_list, max_iter, wavelet, plot_interval, iter_count)

        # Handling of the misfit weights. If not provided, equal weights are used, otherwise the weights are used to derive percentages.
        if misfit_weights is None:
            misfit_weights = [1.0] * len(data_list)
        elif len(misfit_weights) != len(data_list):
            raise ValueError('Length of misfit_weights must match the length of data_list.')

        total_weight = sum(misfit_weights)
        if total_weight == 0:
            raise ValueError('Sum of weights cannot be zero.')

        misfit_weights = [w / total_weight for w in misfit_weights]

        for i in range(max_iter):
            if self.rank == 0:
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
                    if i == 1 and type(current_misfit) == WaveformMisfit:
                        raw_greens_to_cache = copy.deepcopy(greens)
                        raw_data_to_cache = copy.deepcopy(current_data)
                        self._prep_and_cache_C_arrays(raw_data_to_cache, raw_greens_to_cache, current_misfit, stations)
                    misfit_values = self.eval_fitness(current_data, stations, current_misfit, greens, **kwargs)

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
                if self.rank == 0:
                    plot_mean_waveforms(self, data_list, process_list, misfit_list, stations, db_or_greens_list)
                    if self.mode in ['mt', 'mt_dc', 'mt_dev']:
                        print('Plotting results for iteration %d\n' % (i + iter_count))
                        result = self.mutants_logger_list

                        # Handling the mean solution
                        if self.mode == 'mt':
                            V, W = self._datalogger(mean=True)['v'], self._datalogger(mean=True)['w']
                        elif self.mode == 'mt_dev':
                            V = self._datalogger(mean=True)['v']
                            W = 0
                        elif self.mode == 'mt_dc':
                            V = W = 0

                    # If mode is mt, mt_dev or mt_dc, plot the misfit map
                    if self.mode in ['mt', 'mt_dev', 'mt_dc']:
                        plot_combined(self.event_id + '_combined_misfit_map.png', result, colormap='viridis', best_vw=(V, W))
                    elif self.mode == 'force':
                        print('Plotting results for iteration %d\n' % (i + iter_count))
                        result = self.mutants_logger_list
                        plot_misfit_force(self.event_id + '_misfit_map.png', result, colormap='viridis', backend=_plot_force_matplotlib, plot_type='colormesh', best_force=self.return_candidate_solution()[0][1::])


    def _transform_mutants(self):
        """
        Transforms local mutants on each process based on the parameters scaling and projection settings.

        For each parameter, depending on its scaling setting ('linear' or 'log'), 
        it applies a transformation to the corresponding elements of scattered_mutants.
        If a projection is specified, it applies this projection to the transformed values.

        Attributes
        ----------
        scattered_mutants : np.ndarray
            A 2D numpy array with the original mutant data. When MPI is used, correspond to the local mutants on each process.
        _parameters : list
            A list of Parameter objects, each with attributes 'scaling', 'lower_bound', 'upper_bound', 
            and 'projection' specifying how to transform the corresponding scattered_mutants.

        Raises
        ------
        ValueError
            If an unrecognized scaling is provided.
        """
        self.transformed_mutants = np.zeros_like(self.scattered_mutants)
        for i, param in enumerate(self._parameters):
            if param.scaling == 'linear':
                self.transformed_mutants[i] = linear_transform(self.scattered_mutants[i], param.lower_bound, param.upper_bound)
            elif param.scaling == 'log':
                self.transformed_mutants[i] = logarithmic_transform(self.scattered_mutants[i], param.lower_bound, param.upper_bound)
            else:
                raise ValueError('Unrecognized scaling, must be linear or log')
            if param.projection is not None:
                self.transformed_mutants[i] = np.asarray(list(map(param.projection, self.transformed_mutants[i])))

    def _generate_sources(self):
        """
        Generates sources by calling the callback function on transformed data according to the set mode.

        Depending on the mode, the method selects a subset of transformed mutants, possibly extending
        it with zero-filled columns at specific positions, and then passes the processed data to the
        callback function. The results are stored in a contiguous NumPy array in self.sources.

        Raises
        ------
        ValueError
            If an unsupported mode is provided.

        Attributes
        ----------
        mode : str
            A string representing the mode of operation, which can be 'mt', 'mt_dev', 'mt_dc', or 'force'.
        transformed_mutants : np.ndarray
            A 2D numpy array that contains the transformed data to be processed.
        callback : callable
            A callable that is used to process the data.
        """
        # Mapping between modes and respective slices or insertion positions for processing.
        mode_to_indices = {
            'mt': (0, 6),  # For 'mt', a slice from the first 6 elements of transformed_mutants is used.
            'mt_dev': (0, 5, 2),  # For 'mt_dev', a zero column is inserted at position 2 after slicing the first 5 elements.
            'mt_dc': (0, 4, 1, 2),  # For 'mt_dc', zero columns are inserted at positions 1 and 2 after slicing the first 4 elements.
            'force': (0, 3),  # For 'force', a slice from the first 3 elements of transformed_mutants is used.
        }

        # Check the mode's validity. Raise an error if the mode is unsupported.
        if self.mode not in mode_to_indices:
            raise ValueError(f'Invalid mode: {self.mode}')

        # Get the slice or insertion positions based on the current mode.
        indices = mode_to_indices[self.mode]

        # If the mode is 'mt' or 'force', take a slice from transformed_mutants and pass it to the callback.
        if self.mode in ['mt', 'force']:
            self.sources = np.ascontiguousarray(self.callback(*self.transformed_mutants[indices[0]:indices[1]]))
        else:
            # For 'mt_dev' and 'mt_dc' modes, insert zeros at specific positions after slicing transformed_mutants.
            self.extended_mutants = self.transformed_mutants[indices[0]:indices[1]]
            for insertion_index in indices[2:]:
                self.extended_mutants = np.insert(self.extended_mutants, insertion_index, 0, axis=0)
            # Pass the processed data to the callback, and save the result as a contiguous array in self.sources.
            self.sources = np.ascontiguousarray(self.callback(*self.extended_mutants[0:6]))

    def _get_greens_tensors_key(self, process):
        """
        Gets the body-wave or surface-wave key for the GreensTensors object from the ProcessData object.

        Parameters
        ----------
        process : ProcessData
            The ProcessData object.

        Returns
        -------
        str
            The body-wave or surface-wave key.
        """
        return process.window_type

    def _check_greens_input_combination(self, db, process, wavelet):
        """
        Checks the validity of the given parameters.

        Raises a ValueError if the database object is not an AxiSEM_Client or GreensTensorList, 
        or if the process function and wavelet are not defined when the database object is an AxiSEM_Client.

        Parameters
        ----------
        db : AxiSEM_Client or GreensTensorList
            The database object to check, expected to be an instance of either AxiSEM_Client or GreensTensorList.
        process : ProcessData
            The process function to be used if the database is an AxiSEM_Client.
        wavelet : mtuq.wavelet
            The wavelet to be used if the database is an AxiSEM_Client.

        Raises
        ------
        ValueError
            If the input combination of db, process, and wavelet is invalid.
        """
        if not isinstance(db, (AxiSEM_Client, GreensTensorList)):
            raise ValueError('database must be either an AxiSEM_Client object or a GreensTensorList object')
        if isinstance(db, AxiSEM_Client) and (process is None or wavelet is None):
            raise ValueError('process_function and wavelet must be specified if database is an AxiSEM_Client')

    def _check_Solve_inputs(self, data_list, stations, misfit_list, process_list, db_or_greens_list, max_iter=100, wavelet=None, plot_interval=10, iter_count=0, **kwargs):
        """
        Checks the validity of input arguments for the Solve method.

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
        **kwargs
            Additional keyword arguments passed to eval_fitness method.

        Raises
        ------
        ValueError
            If any of the inputs are invalid.
        """
        if not isinstance(data_list, list):
            if isinstance(data_list, Dataset):
                data_list = [data_list]
            else:
                raise ValueError('`data_list` should be a list of mtuq Dataset or an array containing polarities.')
        if not isinstance(stations, list):
            raise ValueError('`stations` should be a list of mtuq Station objects.')
        if not isinstance(misfit_list, list):
            if isinstance(misfit_list, PolarityMisfit) or isinstance(misfit_list, Misfit):
                misfit_list = [misfit_list]
            else:
                raise ValueError('`misfit_list` should be a list of mtuq Misfit objects.')
        if not isinstance(process_list, list):
            if isinstance(process_list, ProcessData):
                process_list = [process_list]
            else:
                raise ValueError('`process_list` should be a list of mtuq ProcessData objects.')
        if not isinstance(db_or_greens_list, list):
            if isinstance(db_or_greens_list, AxiSEM_Client) or isinstance(db_or_greens_list, GreensTensorList):
                db_or_greens_list = [db_or_greens_list]
            else:
                raise ValueError('`db_or_greens_list` should be a list of either mtuq AxiSEM_Client or GreensTensorList objects.')
        if not isinstance(max_iter, int):
            raise ValueError('`max_iter` should be an integer.')
        if any(isinstance(db, AxiSEM_Client) for db in db_or_greens_list) and wavelet is None:
            raise ValueError('wavelet must be specified if database is an AxiSEM_Client')
        if not isinstance(plot_interval, int):
            raise ValueError('`plot_interval` should be an integer.')
        if iter_count is not None and not isinstance(iter_count, int):
            raise ValueError('`iter_count` should be an integer or None.')

    def _get_data_norm(self, data, misfit):
        """
        Computes the norm of the data using the calculate_norm_data function.

        Parameters
        ----------
        data : mtuq.Dataset
            The evaluated processed data.
        misfit : mtuq.Misfit
            The misfit object used to evaluate the data object.

        Returns
        -------
        float
            The norm of the data.
        """
        # If misfit type is Polarity misfit, use the sum of the absolute values of the data as number of used stations.
        if isinstance(misfit, PolarityMisfit):
            return np.sum(np.abs(data))
        # Else, use the calculate_norm_data function.
        else:
            if isinstance(misfit.time_shift_groups, str):
                components = list(misfit.time_shift_groups)
            elif isinstance(misfit.time_shift_groups, list):
                components = list(''.join(misfit.time_shift_groups))
            
            return calculate_norm_data(data, misfit.norm, components)
