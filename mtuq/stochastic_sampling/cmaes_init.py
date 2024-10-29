import numpy as np
from mpi4py import MPI
import pandas as pd
from mtuq.event import Origin
from mtuq.util.math import to_mij, to_rtp

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

    # Initialize the misfit holder which will store the misfit values for each mutant for a given iteration
    self._misfit_holder = np.zeros((int(self.lmbda), 1))

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
        _set_default_callback(self)

    # Initialize parameters tied to the CMA-ES algorithm
    self.xmean = np.asarray([[param.initial for param in self._parameters]]).T
    self.sigma = 1.67  # Default initial Gaussian variance for all parameters
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

def _initialize_ipop(self, max_restarts: int = 10, lambda_increase_factor: float = 2.0, patience: int = 20):
    """
    Initializes the IPOP (Increasing Population) strategy for the CMA-ES algorithm.

    Parameters
    ----------
    max_restarts : int, optional
        Maximum number of restarts allowed. Default is 10.
    lambda_increase_factor : float, optional
        Factor by which to increase lambda upon each restart. Default is 2.0.
    patience : int, optional
        Number of iterations to wait without improvement before restarting. Default is 20.
    """
    self.ipop = True
    self.ipop_terminated = False
    self.max_restarts = max_restarts
    self.lambda_increase_factor = lambda_increase_factor
    self.patience = patience
    self.current_restarts = 0
    self.no_improve_counter = 0
    self.best_misfit = np.inf
    self.best_solution = None
    self.best_origins = None

    if self.rank == 0:
        print("IPOP strategy initialized with the following parameters:")
        print(f"  Max Restarts: {self.max_restarts}")
        print(f"  Lambda Increase Factor: {self.lambda_increase_factor}")
        print(f"  Patience: {self.patience} iterations without improvement")

def _restart_ipop(self):
    """
    Handles the restart mechanism for the IPOP strategy by increasing the population size
    and reinitializing CMA-ES parameters.
    """
    if self.current_restarts >= self.max_restarts:
        if self.rank == 0:
            print(f"Maximum number of restarts ({self.max_restarts}) reached. Terminating optimization.")
        raise StopIteration("IPOP-CMA-ES: Maximum restarts reached.")

    # Increase population size
    new_lambda = int(self.lmbda * self.lambda_increase_factor)
    if self.rank == 0:
        print(f"Restarting CMA-ES with increased population size: {new_lambda}")

    # Update lambda
    self.lmbda = new_lambda

    self._misfit_holder = np.zeros((int(self.lmbda), 1))

    # Reinitialize CMA-ES parameters
    self.iteration = 0
    # self.counteval = 0
    self.mutants = np.zeros((self.n, self.lmbda))

    # Reset step-size and covariance-related variables
    restart_from_best = False
    if restart_from_best:
        self.xmean = np.asarray([self.best_solution.copy()]).T
    else:
        # restart at random
        self.xmean = np.random.uniform(0, 10, (self.n, 1))
        
    self.sigma = 1.67  # You might want to adjust this based on your problem
    self.B = np.eye(self.n, self.n)
    self.D = np.ones((self.n, 1))
    self.C = self.B @ np.diag(self.D[:, 0]**2) @ self.B.T
    self.invsqrtC = self.B @ np.diag(self.D[:, 0]**-1) @ self.B.T
    self.ps = np.zeros_like(self.xmean)
    self.pc = np.zeros_like(self.xmean)
    self.eigeneval = 0

    # Update weights and related parameters based on new lambda
    self.mu = np.floor(self.lmbda / 2)
    a = 1  # Use 1 in publication
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

    self.current_restarts += 1
    self.no_improve_counter = 0  # Reset patience counter

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
    """Initializes pandas DataFrames for logging. Used for post-processing and plotting."""
    self.mutants_logger_list = pd.DataFrame()
    self.mean_logger_list = pd.DataFrame()

    # Define holder variables for plotting
    self.fig = None
    self.ax = None