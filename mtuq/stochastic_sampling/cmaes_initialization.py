class CMAESInitialization:
    def __init__(self, parameters_list, lmbda, origin, callback_function, event_id, verbose_level):
        self._initialize_mpi_communicator()
        self._initialize_logging(event_id, verbose_level)
        self._initialize_parameters(parameters_list, lmbda, origin, callback_function)
        self._setup_caches()

    def _initialize_mpi_communicator(self):
        self.rank = 0
        self.size = 1
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def _initialize_logging(self, event_id, verbose_level):
        self.event_id = event_id
        self.verbose_level = verbose_level
        if self.rank == 0:
            print(f'Initializing CMA-ES inversion for event {self.event_id}')

    def _initialize_parameters(self, parameters_list, lmbda, origin, callback_function):
        self._parameters = parameters_list
        self._parameters_names = [parameter.name for parameter in parameters_list]
        self.n = len(self._parameters)

        if lmbda is None:
            self.lmbda = int(4 + np.floor(3 * np.log(self.n)))
        else:
            self.lmbda = lmbda

        if self.size > self.lmbda:
            raise ValueError(f'Number of MPI processes ({self.size}) exceeds population size ({self.lmbda})')

        if not isinstance(origin, Origin):
            raise ValueError("The 'origin' parameter must be an instance of mtuq.event.Origin. Please provide a valid object to be used as catalog origin.")

        self.catalog_origin = origin

        self.callback = callback_function
        if self.callback is None:
            self._set_default_callback()

        self.xmean = np.asarray([[param.initial for param in self._parameters]]).T
        self.sigma = 0.5
        self.iteration = 0
        self.counteval = 0
        self._greens_tensors_cache = {}

        self.mu = np.floor(self.lmbda / 2)
        a = 1
        self.weights = np.array([np.log(self.mu + a) - np.log(np.arange(1, self.mu + 1))]).T
        self.weights /= sum(self.weights)
        self.mueff = sum(self.weights)**2 / sum(self.weights**2)

        self.cs = (self.mueff + 2) / (self.n + self.mueff + 5)
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.n + 1)) - 1) + self.cs

        self.cc = (4 + self.mueff / self.n) / (self.n + 4 + 2 * self.mueff / self.n)
        self.acov = 2
        self.c1 = self.acov / ((self.n + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, self.acov * (self.mueff - 2 + 1 / self.mueff) / ((self.n + 2)**2 + self.acov * self.mueff / 2))

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
        self.cache_size = 10
        self.cache_counter = 0
        self.mutants_cache = np.zeros((self.n + 1, self.lmbda * self.cache_size))
        self.mutants_logger_list = pd.DataFrame()
        self.mean_logger_list = pd.DataFrame()

        self._misfit_holder = np.zeros((int(self.lmbda), 1))
        self.fig = None
        self.ax = None
