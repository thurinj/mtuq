from mtuq.stochastic_sampling.cmaes_utils import linear_transform, inverse_linear_transform, logarithmic_transform, in_bounds, array_in_bounds, Repair

class CMAESFitnessEvaluation:
    def __init__(self, parameters, xmean, sigma, B, D, n, lmbda, size, rank, comm, verbose_level, callback, catalog_origin, greens_tensors_cache):
        self.parameters = parameters
        self.xmean = xmean
        self.sigma = sigma
        self.B = B
        self.D = D
        self.n = n
        self.lmbda = lmbda
        self.size = size
        self.rank = rank
        self.comm = comm
        self.verbose_level = verbose_level
        self.callback = callback
        self.catalog_origin = catalog_origin
        self.greens_tensors_cache = greens_tensors_cache
        self.counteval = 0
        self.transformed_mutants = None
        self.sources = None
        self.origins = None
        self.local_greens = None
        self.local_misfit_val = None
        self.misfit_val = None
        self._misfit_holder = None

    def eval_fitness(self, data, stations, misfit, db_or_greens_list, process=None, wavelet=None, verbose=False):
        self._check_greens_input_combination(db_or_greens_list, process, wavelet)
        mode = 'db' if isinstance(db_or_greens_list, AxiSEM_Client) else 'greens'
        self._transform_mutants()
        self._generate_sources()

        if mode == 'db':
            if not any(x in self.parameters_names for x in ['depth', 'latitude', 'longitude']):
                if self.rank == 0 and self.verbose_level >= 1:
                    print('using catalog origin')
                self.origins = [self.catalog_origin]
                key = self._get_greens_tensors_key(process)
                if self.rank == 0:
                    if key not in self.greens_tensors_cache:
                        self.greens_tensors_cache[key] = db_or_greens_list.get_greens_tensors(stations, self.origins)
                        self.greens_tensors_cache[key].convolve(wavelet)
                        self.greens_tensors_cache[key] = self.greens_tensors_cache[key].map(process)
                else:
                    self.greens_tensors_cache[key] = None
                self.local_greens = self.comm.bcast(self.greens_tensors_cache[key], root=0)
                self.local_misfit_val = misfit(data, self.local_greens, self.sources)
                self.local_misfit_val = np.asarray(self.local_misfit_val).T
                if self.verbose_level >= 2:
                    print("local misfit is :", self.local_misfit_val)
                self.misfit_val = self.comm.gather(self.local_misfit_val.T, root=0)
                self.misfit_val = self.comm.bcast(self.misfit_val, root=0)
                self.misfit_val = np.asarray(np.concatenate(self.misfit_val)).T
                return self.misfit_val.T
            else:
                if self.rank == 0 and self.verbose_level >= 1:
                    print('creating new origins list')
                self.create_origins()
                if self.verbose_level >= 2:
                    for X in self.origins:
                        print(X)
                start_time = MPI.Wtime()
                self.local_greens = db_or_greens_list.get_greens_tensors(stations, self.origins)
                end_time = MPI.Wtime()
                if self.rank == 0:
                    print("Fetching greens tensor: " + str(end_time-start_time))
                start_time = MPI.Wtime()
                self.local_greens.convolve(wavelet)
                end_time = MPI.Wtime()
                if self.rank == 0:
                    print("Convolution: " + str(end_time-start_time))
                start_time = MPI.Wtime()
                self.local_greens = self.local_greens.map(process)
                end_time = MPI.Wtime()
                if self.rank == 0:
                    print("Processing: " + str(end_time-start_time))
                if self.verbose_level >= 2:
                    print("Number of greens functions loaded on process", self.rank, ":", len(self.local_greens))
                start_time = MPI.Wtime()
                self.local_misfit_val = [misfit(data, self.local_greens.select(origin), np.array([self.sources[_i]])) for _i, origin in enumerate(self.origins)]
                self.local_misfit_val = np.asarray(self.local_misfit_val).T[0]
                end_time = MPI.Wtime()
                if self.verbose_level >= 2:
                    print("local misfit is :", self.local_misfit_val)
                if self.rank == 0:
                    print("Misfit: " + str(end_time-start_time))
                self.misfit_val = self.comm.gather(self.local_misfit_val.T, root=0)
                self.misfit_val = self.comm.bcast(self.misfit_val, root=0)
                self.misfit_val = np.asarray(np.concatenate(self.misfit_val))
                return self.misfit_val
        elif mode == 'greens':
            if not any(x in self.parameters_names for x in ['depth', 'latitude', 'longitude']):
                if self.rank == 0 and self.verbose_level >= 1:
                    print('using catalog origin')
                self.local_greens = db_or_greens_list
                if hasattr(self, 'data_cache'):
                    if self.verbose_level >= 1:
                        print("Using cached data")
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
                        print("First iteration, calculating misfit using misfit object")
                    self.local_misfit_val = misfit(data, self.local_greens, self.sources)
                self.local_misfit_val = np.asarray(self.local_misfit_val).T
                if self.verbose_level >= 2:
                    print("local misfit is :", self.local_misfit_val)
                self.misfit_val = self.comm.gather(self.local_misfit_val.T, root=0)
                self.misfit_val = self.comm.bcast(self.misfit_val, root=0)
                self.misfit_val = np.asarray(np.concatenate(self.misfit_val)).T
                return self.misfit_val.T
            else:
                if self.rank == 0:
                    print('WARNING: Greens mode is not compatible with latitude, longitude or depth parameters. Consider using a local Axisem database instead.')
                return None

    def _transform_mutants(self):
        self.transformed_mutants = np.zeros_like(self.scattered_mutants)
        for i, param in enumerate(self.parameters):
            if param.scaling == 'linear':
                self.transformed_mutants[i] = linear_transform(self.scattered_mutants[i], param.lower_bound, param.upper_bound)
            elif param.scaling == 'log':
                self.transformed_mutants[i] = logarithmic_transform(self.scattered_mutants[i], param.lower_bound, param.upper_bound)
            else:
                raise ValueError("Unrecognized scaling, must be linear or log")
            if param.projection is not None:
                self.transformed_mutants[i] = np.asarray(list(map(param.projection, self.transformed_mutants[i])))

    def _generate_sources(self):
        mode_to_indices = {
            'mt': (0, 6),
            'mt_dev': (0, 5, 2),
            'mt_dc': (0, 4, 1, 2),
            'force': (0, 3),
        }
        if self.mode not in mode_to_indices:
            raise ValueError(f'Invalid mode: {self.mode}')
        indices = mode_to_indices[self.mode]
        if self.mode in ['mt', 'force']:
            self.sources = np.ascontiguousarray(self.callback(*self.transformed_mutants[indices[0]:indices[1]]))
        else:
            self.extended_mutants = self.transformed_mutants[indices[0]:indices[1]]
            for insertion_index in indices[2:]:
                self.extended_mutants = np.insert(self.extended_mutants, insertion_index, 0, axis=0)
            self.sources = np.ascontiguousarray(self.callback(*self.extended_mutants[0:6]))

    def _get_greens_tensors_key(self, process):
        return process.window_type

    def _check_greens_input_combination(self, db, process, wavelet):
        if not isinstance(db, (AxiSEM_Client, GreensTensorList)):
            raise ValueError("database must be either an AxiSEM_Client object or a GreensTensorList object")
        if isinstance(db, AxiSEM_Client) and (process is None or wavelet is None):
            raise ValueError("process_function and wavelet must be specified if database is an AxiSEM_Client")

    def create_origins(self):
        if 'depth' in self.parameters_names:
            depth = self.transformed_mutants[self.parameters_names.index('depth')]
        else:
            depth = self.catalog_origin.depth_in_m
        if 'latitude' in self.parameters_names:
            latitude = self.transformed_mutants[self.parameters_names.index('latitude')]
        else:
            latitude = self.catalog_origin.latitude
        if 'longitude' in self.parameters_names:
            longitude = self.transformed_mutants[self.parameters_names.index('longitude')]
        else:
            longitude = self.catalog_origin.longitude
        self.origins = []
        for i in range(len(self.scattered_mutants[0])):
            self.origins += [self.catalog_origin.copy()]
            if 'depth' in self.parameters_names:
                setattr(self.origins[-1], 'depth_in_m', depth[i])
            if 'latitude' in self.parameters_names:
                setattr(self.origins[-1], 'latitude', latitude[i])
            if 'longitude' in self.parameters_names:
                setattr(self.origins[-1], 'longitude', longitude[i])
