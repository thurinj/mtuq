import numpy as np
import pandas as pd
from mtuq.util.cmaes import *
from mtuq.util.math import to_mij, to_rtp, to_rho
from mtuq.grid.force import to_force
import mpi4py.MPI as MPI
from mtuq.greens_tensor import GreensTensor
from mtuq import MTUQDataFrame
from mtuq.grid.moment_tensor import UnstructuredGrid, to_mt
from mtuq.grid.force import UnstructuredGrid, to_force
from mtuq.graphics import plot_data_greens2, plot_data_greens1

# class CMA_ES(object):


class parallel_CMA_ES(object):

    def __init__(self, parameters_list, lmbda=None, data=None, GFclient=None, origin=None, callback_function=None, event_id=''):
        '''
        parallel_CMA_ES class

        CMA-ES class for moment tensor and force inversion. The class accept a list of `CMAESParameters` objects containing the options and tunning of each of the inverted parameters. CMA-ES will be carried automatically based of the content of the `CMAESParameters` list.

        .. rubric :: Usage

        The inversion is carried in a two step procedure

        .. code::

        cma_es = parallel_CMA_ES(**parameters)
        cma_es.solve(data, process, misfit, stations, db, wavelet, iterations=10)

        In the first step, the user supplies parameters such as the number of mutants, the list of inverted parameters, the catalog origin, etc. (see below for detailed argument descriptions).

        In the second step, the user supplies data, data process, misfit type, stations list, an Axisem Green's function database, a source wavelet and a number of iterations on which to carry the CMA-ES inversion (number of generations).

        .. rubric:: Parameters

        '''

        # Generate Class based MPI communicator
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()


        # Initialize parameters-tied variables.
        # Variables are initially shared across all processes, but most of the important computations will be carried on the root process (self.rank == 0) only.
        self.event_id = event_id
        self.iteration = 0
        self._parameters = parameters_list
        self._parameters_names = [parameter.name for parameter in parameters_list]
        self.n = len(self._parameters)
        self.xmean = np.asarray([[val.initial for val in self._parameters]]).T
        self.sigma = 1 # Default initial gaussian variance for all parameters.
        self.catalog_origin = origin
        self.counteval = 0
        if not callback_function == None:
            self.callback = callback_function
        elif 'Mw' in self._parameters_names or 'kappa' in self._parameters_names:
            self.callback = to_mij
            self.mij_args = ['rho', 'v', 'w', 'kappa', 'sigma', 'h']
            self.mode = 'mt'
        elif 'F0' in self._parameters_names == 3:
            self.callback = to_force
            self.mode = 'force'

        # Main user input: lmbda is the number of mutants. If no lambda is given, it will determine the number of mutants based on the number of parameters.
        if lmbda == None:
            self.lmbda = int(4 + np.floor(3*np.log(len(self._parameters))))
        else:
            self.lmbda = lmbda

        self.mu = np.floor(self.lmbda/2)
        a = 1 # Original author uses 1/2 in tutorial and 1 in publication
        self.weights = np.array([np.log(self.mu+a) - np.log(np.arange(1, self.mu+1))]).T
        self.weights /= sum(self.weights)
        self.mueff = sum(self.weights)**2/sum(self.weights**2)

        # Step-size control
        self.cs = (self.mueff + 2) / (self.n + self.mueff + 5)
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1)/(self.n + 1)) - 1) + self.cs

        # Covariance matrix adaptation
        self.cc = (4 + self.mueff / self.n)/(self.n + 4 + 2 * self.mueff / self.n)
        self.acov = 2
        self.c1 = self.acov / ((self.n + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, self.acov * (self.mueff - 2 + 1 / self.mueff) / ((self.n + 2)**2 + self.acov*self.mueff/2))

        # Defining 'holder' variable for post processing and plotting.
        self._misfit_holder = np.zeros((int(self.lmbda),1))
        self.mutants_logger_list = pd.DataFrame()
        self.mean_logger_list = pd.DataFrame()

        # INITIALIZATION
        self.ps = np.zeros_like(self.xmean)
        self.pc = np.zeros_like(self.xmean)
        self.B = np.eye(self.n,self.n)
        self.D = np.ones((self.n, 1))
        self.C = self.B @ np.diag(self.D[:,0]**2) @ self.B.T
        self.invsqrtC = self.B @ np.diag(self.D[:,0]**-1) @ self.B.T
        self.eigeneval = 0
        self.chin = self.n**0.5 * (1 - 1 / ( 4 * self.n) + 1 / (21 * self.n**2))
        self.mutants = np.zeros((self.n, self.lmbda))

    def draw_mutants(self):
        '''
        Function that randomly draw `self.lmbda` mutants from a Gaussian repartition on the root process. It then applies the corresponding repair method for each of the parameters, and scatter the splitted list of muttant accross all the computing processes.
        '''
        if self.rank == 0:
            # Hardcode the bounds of CMA-ES search for safety.
            bounds = [0,10]
            for i in range(self.lmbda):
                mutant = self.xmean + self.sigma * self.B @ (self.D * np.random.randn(self.n,1))
                self.mutants[:,i] = mutant.T
            # Loop through all parameters to get their repair methods
            for _i, param in enumerate(self.mutants):
                if not self._parameters[_i].repair == None:
                    # If samples are out of the [0,10] range, apply repair method
                    while array_in_bounds(self.mutants[_i], 0, 10) == False:
                        print('repairing '+self._parameters[_i].name+' with '+self._parameters[_i].repair+' method')
                        Repair(self._parameters[_i].repair, self.mutants[_i], self.xmean[_i])
            self.mutant_lists = np.array_split(self.mutants, self.size, axis=1)
        else:
            self.mutant_lists = None
        # Scatter the splited mutant_lists across processes.
        self.scattered_mutants = self.comm.scatter(self.mutant_lists, root=0)

    def mean_diff(self, new, old):
        '''
        Compute mean change, and apply circular difference for wrapped repair methods (implying periodic parameters)
        '''
        diff = new-old
        for _i, param in enumerate(self._parameters):
            if param.repair == 'wrapping':
                angular_diff = linear_transform(new[_i], 0, 360)-linear_transform(old[_i], 0, 360)
                angular_diff = inverse_linear_transform((angular_diff+180)%360 - 180, 0, 360)
                diff[_i] = angular_diff
        return diff

    def circular_mean(self, id):
        '''
        Compute the circular mean on the "id"th parameter. Ciruclar mean allows to compute mean of the samples on a periodic space.
        '''
        param = self.mutants[id]
        a = linear_transform(param, 0, 360)-180
        mean = np.rad2deg(np.arctan2(np.sum(np.sin(np.deg2rad(a[range(int(self.mu))]))*self.weights.T), np.sum(np.cos(np.deg2rad(a[range(int(self.mu))]))*self.weights.T)))+180
        mean = inverse_linear_transform(mean, 0, 360)
        return mean

    # Evaluate the misfit for each mutant of the population.
    def eval_fitness(self, data, stations, db, process, misfit, wavelet, verbose=False):
        # Project each parameter in their respective physical domain, according to their `scaling` property
        self.transformed_mutants = np.zeros_like(self.scattered_mutants)
        for _i, param in enumerate(self._parameters):
            # print(param.scaling) # - DEBUG PRINT
            if param.scaling == 'linear':
                self.transformed_mutants[_i] = linear_transform(self.scattered_mutants[_i], param.lower_bound, param.upper_bound)
            elif param.scaling == 'log':
                self.transformed_mutants[_i] = logarithmic_transform(self.scattered_mutants[_i], param.lower_bound, param.upper_bound)
            else:
                raise ValueError("Unrecognized scaling, must be linear or log")
            # Apply optional projection operator to each parameter
            if not param.projection is None:
                self.transformed_mutants[_i] = np.asarray(list(map(param.projection, self.transformed_mutants[_i])))

        # Using callback for mt or force inversion
        if self.mode == 'mt':
            self.sources = np.ascontiguousarray(self.callback(*self.transformed_mutants[0:6]))

        # Generate origin list if required
        if 'latitude' in self._parameters_names\
        or 'longitude'in self._parameters_names\
        or 'depth' in self._parameters_names:
            if self.rank == 0:
                print('creating new origins list')
            self.create_origins()
            if verbose == True:
                for X in self.origins:
                    print(X)
        # Load, convolve and process local greens function
        start_time = MPI.Wtime()
        self.local_greens = db.get_greens_tensors(stations, self.origins)
        end_time = MPI.Wtime()
        if self.rank == 0:
            print("Fetching greens tensor: " + str(end_time-start_time))
        start_time = MPI.Wtime()
        self.local_greens.convolve(wavelet)
        end_time = MPI.Wtime()
        if self.rank == 0:
            print("Convolving: " + str(end_time-start_time))
        start_time = MPI.Wtime()
        self.local_greens = self.local_greens.map(process)
        end_time = MPI.Wtime()
        if self.rank == 0:
            print("Processing greens: " + str(end_time-start_time))


        self.local_misfit_val = [misfit(data, self.local_greens.select(origin), np.array([self.sources[_i]])) for _i, origin in enumerate(self.origins)]
        self.local_misfit_val = np.asarray(self.local_misfit_val).T[0]
        if verbose == True:
            print("local misfit is :", self.local_misfit_val) # - DEBUG PRINT


        # # Gather_time_shift_info
        # n_groups = len(misfit.time_shift_groups)
        # n_stations = len(data)
        # for sta in range(n_stations):
        #     for grp in range(n_groups):
        #         misfit.collect_attributes(data, self.local_greens.select(self.origins[0]), mtuq.event.MomentTensor(self.sources[0]))[n_stations]

        # Gather local misfit values
        self.misfit_val = self.comm.gather(self.local_misfit_val.T, root=0)
        # Broadcast the gathered values and concatenate to return across processes.
        self.misfit_val = self.comm.bcast(self.misfit_val, root=0)
        self.misfit_val = np.asarray(np.concatenate(self.misfit_val))
        self._misfit_holder += self.misfit_val
        return self.misfit_val

    def gather_mutants(self, verbose=False):
        self.mutants = self.comm.gather(self.scattered_mutants, root=0)
        if self.rank == 0:
            self.mutants = np.concatenate(self.mutants, axis=1)
            if verbose == True:
                print(self.mutants, '\n', 'shape is', np.shape(self.mutants), '\n', 'type is', type(self.mutants)) # - DEBUG PRINT
        else:
            self.mutants = None

        self.transformed_mutants = self.comm.gather(self.transformed_mutants, root=0)
        if self.rank == 0:
            self.transformed_mutants = np.concatenate(self.transformed_mutants, axis=1)
            if verbose == True:
                print(self.transformed_mutants, '\n', 'shape is', np.shape(self.transformed_mutants), '\n', 'type is', type(self.transformed_mutants)) # - DEBUG PRINT
        else:
            self.transformed_mutants = None

        self.mutants_logger_list = self.mutants_logger_list.append(
                            self._datalogger(mean=False))

    def fitness_sort(self, misfit):
        # Sort by fitness
        if self.rank == 0:
            self.mutants = self.mutants[:,np.argsort(misfit.T)[0]]
            self.transformed_mutants = self.transformed_mutants[:,np.argsort(misfit.T)[0]]
            self._misfit_holder *= 0

    def update_mean(self):
        # Update the mean
        if self.rank == 0:
            self.xold = self.xmean
            self.xmean = np.array([np.sum(self.mutants[:,range(int(self.mu))]*self.weights.T, axis=1)]).T
            for _i, param in enumerate(self._parameters):
                if param.repair == 'wrapping':
                    print('computing wrapped mean for parameter:', param.name)
                    self.xmean[_i] = self.circular_mean(_i)
            self.mean_logger_list=self.mean_logger_list.append(
                                self._datalogger(mean=True), ignore_index=True)


    def update_step_size(self):
        # Step size control
        if self.rank == 0:
            self.ps = (1-self.cs)*self.ps + np.sqrt(self.cs*(2-self.cs)*self.mueff) * self.invsqrtC @ (self.mean_diff(self.xmean, self.xold) / self.sigma)

    def update_covariance(self):
        # Covariance matrix adaptation
        if self.rank == 0:
            if np.linalg.norm(self.ps)/np.sqrt(1-(1-self.cs)**(2*self.counteval/self.lmbda))/self.chin < 1.4 + 2/(self.n+1):
                self.hsig = 1
            if np.linalg.norm(self.ps)/np.sqrt(1-(1-self.cs)**(2*self.counteval/self.lmbda))/self.chin >= 1.4 + 2/(self.n+1):
                self.hsig = 0
            self.pc = (1-self.cc) * self.pc + self.hsig * np.sqrt(self.cc*(2-self.cc)*self.mueff) * self.mean_diff(self.xmean, self.xold) / self.sigma

            artmp = (1/self.sigma) * self.mean_diff(self.mutants[:,0:int(self.mu)], self.xold)
            self.C = (1-self.c1-self.cmu) * self.C + self.c1 * (self.pc@self.pc.T + (1-self.hsig) * self.cc*(2-self.cc) * self.C) + self.cmu * artmp @ np.diag(self.weights.T[0]) @ artmp.T
            # Adapt step size
            self.sigma = self.sigma * np.exp((self.cs/self.damps)*(np.linalg.norm(self.ps)/self.chin - 1))

            if self.counteval - self.eigeneval > self.lmbda/(self.c1+self.cmu)/self.n/10:
                self.eigeneval = self.counteval
                self.C = np.triu(self.C) + np.triu(self.C,1).T
                self.D,self.B = np.linalg.eig(self.C)
                self.D = np.array([self.D]).T
                self.D = np.sqrt(self.D)
                self.invsqrtC = self.B @ np.diag(self.D[:,0]**-1) @ self.B.T

    def create_origins(self):
        self.catalog_origin

        depth = self.transformed_mutants[6]
        latitude = self.transformed_mutants[7]
        longitude = self.transformed_mutants[8]

        self.origins = []
        for i in range(len(depth)):
            self.origins += [self.catalog_origin.copy()]
            setattr(self.origins[-1], 'depth_in_m', depth[i])
            setattr(self.origins[-1], 'latitude', latitude[i])
            setattr(self.origins[-1], 'longitude', longitude[i])

    def return_candidate_solution(self, id=None):
        if not id == None:
            return_x = np.array([self.mutants[:,id]]).T
        else:
            return_x = self.xmean
        self.transformed_mean = np.zeros_like(return_x)
        for _i, param in enumerate(self._parameters):
            print(param.scaling)
            if param.scaling == 'linear':
                self.transformed_mean[_i] = linear_transform(return_x[_i], param.lower_bound, param.upper_bound)
            elif param.scaling == 'log':
                self.transformed_mean[_i] = logarithmic_transform(return_x[_i], param.lower_bound, param.upper_bound)
            else:
                raise ValueError("Unrecognized scaling, must be linear or log")
            # Apply optional projection operator to each parameter
            if not param.projection is None:
                self.transformed_mean[_i] = np.asarray(list(map(param.projection, self.transformed_mean[_i])))

        self.catalog_origin
        depth = self.transformed_mean[6]
        latitude = self.transformed_mean[7]
        longitude = self.transformed_mean[8]
        self.final_origin = []
        for i in range(len(depth)):
            self.final_origin += [self.catalog_origin.copy()]
            setattr(self.final_origin[-1], 'depth_in_m', depth[i])
            setattr(self.final_origin[-1], 'latitude', latitude[i])
            setattr(self.final_origin[-1], 'longitude', longitude[i])

        return(self.transformed_mean, self.final_origin)

    def _datalogger(self, mean=False):
        """
        The datalogger save in memory all of the CMA-ES mutant drawn and evaluated during the inversion. This allows quickly access the inversion recods in order to plot the misfit. The data is stored within a pandas.DataFrame().

        When mean=False the datalogger stores the coordinates of each mutants (Mw, v, w, kappa, sigma,...) and misfit at the current iterations.

        When mean=True the datalogger stores the coordinates of the mean mutant at the current iterations. The mean mutant's misfit is not evaluated and thus only its coordinates are returned.
        """
        if self.rank == 0:
            if mean==False:
                coordinates = self.transformed_mutants.T
                misfit_values = self._misfit_holder
                results = np.hstack((coordinates, misfit_values))
                columns_labels=self._parameters_names+["misfit"]

            if mean==True:
                self.transformed_mean = np.zeros_like(self.xmean)
                for _i, param in enumerate(self._parameters):
                    print(param.scaling)
                    if param.scaling == 'linear':
                        self.transformed_mean[_i] = linear_transform(self.xmean[_i], param.lower_bound, param.upper_bound)
                    elif param.scaling == 'log':
                        self.transformed_mean[_i] = logarithmic_transform(self.xmean[_i], param.lower_bound, param.upper_bound)
                    else:
                        raise ValueError("Unrecognized scaling, must be linear or log")
                    # Apply optional projection operator to each parameter
                    if not param.projection is None:
                        self.transformed_mean[_i] = np.asarray(list(map(param.projection, self.transformed_mean[_i])))

                results = self.transformed_mean.T
                columns_labels=self._parameters_names

            da = pd.DataFrame(
            data=results,
            columns=columns_labels
            )
            return(MTUQDataFrame(da))


    def solve(self, data, process, misfit, stations, db, wavelet, iterations=1, verbose=False, plot_waveforms=True):
        for it in range(iterations):
            self.draw_mutants()
            total_misfit = np.zeros((self.lmbda, 1))
            for _i in range(len(data)):
                total_misfit += self.eval_fitness(data[_i], stations, db, process[_i], misfit[_i], wavelet, verbose)
            self.counteval += self.lmbda
            self.gather_mutants()
            self.fitness_sort(total_misfit)

            # CMA-ES update and adaptation steps
            self.update_mean()
            self.update_step_size()
            self.update_covariance()
            if self.rank == 0:
                if plot_waveforms==True:
                    self.plot_mean_waveforms(data, process, misfit, stations, db)
            if plot_waveforms==True:
                self.iteration += 1

    def plot_mean_waveforms(self, data, process, misfit, stations, db):

        mean_solution, final_origin = self.return_candidate_solution()

        mean_solution = self.mean_logger_list.iloc[-1][['Mw','v','w','kappa','sigma','h']].values
        solution_grid = UnstructuredGrid(dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'),coords=(mean_solution),callback=to_mt)


        best_source = solution_grid.get(0)
        lune_dict = solution_grid.get_dict(0)
        greens = db.get_greens_tensors(stations, final_origin[0])

        if len(data) == len(process) == len(misfit) == 2:
            greens_0 = greens.map(process[0])
            greens_1 = greens.map(process[1])
            greens_1[0].tags[0]='model:ak135f_2s'
            plot_data_greens2(self.event_id+'FMT_waveforms_mean_'+str(self.iteration)+'.png',data[0], data[1], greens_0, greens_1, process[0], process[1], misfit[0], misfit[1], stations, final_origin[0], best_source, lune_dict)

        if len(data) == len(process) == len(misfit) == 1:
            greens_1 = greens.map(process[0])
            greens_1[0].tags[0]='model:ak135f_2s'
            plot_data_greens1(self.event_id+'FMT_waveforms_mean_'+str(self.iteration)+'.png',data, greens, process, misfit, stations, final_origin[0], best_source, lune_dict)
