import numpy as np
from mtuq.stochastic_sampling.cmaes_utils import linear_transform, inverse_linear_transform, logarithmic_transform, in_bounds, array_in_bounds, Repair


class CMAESMutantGeneration:
    def __init__(self, parameters, xmean, sigma, B, D, n, lmbda, size, rank, comm, verbose_level):
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
        self.mutants = np.zeros((self.n, self.lmbda))

    def draw_mutants(self):
        if self.rank == 0:
            self._generate_mutants()
            self._repair_and_redraw_mutants()
            self._scatter_mutants()
        else:
            self._receive_mutants()
        self.scattered_mutants = self.mutant_slice

    def _generate_mutants(self):
        for i in range(self.lmbda):
            mutant = self._draw_single_mutant()
            self.mutants[:, i] = mutant.flatten()

    def _draw_single_mutant(self):
        return self.xmean + self.sigma * self.B @ (self.D * np.random.randn(self.n, 1))

    def _repair_and_redraw_mutants(self):
        bounds = [0, 10]
        redraw_counter = 0
        for param_idx in range(self.n):
            param_values = self.mutants[param_idx, :]
            if array_in_bounds(param_values, bounds[0], bounds[1]):
                continue
            if self.parameters[param_idx].repair is None:
                self.mutants[param_idx, :], was_redrawn = self._redraw_param_until_valid(param_values, bounds)
                if was_redrawn:
                    redraw_counter += 1
                print(f'Redrawn {redraw_counter} out-of-bounds mutants for parameter {self.parameters[param_idx].name}')
            else:
                self.mutants[param_idx, :] = self._apply_repair_to_param(param_values, bounds, param_idx)

    def _redraw_param_until_valid(self, param_values, bounds):
        was_redrawn = False
        for i in range(len(param_values)):
            while not in_bounds(param_values[i], bounds[0], bounds[1]):
                param_values[i] = np.random.randn()
                was_redrawn = True
        return param_values, was_redrawn

    def _apply_repair_to_param(self, param_values, bounds, param_idx):
        param = self.parameters[param_idx]
        printed_repair = False
        while not array_in_bounds(param_values, bounds[0], bounds[1]):
            if self.verbose_level >= 0 and not printed_repair:
                print(f'Repairing parameter {param.name} using method {param.repair}')
                printed_repair = True
            Repair(param.repair, param_values, self.xmean[param_idx])
        return param_values

    def _scatter_mutants(self):
        self.mutant_lists = np.array_split(self.mutants, self.size, axis=1)
        self.mutant_slice = self.comm.scatter(self.mutant_lists, root=0)

    def _receive_mutants(self):
        self.mutant_slice = self.comm.scatter(None, root=0)
