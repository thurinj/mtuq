class CMAESBase:
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
        self.scattered_mutants = None
        self.mode = None
        self.parameters_names = [param.name for param in parameters]
        self.transformed_mutants = None
        self.mij_args = None
