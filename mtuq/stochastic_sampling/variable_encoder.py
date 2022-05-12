import numpy as np

class CMAESParameters:
    def __init__(self, name, lower_bound, upper_bound, scaling = 'linear', initial=None, repair='bounce_back', projection=None, **kwargs):
        self.name = name # Name of the parameter, used for printing and saving to file names etc...
        if lower_bound > upper_bound:
            raise ValueError('Lower bound is larger than the Upper Bound')

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if scaling not in ['linear', 'log']:
            raise ValueError('Scaling must be either linear or log')

        self.scaling=scaling # Define the scaling type to rescale the randomly drawn parameter in[0,10] to a value between self.lower_bound and self.upper_bound
        if initial == None:
            self.initial = np.random.uniform(0,10)
        elif initial < 0 or initial > 10:
            raise ValueError('Initial value is outside of the expected bounds')
        else:
            self.initial = initial # Initial guess for parameter values, used to seed CMA-ES optimisation algorithm
        self.repair = repair # Repair method used to redraw samples out of the [0,10] range.
        self.projection = projection

        if 'grid' in kwargs:
            self.grid = kwargs['grid'] # Grid of parameter values to use a non continuous optimization purposes. must be an array with initial and final value consistent with predefined lower_bound and upper_bound.
        else:
            self.grid = None

    # def get_scaled_value(self, value): # Returns the scaled parameter values based on whether it is linear or a log scale. If grid has been defined then returns an interpolated value between two grid points.
    #     if self.scaling == 'linear': # Linear scaling, returns a value between lower and upper bounds based on the input parameter values
    #
    #         scaled_value = (self.upper_bound - self.lower_bound) * ((value-1)/9)+self.lower_bound
    #         scaled_value = linear_transform()
    #
    #     elif self.scaling == 'log': # Log scaling, returns a value between lower and upper bounds based on the log of parameter values
    #
    #         scaled_value = (self.upper_bound - self.lower_bound) * ((np.log(value)-np.log(1))/(np.log(10)-np.log(1)))+self.lower_bound
    #
    #     if self.grid is not None: # If a grid has been defined, then return the closest value on the grid
    #         scaled_value = self.grid[np.argmin(abs((self.grid - scaled_value)))]
    #
    #     return np.round(scaled_value, 2) # Rounds the value to two decimal places and returns it
