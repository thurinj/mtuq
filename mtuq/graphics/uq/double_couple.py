#
# graphics/uq/double_couple.py - uncertainty quantification of double couple sources
#

import numpy as np

from matplotlib import pyplot
from pandas import DataFrame
from xarray import DataArray
from mtuq.graphics._gmt import read_cpt, _cpt_path
from mtuq.graphics.uq._matplotlib import _plot_dc_matplotlib
from mtuq.grid_search import MTUQDataArray, MTUQDataFrame
from mtuq.util import dataarray_idxmin, dataarray_idxmax, defaults, warn
from mtuq.util.math import closed_interval, open_interval, to_delta, to_gamma, to_mij
from os.path import exists



def plot_misfit_dc(filename, ds, **kwargs):
    """ Plots misfit values over strike, dip, slip

    .. rubric :: Required input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_dc.html>`_

    """
    defaults(kwargs, {
        'colormap': 'viridis',
        'squeeze': 'min',
        })

    _check(ds)

    if issubclass(type(ds), DataArray):
        misfit = _misfit_dc_regular(ds)
        
    elif issubclass(type(ds), DataFrame):
        # warn('plot_misfit_dc() not implemented for irregularly-spaced grids.\n'
            #  'No figure will be generated.')
        misfit = _misfit_dc_random(ds)
        # return

    _plot_dc(filename, misfit, **kwargs)



def plot_likelihood_dc(filename, ds, var, **kwargs):
    """ Plots maximum likelihood values over strike, dip, slip

    .. rubric :: Required input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

   ``var`` (`float` or `array`):
    Data variance


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_dc.html>`_

    """
    defaults(kwargs, {
        'colormap': 'hot_r',
        'squeeze': 'max',
        })

    _check(ds)

    if issubclass(type(ds), DataArray):
        likelihoods = _likelihoods_dc_regular(ds, var)

    elif issubclass(type(ds), DataFrame):
        warn('plot_likelihood_dc() not implemented for irregularly-spaced grids.\n'
             'No figure will be generated.')
        return

    _plot_dc(filename, likelihoods, **kwargs)



def plot_marginal_dc(filename, ds, var, **kwargs):
    """ Plots marginal likelihood values over strike, dip, slip

    .. rubric :: Required input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

   ``var`` (`float` or `array`):
    Data variance


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_dc.html>`_

    """
    defaults(kwargs, {
        'colormap': 'hot_r',
        'squeeze': 'max',
        })

    _check(ds)

    if issubclass(type(ds), DataArray):
        marginals = _marginals_dc_regular(ds, var)

    elif issubclass(type(ds), DataFrame):
        warn('plot_marginal_dc() not implemented for irregularly-spaced grids.\n'
             'No figure will be generated.')
        return

    _plot_dc(filename, marginals, **kwargs)



def plot_variance_reduction_dc(filename, ds, data_norm, **kwargs):
    """ Plots variance reduction values over strike, dip, slip

    .. rubric :: Required input arguments

    ``filename`` (`str`):
    Name of output image file

    ``ds`` (`DataArray` or `DataFrame`):
    Data structure containing moment tensors and corresponding misfit values

   ``data_norm`` (`float`):
    Data norm


    .. rubric :: Optional input arguments

    For optional argument descriptions, 
    `see here <mtuq.graphics._plot_dc.html>`_

    """
    defaults(kwargs, {
        'colormap': 'viridis_r',
        'squeeze': 'max',
        })

    _check(ds)

    if issubclass(type(ds), DataArray):
        variance_reduction = _variance_reduction_dc_regular(ds, var)

    elif issubclass(type(ds), DataFrame):
        warn('plot_variance_reduction_dc() not implemented for irregularly-spaced grids.\n'
             'No figure will be generated.')
        return

    _plot_dc(filename, variance_reduction, **kwargs)



def _plot_dc(filename, da, show_best=True, backend=_plot_dc_matplotlib,
    squeeze='min', **kwargs):

    """ Plots DataArray values over strike, dip, slip

    .. rubric :: Keyword arguments

    ``colormap`` (`str`)
    Color palette used for plotting values 
    (choose from GMT or MTUQ built-ins)

    ``show_best`` (`bool`):
    Show where best-fitting moment tensor falls in terms of strike, dip, slip

    ``squeeze`` (`str`):
    By default, 2-D surfaces are obtained by minimizing or maximizing.
    For slices instead, use `slice_min` or `slice_max`.

    ``backend`` (`function`):
    Choose from `_plot_dc_matplotlib` (default) or user-supplied function

    """

    if not issubclass(type(da), DataArray):
        raise Exception()

    if show_best:
        if 'best_dc' in da.attrs:
            best_dc = da.attrs['best_dc']
        else:
            warn("Best-fitting orientation not given")
            best_dc = None

    # note the following parameterization details
    #     kappa = strike
    #     sigma = slip
    #     h = cos(dip)

    # squeeze full 3-D array into 2-D arrays
    if squeeze=='min':
        values_h_kappa = da.min(dim=('sigma')).values
        values_sigma_kappa = da.min(dim=('h')).values
        values_sigma_h = da.min(dim=('kappa')).values.T

    elif squeeze=='max':
        values_h_kappa = da.max(dim=('sigma')).values
        values_sigma_kappa = da.max(dim=('h')).values
        values_sigma_h = da.max(dim=('kappa')).values.T

    elif squeeze=='slice_min':
        argmin = da.argmin(('kappa','sigma','h'))
        values_h_kappa = da.isel(sigma=argmin['sigma'], drop=True).values
        values_sigma_kappa = da.isel(h=argmin['h'], drop=True).values
        values_sigma_h = da.isel(kappa=argmin['kappa'], drop=True).values.T

    elif squeeze=='slice_max':
        argmax = da.argmax(('kappa','sigma','h'))
        values_h_kappa = da.isel(sigma=argmax['sigma'], drop=True).values
        values_sigma_kappa = da.isel(h=argmax['h'], drop=True).values
        values_sigma_h = da.isel(kappa=argmax['kappa'], drop=True).values.T

    else:
        raise ValueError
    
    if values_h_kappa.dtype == 'object':
        values_h_kappa = np.asarray(values_h_kappa, dtype=float)
    if values_sigma_kappa.dtype == 'object':
        values_sigma_kappa = np.asarray(values_sigma_kappa, dtype=float)
    if values_sigma_h.dtype == 'object':
        values_sigma_h = np.asarray(values_sigma_h, dtype=float)

    backend(filename,
        da.coords,
        values_h_kappa,
        values_sigma_kappa,
        values_sigma_h,
        best_dc=best_dc,
        **kwargs)


#
# for extracting misfit, variance reduction and likelihood from
# regularly-spaced grids
#

def _misfit_dc_regular(da):
    """ For each moment tensor orientation, extract minimum misfit
    """
    misfit = da.min(dim=('origin_idx', 'rho', 'v', 'w'))

    return misfit.assign_attrs({
        'best_mt': _min_mt(da),
        'best_dc': _min_dc(da),
        })

def _misfit_dc_random(df, **kwargs):
    df = df.copy()
    df = df.reset_index()
    da = _bin_dc_regular(df, lambda df: df.min(), **kwargs)

    return da.assign_attrs({
        'best_dc': _min_dc(da),
    })


def _likelihoods_dc_regular(da, var):
    """ For each moment tensor orientation, calculate maximum likelihood
    """
    likelihoods = da.copy()
    likelihoods.values = np.exp(-likelihoods.values/(2.*var))
    likelihoods.values /= likelihoods.values.sum()

    likelihoods = likelihoods.max(dim=('origin_idx', 'rho', 'v', 'w'))
    likelihoods.values /= likelihoods.values.sum()
    #likelihoods /= dc_area

    return likelihoods.assign_attrs({
        'best_mt': _min_mt(da),
        'best_dc': _min_dc(da),
        'maximum_likelihood_estimate': dataarray_idxmax(likelihoods).values(),
        })


def _marginals_dc_regular(da, var):
    """ For each moment tensor orientation, calculate marginal likelihood
    """
    likelihoods = da.copy()
    likelihoods.values = np.exp(-likelihoods.values/(2.*var))
    likelihoods.values /= likelihoods.values.sum()

    marginals = likelihoods.sum(dim=('origin_idx', 'rho', 'v', 'w'))
    marginals.values /= marginals.values.sum()

    return marginals.assign_attrs({
        'best_dc': _max_dc(marginals),
        })


def _variance_reduction_dc_regular(da, data_norm):
    """ For each moment tensor orientation, extracts maximum variance reduction
    """
    variance_reduction = 1. - da.copy()/data_norm

    variance_reduction = variance_reduction.max(
        dim=('origin_idx', 'rho', 'v', 'w'))

    # widely-used convention - variance reducation as a percentage
    variance_reduction.values *= 100.

    return variance_reduction.assign_attrs({
        'best_mt': _min_mt(da),
        'best_dc': _min_dc(da),
        'lune_array': _lune_array(da),
        })


#
# utility functions
#

def _min_mt(da):
    """ Returns moment tensor vector corresponding to minimum DataArray value
    """
    da = dataarray_idxmin(da)
    lune_keys = ['rho', 'v', 'w', 'kappa', 'sigma', 'h']
    lune_vals = [da[key].values for key in lune_keys]
    return to_mij(*lune_vals)


def _max_mt(da):
    """ Returns moment tensor vector corresponding to maximum DataArray value
    """
    da = dataarray_idxmax(da)
    lune_keys = ['rho', 'v', 'w', 'kappa', 'sigma', 'h']
    lune_vals = [da[key].values for key in lune_keys]
    return to_mij(*lune_vals)


def _min_dc(da):
    """ Returns orientation angles corresponding to minimum DataArray value
    """
    da = dataarray_idxmin(da)
    dc_keys = ['kappa', 'sigma', 'h']
    dc_vals = [da[key].values for key in dc_keys]
    return dc_vals

def _max_dc(da):
    """ Returns orientation angles corresponding to maximum DataArray value
    """
    da = dataarray_idxmax(da)
    dc_keys = ['kappa', 'sigma', 'h']
    dc_vals = [da[key].values for key in dc_keys]
    return dc_vals


def _bin_dc_regular(df, handle, npts=15, **kwargs):
    """ Bins irregularly-spaced moment tensors orientations into square cells
    to plot dc-misfit grids
    """
    # Orientation bins
    kappa_min, kappa_max = 0, 360
    sigma_min, sigma_max = -90, 90
    h_min, h_max = 0, 1

    kappa_centers = open_interval(kappa_min, kappa_max, npts)
    sigma_centers = open_interval(sigma_min, sigma_max, npts)
    h_centers = open_interval(h_min, h_max, npts)

    kappa_edges = closed_interval(kappa_min, kappa_max, npts + 1)
    sigma_edges = closed_interval(sigma_min, sigma_max, npts + 1)
    h_edges = closed_interval(h_min, h_max, npts + 1)

    # Prepare the data arrays
    kappa_vals = df['kappa'].values
    sigma_vals = df['sigma'].values
    h_vals = df['h'].values

    # Compute the 3D histogram
    hist, edges = np.histogramdd(
        np.column_stack((kappa_vals, sigma_vals, h_vals)),
        bins=(kappa_edges, sigma_edges, h_edges)
    )

    # Process the binned data
    binned = np.empty_like(hist, dtype=object)
    binned[:] = None
    nonzero_bins = hist.nonzero()
    for idx in zip(*nonzero_bins):
        subset = df.loc[(kappa_vals >= kappa_edges[idx[0]]) &
                        (kappa_vals < kappa_edges[idx[0] + 1]) &
                        (sigma_vals >= sigma_edges[idx[1]]) &
                        (sigma_vals < sigma_edges[idx[1] + 1]) &
                        (h_vals >= h_edges[idx[2]]) &
                        (h_vals < h_edges[idx[2] + 1])]
        if len(subset) > 0:
            binned[idx] = handle((subset['misfit']))

    return DataArray(
        dims=('h', 'sigma', 'kappa'),
        coords=(h_centers, sigma_centers, kappa_centers),
        data=binned.transpose()
        )

def _check(ds):
    """ Checks data structures
    """
    if type(ds) not in (DataArray, DataFrame, MTUQDataArray, MTUQDataFrame):
        raise TypeError("Unexpected grid format")


