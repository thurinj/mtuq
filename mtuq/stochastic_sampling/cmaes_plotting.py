import numpy as np
from mtuq.graphics import plot_data_greens2, plot_data_greens1
from mtuq.graphics.uq._matplotlib import _hammer_projection, _generate_lune, _generate_sphere
from mtuq.graphics import plot_combined, plot_misfit_force
from mtuq.graphics.uq._matplotlib import _plot_force_matplotlib
from mtuq.event import MomentTensor, Force, Origin
from mtuq.grid.moment_tensor import UnstructuredGrid
from mtuq.grid.force import UnstructuredGrid
from mtuq.misfit import Misfit, PolarityMisfit
from mtuq.util.cmaes import linear_transform, logarithmic_transform, in_bounds, array_in_bounds, Repair

def plot_mean_waveforms(data_list, process_list, misfit_list, stations, db_or_greens_list, mean_solution, final_origin, mode, callback, event_id, iteration, rank):
    """
    Plots the mean waveforms using the base mtuq waveform plots (mtuq.graphics.waveforms).

    Depending on the mode, different parameters are inserted into the mean solution (padding w or v with 0s for instance)
    If green's functions a provided directly, they are used as is. Otherwise, extrace green's function from Axisem database and preprocess them.
    Support only 1 or 2 waveform groups (body and surface waves, or surface waves only)

    Parameters
    ----------
    data_list : list
        A list of data to be plotted (typically `data_bw` and `data_sw`).
    process_list : list
        A list of processes for each data (typically `process_bw` and `process_sw`).
    misfit_list : list
        A list of misfits for each data (typically `misfit_bw` and `misfit_sw`).
    stations : list
        A list of stations.
    db_or_greens_list : list
        Either an AxiSEM_Client instance or a list of GreensTensors (typically `greens_bw` and `greens_sw`).
    mean_solution : numpy.ndarray
        The mean solution to be plotted.
    final_origin : list
        The final origin to be plotted.
    mode : str
        The mode of the inversion ('mt', 'mt_dev', 'mt_dc', or 'force').
    callback : function
        The callback function used for the inversion.
    event_id : str
        The event ID for the inversion.
    iteration : int
        The current iteration of the inversion.
    rank : int
        The rank of the current process.

    Raises
    ------
    ValueError
        If the mode is not 'mt', 'mt_dev', 'mt_dc', or 'force'.
    """

    if rank != 0:
        return  # Exit early if not rank 0

    # Solution grid will change depending on the mode (mt, mt_dev, mt_dc, or force)
    modes = {
        'mt': ('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
        'mt_dev': ('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
        'mt_dc': ('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
        'force': ('F0', 'phi', 'h'),
    }

    if mode not in modes:
        raise ValueError("Invalid mode. Supported modes for the plotting functions in the Solve method: 'mt', 'mt_dev', 'mt_dc', 'force'")

    mode_dimensions = modes[mode]

    # Pad mean_solution based on moment tensor mode (deviatoric or double couple)
    if mode == 'mt_dev':
        mean_solution = np.insert(mean_solution, 2, 0, axis=0)
    elif mode == 'mt_dc':
        mean_solution = np.insert(mean_solution, 1, 0, axis=0)
        mean_solution = np.insert(mean_solution, 2, 0, axis=0)

    solution_grid = UnstructuredGrid(dims=mode_dimensions, coords=mean_solution, callback=callback)

    final_origin = final_origin[0]
    if mode.startswith('mt'):
        best_source = MomentTensor(solution_grid.get(0))
    elif mode == 'force':
        best_source = Force(solution_grid.get(0))

    lune_dict = solution_grid.get_dict(0)

    # Assignments for brevity (might be removed later)
    data = data_list.copy()
    process = process_list.copy()
    misfit = misfit_list.copy()
    greens_or_db = db_or_greens_list.copy() if isinstance(db_or_greens_list, list) else db_or_greens_list

    # greens_or_db = db_or_greens_list

    mode = 'db' if isinstance(greens_or_db, AxiSEM_Client) else 'greens'
    if mode == 'db':
        _greens = greens_or_db.get_greens_tensors(stations, final_origin)
        greens = [None] * len(process_list)
        for i in range(len(process_list)):
            greens[i] = _greens.map(process_list[i])
            greens[i][0].tags[0] = 'model:ak135f_2s'
    elif mode == 'greens':
        greens = greens_or_db

    # Check for the occurences of mtuq.misfit.polarity.PolarityMisfit in misfit_list:
    # if present, remove the corresponding data, greens, process and misfit from the lists
    # Run backward to avoid index errors
    for i in range(len(misfit)-1, -1, -1):
        if isinstance(misfit[i], PolarityMisfit):
            del data[i]
            del process[i]
            del misfit[i]
            del greens[i]

    # Plot based on the number of ProcessData objects in the process_list
    if len(process) == 2:
        plot_data_greens2(event_id + '_waveforms_mean_' + str(iteration) + '.png',
                        data[0], data[1], greens[0], greens[1], process[0], process[1],
                        misfit[0], misfit[1], stations, final_origin, best_source, lune_dict)
    elif len(process) == 1:
        plot_data_greens1(event_id + '_waveforms_mean_' + str(iteration) + '.png',
                        data[0], greens[0], process[0], misfit[0], stations, final_origin, best_source, lune_dict)

def _scatter_plot(mutants_logger_list, mode, fig, ax, rank, _datalogger):
    """
    Generates a scatter plot of the mutants and the current mean solution
    
    Parameters
    ----------
    mutants_logger_list : pandas.DataFrame
        The list of mutants to be plotted.
    mode : str
        The mode of the inversion ('mt', 'mt_dev', 'mt_dc', or 'force').
    fig : matplotlib.figure.Figure
        The figure object for the plot.
    ax : matplotlib.axes.Axes
        The axis object for the plot.
    rank : int
        The rank of the current process.
    _datalogger : function
        The datalogger function used for the inversion.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object for the plot.
    """
    if rank == 0:
        # Check if mode is mt, mt_dev or mt_dc or force
        if mode in ['mt', 'mt_dev', 'mt_dc']:
            if fig is None:  
                fig, ax = _generate_lune()

            # Define v as by values from mutants_logger_list if it exists, otherwise pad with values of zeroes
            m = np.asarray(mutants_logger_list['misfit'])

            if 'v' in mutants_logger_list:
                v = np.asarray(mutants_logger_list['v'])
            else:
                v = np.zeros_like(m)

            if 'w' in mutants_logger_list:
                w = np.asarray(mutants_logger_list['w'])
            else:
                w = np.zeros_like(m)
            
            # Handling the mean solution
            if mode == 'mt':
                V,W = _datalogger(mean=True)['v'], _datalogger(mean=True)['w']
            elif mode == 'mt_dev':
                V = _datalogger(mean=True)['v']
                W = 0
            elif mode == 'mt_dc':
                V = W = 0

            # Projecting the mean solution onto the lune
            V, W = _hammer_projection(to_gamma(V), to_delta(W))
            ax.scatter(V, W, c='red', marker='x', zorder=10000000)
            # Projecting the mutants onto the lune
            v, w = _hammer_projection(to_gamma(v), to_delta(w))


            vmin, vmax = np.percentile(np.asarray(m), [0,90])

            ax.scatter(v, w, c=m, s=3, vmin=vmin, vmax=vmax, zorder=100)

            fig.canvas.draw()
            return fig
        
        elif mode == 'force':
            if fig is None:
                fig, ax = _generate_sphere()

            # phi and h will always be present in the mutants_logger_list
            m = np.asarray(mutants_logger_list['misfit'])
            phi, h = np.asarray(mutants_logger_list['phi']), np.asarray(mutants_logger_list['h'])
            latitude = np.degrees(np.pi/2 - np.arccos(h))
            longitude = wrap_180(phi + 90)
            # Getting mean solution
            PHI, H = _datalogger(mean=True)['phi'], _datalogger(mean=True)['h']
            LATITUDE = np.asarray(np.degrees(np.pi/2 - np.arccos(H)))
            LONGITUDE = wrap_180(np.asarray(PHI + 90))
            
            # Projecting the mean solution onto the sphere
            LONGITUDE, LATITUDE = _hammer_projection(LONGITUDE, LATITUDE)
            # Projecting the mutants onto the sphere
            longitude, latitude = _hammer_projection(longitude, latitude)

            vmin, vmax = np.percentile(np.asarray(m), [0,90])

            ax.scatter(longitude, latitude, c=m, s=3, vmin=vmin, vmax=vmax, zorder=100)
            ax.scatter(LONGITUDE, LATITUDE, c='red', marker='x', zorder=10000000)
            return fig
