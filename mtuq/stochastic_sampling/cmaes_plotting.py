import numpy as np
from mtuq.graphics import plot_data_greens2, plot_data_greens1
from mtuq.graphics.uq._matplotlib import _hammer_projection, _generate_lune, _generate_sphere
from mtuq.util.math import wrap_180, to_gamma, to_delta
from mtuq.event import MomentTensor, Force
from mtuq.grid.moment_tensor import UnstructuredGrid
from mtuq.grid.force import UnstructuredGrid
from mtuq.misfit import Misfit, PolarityMisfit
from mtuq.io.clients.AxiSEM_NetCDF import Client as AxiSEM_Client

def plot_mean_waveforms(CMA, data_list, process_list, misfit_list, stations, db_or_greens_list):
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

    if CMA.rank != 0:
        return  # Exit early if not rank 0

    mean_solution, final_origin = CMA.return_candidate_solution()

    # Solution grid will change depending on the mode (mt, mt_dev, mt_dc, or force)
    modes = {
        'mt': ('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
        'mt_dev': ('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
        'mt_dc': ('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
        'force': ('F0', 'phi', 'h'),
    }

    if CMA.mode not in modes:
        raise ValueError('Invalid mode. Supported modes for the plotting functions in the Solve method: "mt", "mt_dev", "mt_dc", "force"')

    mode_dimensions = modes[CMA.mode]

    # Pad mean_solution based on moment tensor mode (deviatoric or double couple)
    if CMA.mode == 'mt_dev':
        mean_solution = np.insert(mean_solution, 2, 0, axis=0)
    elif CMA.mode == 'mt_dc':
        mean_solution = np.insert(mean_solution, 1, 0, axis=0)
        mean_solution = np.insert(mean_solution, 2, 0, axis=0)

    solution_grid = UnstructuredGrid(dims=mode_dimensions, coords=mean_solution, callback=CMA.callback)

    final_origin = final_origin[0]
    if CMA.mode.startswith('mt'):
        best_source = MomentTensor(solution_grid.get(0))
    elif CMA.mode == 'force':
        best_source = Force(solution_grid.get(0))

    lune_dict = solution_grid.get_dict(0)

    # Assignments for brevity (might be removed later)
    data = data_list.copy()
    process = process_list.copy()
    misfit = misfit_list.copy()
    greens_or_db = db_or_greens_list.copy() if isinstance(db_or_greens_list, list) else db_or_greens_list

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
    for i in range(len(misfit) - 1, -1, -1):
        if isinstance(misfit[i], PolarityMisfit):
            del data[i]
            del process[i]
            del misfit[i]
            del greens[i]

    # Plot based on the number of ProcessData objects in the process_list
    if len(process) == 2:
        plot_data_greens2(CMA.event_id + '_waveforms_mean_' + str(CMA.iteration) + '.png',
                            data[0], data[1], greens[0], greens[1], process[0], process[1],
                            misfit[0], misfit[1], stations, final_origin, best_source, lune_dict)
    elif len(process) == 1:
        plot_data_greens1(CMA.event_id + '_waveforms_mean_' + str(CMA.iteration) + '.png',
                            data[0], greens[0], process[0], misfit[0], stations, final_origin, best_source, lune_dict)


def _cmaes_scatter_plot(CMA):
    """
    Generates a scatter plot of the mutants and the current mean solution
    
    Parameters
    ----------
    CMA : CMA_ES    
        The CMA_ES object containing the necessary information for plotting.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object for the plot.
    """
    if CMA.rank == 0:
        # Check if mode is mt, mt_dev or mt_dc or force
        if CMA.mode in ['mt', 'mt_dev', 'mt_dc']:
            if CMA.fig is None:
                CMA.fig, CMA.ax = _generate_lune()

            # Define v as by values from CMA.mutants_logger_list if it exists, otherwise pad with values of zeroes
            m = np.asarray(CMA.mutants_logger_list['misfit'])

            if 'v' in CMA.mutants_logger_list:
                v = np.asarray(CMA.mutants_logger_list['v'])
            else:
                v = np.zeros_like(m)

            if 'w' in CMA.mutants_logger_list:
                w = np.asarray(CMA.mutants_logger_list['w'])
            else:
                w = np.zeros_like(m)

            # Handling the mean solution
            if CMA.mode == 'mt':
                V, W = CMA._datalogger(mean=True)['v'], CMA._datalogger(mean=True)['w']
            elif CMA.mode == 'mt_dev':
                V = CMA._datalogger(mean=True)['v']
                W = 0
            elif CMA.mode == 'mt_dc':
                V = W = 0

            # Projecting the mean solution onto the lune
            V, W = _hammer_projection(to_gamma(V), to_delta(W))
            CMA.ax.scatter(V, W, c='red', marker='x', zorder=10000000)
            # Projecting the mutants onto the lune
            v, w = _hammer_projection(to_gamma(v), to_delta(w))

            vmin, vmax = np.percentile(np.asarray(m), [0, 90])

            CMA.ax.scatter(v, w, c=m, s=3, vmin=vmin, vmax=vmax, zorder=100)

            CMA.fig.canvas.draw()
            return CMA.fig

        elif CMA.mode == 'force':
            if CMA.fig is None:
                CMA.fig, CMA.ax = _generate_sphere()

            # phi and h will always be present in the mutants_logger_list
            m = np.asarray(CMA.mutants_logger_list['misfit'])
            phi, h = np.asarray(CMA.mutants_logger_list['phi']), np.asarray(CMA.mutants_logger_list['h'])
            latitude = np.degrees(np.pi / 2 - np.arccos(h))
            longitude = wrap_180(phi + 90)
            # Getting mean solution
            PHI, H = CMA._datalogger(mean=True)['phi'], CMA._datalogger(mean=True)['h']
            LATITUDE = np.asarray(np.degrees(np.pi / 2 - np.arccos(H)))
            LONGITUDE = wrap_180(np.asarray(PHI + 90))

            # Projecting the mean solution onto the sphere
            LONGITUDE, LATITUDE = _hammer_projection(LONGITUDE, LATITUDE)
            # Projecting the mutants onto the sphere
            longitude, latitude = _hammer_projection(longitude, latitude)

            vmin, vmax = np.percentile(np.asarray(m), [0, 90])

            CMA.ax.scatter(longitude, latitude, c=m, s=3, vmin=vmin, vmax=vmax, zorder=100)
            CMA.ax.scatter(LONGITUDE, LATITUDE, c='red', marker='x', zorder=10000000)
            return CMA.fig