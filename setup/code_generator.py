

Imports="""
import os
import numpy as np

from copy import deepcopy
from os.path import join
from mtuq import read, get_greens_tensors, open_db
from mtuq.grid import DoubleCoupleGridRandom
from mtuq.grid_search.mpi import grid_search_mt
from mtuq.cap.misfit import Misfit
from mtuq.cap.process_data import ProcessData
from mtuq.cap.util import Trapezoid
from mtuq.graphics.beachball import plot_beachball
from mtuq.graphics.waveform import plot_data_greens_mt
from mtuq.util import path_mtuq


"""


Docstring_GridSearch_DoubleCouple="""
if __name__=='__main__':
    #
    # Double-couple inversion example
    # 
    # Carries out grid search over 50,000 randomly chosen double-couple 
    # moment tensors
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.DoubleCouple.py
    #
    # For a simpler example, see SerialGridSearch.DoubleCouple.py, 
    # which runs the same inversion in serial
    #

"""


Docstring_GridSearch_DoubleCoupleMagnitudeDepth="""
if __name__=='__main__':
    #
    # Double-couple inversion example
    #   
    # Carries out grid search over source orientation, magnitude, and depth
    #   
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.DoubleCouple+Magnitude+Depth.py
    #
    # This is the most complicated example. For much simpler one, see
    # SerialGridSearch.DoubleCouple.py
    #   

"""


Docstring_GridSearch_FullMomentTensor="""
if __name__=='__main__':
    #
    # Full moment tensor inversion example
    #   
    # Carries out grid search over all moment tensor parameters except
    # magnitude 
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.FullMomentTensor.py
    #   

"""


Docstring_CapStyleGridSearch_DoubleCouple="""
if __name__=='__main__':
    #
    # THIS EXAMPLE ONLY WORKS ON CHINOOK.ALASKA.EDU
    #

    #
    # CAP-style double-couple inversion example
    # 

    # 
    # Carries out grid search over 50,000 randomly chosen double-couple 
    # moment tensors, using Green's functions and phase picks from a local
    # FK database

    #
    # USAGE
    #   mpirun -n <NPROC> python CapStyleGridSearch.DoubleCouple.py
    #

"""


Docstring_CapStyleGridSearch_DoubleCoupleMagnitudeDeapth="""
if __name__=='__main__':
    #
    # THIS EXAMPLE ONLY WORKS ON CHINOOK.ALASKA.EDU
    #

    #
    # CAP-style double-couple inversion example
    # 

    #
    # Carries out grid search over source orientation, magnitude, and depth
    # using Green's functions and phase picks from a local FK database
    #

    #
    # USAGE
    #   mpirun -n <NPROC> python CapStyleGridSearch.DoubleCouple+Magntidue+Depth.py
    #

"""


Docstring_SerialGridSearch_DoubleCouple="""
if __name__=='__main__':
    #
    # Double-couple inversion example
    # 
    # Carries out grid search over 50,000 randomly chosen double-couple 
    # moment tensors
    #
    # USAGE
    #   python SerialGridSearch.DoubleCouple.py
    #
    # A typical runtime is about 10 minutes. For faster results try 
    # GridSearch.DoubleCouple.py, which runs the same inversion in parallel
    #

"""


Docstring_TestGridSearch_DoubleCouple="""
if __name__=='__main__':
    #
    # Grid search integration test
    #
    # This script is similar to examples/SerialGridSearch.DoubleCouple.py,
    # except here we use a coarser grid, and at the end we assert that the test
    # result equals the expected result
    #
    # The compare against CAP/FK:
    #
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1/1/10/10/10 -R0/0/0/0/0/360/0/90/-180/180 20090407201255351
    #
    # Note however that CAP uses a different method for defining regular grids
    #

"""


Docstring_TestGridSearch_DoubleCoupleMagnitudeDepth="""
if __name__=='__main__':
    #
    # Grid search integration test
    #
    # This script is similar to examples/SerialGridSearch.DoubleCouple.py,
    # except here we included mangitude and depth and use a coarser grid
    #
"""


Docstring_TestGraphics="""
if __name__=='__main__':
    #
    # Tests data, synthetics and beachball plotting utilities
    #
    # Note that in the figures created by this script, the data and synthetics 
    # are not expected to fit epsecially well; currently, the only requirement 
    # is that the script runs without errors
    #

    import matplotlib
    matplotlib.use('Agg', warn=False, force=True)
    import matplotlib
"""


Docstring_BenchmarkCAP="""
if __name__=='__main__':
    #
    # Given seven "fundamental" moment tensors, generates MTUQ synthetics and
    # compares with corresponding CAP/FK synthetics
    #
    # Before running this script, it is necessary to unpack the CAP/FK 
    # synthetics using data/tests/unpack.bash
    #
    # This script is similar to examples/SerialGridSearch.DoubleCouple.py,
    # except here we consider only seven grid points rather than an entire
    # grid, and here the final plots are a comparison of MTUQ and CAP/FK 
    # synthetics rather than a comparison of data and synthetics
    #
    # Because of the idiosyncratic way CAP implements source-time function
    # convolution, it's not expected that CAP and MTUQ synthetics will match 
    # exactly. CAP's "conv" function results in systematic magnitude-
    # dependent shifts between origin times and arrival times. We deal with 
    # this by applying magnitude-dependent time-shifts to MTUQ synthetics 
    # (which normally lack such shifts) at the end of the benchmark. Even with
    # this correction, the match will not be exact because CAP applies the 
    # shifts before tapering and MTUQ after tapering. The resulting mismatch 
    # will usually be apparent in body-wave windows, but not in surface-wave 
    # windows
    #
    # Note that CAP works with dyne,cm and MTUQ works with N,m, so to make
    # comparisons we convert CAP output from the former to the latter
    #
    # The CAP/FK synthetics used in the comparison were generated by 
    # uafseismo/capuaf:46dd46bdc06e1336c3c4ccf4f99368fe99019c88
    # using the following commands
    #
    # source #0 (explosion):
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R0/1.178/90/45/90 20090407201255351
    #
    # source #1 (on-diagonal)
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R-0.333/0.972/90/45/90 20090407201255351
    #
    # source #2 (on-diagonal)
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R-0.333/0.972/45/90/180 20090407201255351
    #
    # source #3 (on-diagonal):
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R-0.333/0.972/45/90/0 20090407201255351
    #
    # source #4 (off-diagonal):
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R0/0/90/90/90 20090407201255351
    #
    # source #5 (off-diagonal):
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R0/0/90/0/0 20090407201255351
    #
    # source #6 (off-diagonal):
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R0/0/0/90/180 20090407201255351
    #

"""


ArgparseDefinitions="""
    # by default, the script runs with figure generation and error checking
    # turned on
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_checks', action='store_true')
    parser.add_argument('--no_figures', action='store_true')
    args = parser.parse_args()
    run_checks = (not args.no_checks)
    run_figures = (not args.no_figures)

"""


Paths_BenchmarkCAP="""
    from mtuq.cap.util import\\
        get_synthetics_cap, get_synthetics_mtuq,\\
        get_data_cap, compare_cap_mtuq


    # the following directories correspond to the moment tensors in the list 
    # "grid" below
    paths = []
    paths += [join(path_mtuq(), 'data/tests/benchmark_cap/20090407201255351/0')]
    paths += [join(path_mtuq(), 'data/tests/benchmark_cap/20090407201255351/1')]
    paths += [join(path_mtuq(), 'data/tests/benchmark_cap/20090407201255351/2')]
    paths += [join(path_mtuq(), 'data/tests/benchmark_cap/20090407201255351/3')]
    paths += [join(path_mtuq(), 'data/tests/benchmark_cap/20090407201255351/4')]
    paths += [join(path_mtuq(), 'data/tests/benchmark_cap/20090407201255351/5')]
    paths += [join(path_mtuq(), 'data/tests/benchmark_cap/20090407201255351/6')]

"""


PathsComments="""
    #
    # Here we specify the data used for the inversion. The event is an 
    # Mw~4 Alaska earthquake
    #
"""


Paths_Syngine="""
    path_data=    join(path_mtuq(), 'data/examples/20090407201255351/*.[zrt]')
    path_weights= join(path_mtuq(), 'data/examples/20090407201255351/weights.dat')
    event_name=   '20090407201255351'
    model=        'ak135'

"""


Paths_FK="""
    path_greens=  join(path_mtuq(), 'data/tests/benchmark_cap/greens/scak')
    path_data=    join(path_mtuq(), 'data/examples/20090407201255351/*.[zrt]')
    path_weights= join(path_mtuq(), 'data/examples/20090407201255351/weights.dat')
    event_name=   '20090407201255351'
    model=        'scak'

"""



DataProcessingComments="""
    #
    # Body- and surface-wave data are processed separately and held separately 
    # in memory
    #
"""


DataProcessingDefinitions="""
    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 0.1,
        freq_max= 0.333,
        pick_type='from_taup_model',
        taup_model=model,
        window_type='cap_bw',
        window_length=15.,
        padding_length=2.,
        weight_type='cap_bw',
        cap_weight_file=path_weights,
        )

    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        pick_type='from_taup_model',
        taup_model=model,
        window_type='cap_sw',
        window_length=150.,
        padding_length=10.,
        weight_type='cap_sw',
        cap_weight_file=path_weights,
        )

"""


MisfitComments="""
    #
    # We define misfit as a sum of indepedent body- and surface-wave 
    # contributions
    #
"""


MisfitDefinitions="""
    misfit_bw = Misfit(
        time_shift_max=2.,
        time_shift_groups=['ZR'],
        )

    misfit_sw = Misfit(
        time_shift_max=10.,
        time_shift_groups=['ZR','T'],
        )

"""


Grid_DoubleCouple="""
    #
    # Next we specify the source parameter grid
    #

    grid = DoubleCoupleGridRandom(
        npts=50000,
        magnitude=4.5)

    wavelet = Trapezoid(
        magnitude=4.5)

"""


Grid_DoubleCoupleMagnitudeDepth="""
    #
    # Next we specify the source parameter grid
    #

    magnitudes = np.array(
         # moment magnitude (Mw)
        [4.3, 4.4, 4.5,     
         4.6, 4.7, 4.8]) 

    depths = np.array(
         # depth in meters
        [25000, 30000, 35000, 40000,                    
         45000, 50000, 55000, 60000])         

    grid = DoubleCoupleGridRegular(
        npts_per_axis=25,
        magnitude=magnitudes)

    wavelet = Trapezoid(
        magnitude=np.mean(magnitudes))

"""


Grid_FullMomentTensor="""
    #
    # Next we specify the source parameter grid
    #

    grid = FullMomentTensorGridRandom(
        npts=1000000,
        magnitude=4.5)

    wavelet = Trapezoid(
        magnitude=4.5)

"""


Grid_TestDoubleCoupleMagnitudeDepth="""
    #
    # Next we specify the source parameter grid
    #

    depths = np.array(
         # depth in meters
        [34000])

    grid = DoubleCoupleGridRegular(
        npts_per_axis=5,
        magnitude=[4.4, 4.5, 4.6])

    wavelet = Trapezoid(
        magnitude=4.5)

"""


Grid_TestGraphics="""
    mt = np.sqrt(1./3.)*np.array([1., 1., 1., 0., 0., 0.]) # explosion
    mt *= 1.e16

    wavelet = Trapezoid(
        magnitude=4.5)
"""


Grid_BenchmarkCAP="""
    #
    # Next we specify the source parameter grid
    #

    grid = [
       # Mrr, Mtt, Mpp, Mrt, Mrp, Mtp
       np.sqrt(1./3.)*np.array([1., 1., 1., 0., 0., 0.]), # explosion
       np.array([1., 0., 0., 0., 0., 0.]), # source 1 (on-diagonal)
       np.array([0., 1., 0., 0., 0., 0.]), # source 2 (on-diagonal)
       np.array([0., 0., 1., 0., 0., 0.]), # source 3 (on-diagonal)
       np.sqrt(1./2.)*np.array([0., 0., 0., 1., 0., 0.]), # source 4 (off-diagonal)
       np.sqrt(1./2.)*np.array([0., 0., 0., 0., 1., 0.]), # source 5 (off-diagonal)
       np.sqrt(1./2.)*np.array([0., 0., 0., 0., 0., 1.]), # source 6 (off-diagonal)
       ]

    Mw = 4.5
    M0 = 10.**(1.5*Mw + 9.1) # units: N-m
    for mt in grid:
        mt *= np.sqrt(2)*M0

    wavelet = Trapezoid(
        magnitude=Mw)

"""


Main_GridSearch_DoubleCouple="""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD


    #
    # The main I/O work starts now
    #

    if comm.rank==0:
        print 'Reading data...\\n'
        data = read(path_data, format='sac', 
            event_id=event_name,
            tags=['units:cm', 'type:velocity']) 

        data.sort_by_distance()

        stations = data.get_stations()
        origins = data.get_origins()

        print 'Processing data...\\n'
        data_bw = data.map(process_bw, stations, origins)
        data_sw = data.map(process_sw, stations, origins)

        print 'Reading Greens functions...\\n'
        greens = get_greens_tensors(stations, origins, model=model)

        print 'Processing Greens functions...\\n'
        greens.convolve(wavelet)
        greens_bw = greens.map(process_bw, stations, origins)
        greens_sw = greens.map(process_sw, stations, origins)

    else:
        data_bw = None
        data_sw = None
        greens_bw = None
        greens_sw = None

    data_bw = comm.bcast(data_bw, root=0)
    data_sw = comm.bcast(data_sw, root=0)
    greens_bw = comm.bcast(greens_bw, root=0)
    greens_sw = comm.bcast(greens_sw, root=0)


    #
    # The main computational work starts now
    #

    if comm.rank==0:
        print 'Carrying out grid search...\\n'

    results = grid_search_mt(
        [data_bw, data_sw], [greens_bw, greens_sw],
        [misfit_bw, misfit_sw], grid)

    results = comm.gather(results, root=0)

"""


Main_GridSearch_DoubleCoupleMagnitudeDepth="""
    #
    # The main I/O work starts now
    #

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    nproc = comm.Get_size()

    if rank==0:
        print 'Reading data...\\n'
        data = read(path_data, format='sac', 
            event_id=event_name,
            tags=['units:cm', 'type:velocity']) 

        data.sort_by_distance()

        stations = data.get_stations()
        origins = data.get_origins()


        print 'Processing data...\\n'
        data_bw = data.map(process_bw, stations, origins)
        data_sw = data.map(process_sw, stations, origins)

    else:
        data_bw = None
        data_sw = None

    data_bw = comm.bcast(data_bw, root=0)
    data_sw = comm.bcast(data_sw, root=0)

    greens_bw = {}
    greens_sw = {}

    if rank==0:
        print 'Reading Greens functions...\\n'

        for _i, depth in enumerate(depths):
            print '  Depth %d of %d' % (_i+1, len(depths))

            origins = deepcopy(origins)
            [setattr(origin, 'depth_in_m', depth) for origin in origins]

            greens = get_greens_tensors(stations, origins, model=model)

            greens.convolve(wavelet)
            greens_bw[depth] = greens.map(process_bw, stations, origins)
            greens_sw[depth] = greens.map(process_sw, stations, origins)

        print ''

    greens_bw = comm.bcast(greens_bw, root=0)
    greens_sw = comm.bcast(greens_sw, root=0)


    #
    # The main computational work starts now
    #

    if rank==0:
        print 'Carrying out grid search...\\n'

    results = grid_search_mt_depth(
        [data_bw, data_sw], [greens_bw, greens_sw],
        [misfit_bw, misfit_sw], grid, depths)

    # gathering results
    results_unsorted = comm.gather(results, root=0)
    if rank==0:
        results = {}
        for depth in depths:
            results[depth] = np.concatenate(
                [results_unsorted[iproc][depth] for iproc in range(nproc)])
"""


Main_SerialGridSearch_DoubleCouple="""
    #
    # The main I/O work starts now
    #

    print 'Reading data...\\n'
    data = read(path_data, format='sac',
        event_id=event_name,
        tags=['units:cm', 'type:velocity']) 

    data.sort_by_distance()

    stations = data.get_stations()
    origins = data.get_origins()


    print 'Processing data...\\n'
    data_bw = data.map(process_bw, stations, origins)
    data_sw = data.map(process_sw, stations, origins)

    print 'Reading Greens functions...\\n'
    greens = get_greens_tensors(stations, origins, model=model)

    print 'Processing Greens functions...\\n'
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw, stations, origins)
    greens_sw = greens.map(process_sw, stations, origins)


    #
    # The main computational work starts nows
    #

    print 'Carrying out grid search...\\n'

    results = grid_search_mt(
        [data_bw, data_sw], [greens_bw, greens_sw],
        [misfit_bw, misfit_sw], grid, verbose=True)

"""


Main_TestGridSearch_DoubleCoupleMagnitudeDepth="""
    #
    # The main I/O work starts now
    #

    print 'Reading data...\\n'
    data = read(path_data, format='sac', 
        event_id=event_name,
        tags=['units:cm', 'type:velocity']) 

    data.sort_by_distance()

    stations = data.get_stations()
    origins = data.get_origins()


    print 'Processing data...\\n'
    data_bw = data.map(process_bw, stations, origins)
    data_sw = data.map(process_sw, stations, origins)

    greens_bw = {}
    greens_sw = {}

    print 'Reading Greens functions...\\n'

    for _i, depth in enumerate(depths):
        origins = deepcopy(origins)
        [setattr(origin, 'depth_in_m', depth) for origin in origins]

        db = open_db(path_greens, format='FK', model=model)
        greens = db.get_greens_tensors(stations, origins)

        greens.convolve(wavelet)
        greens_bw[depth] = greens.map(process_bw, stations, origins)
        greens_sw[depth] = greens.map(process_sw, stations, origins)


    #
    # The main computational work starts now
    #

    print 'Carrying out grid search...\\n'

    results = grid_search_mt_depth(
        [data_bw, data_sw], [greens_bw, greens_sw],
        [misfit_bw, misfit_sw], grid, depths, verbose=False)

"""


Main_TestGraphics="""

    print 'Reading data...\\n'
    data = read(path_data, format='sac',
        event_id=event_name,
        tags=['units:cm', 'type:velocity'])

    data.sort_by_distance()

    stations = data.get_stations()
    origins = data.get_origins()


    print 'Processing data...\\n'
    data_bw = data.map(process_bw, stations, origins)
    data_sw = data.map(process_sw, stations, origins)

    print 'Reading Greens functions...\\n'
    db = open_db(path_greens, format='FK', model=model)
    greens = db.get_greens_tensors(stations, origins)

    print 'Processing Greens functions...\\n'
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw, stations, origins)
    greens_sw = greens.map(process_sw, stations, origins)


    #
    # Start generating figures
    #

    print 'Figure 1 of 3\\n'

    plot_data_greens_mt('test_graphics1.png',
        [data_bw, data_sw], [greens_bw, greens_sw],
        [misfit_bw, misfit_sw], mt, header=False)

    print 'Figure 2 of 3\\n'

    plot_data_greens_mt('test_graphics2.png',
        [data_bw, data_sw], [greens_bw, greens_sw],
        [misfit_bw, misfit_sw], mt, header=True)

    print 'Figure 3 of 3\\n'

    plot_beachball('test_graphics3.png', mt)

    print 'Finished\\n'
"""



WrapUp_GridSearch_DoubleCouple="""
    if comm.rank==0:
        print 'Saving results...\\n'

        results = np.concatenate(results)

        best_mt = grid.get(results.argmin())

        plot_data_greens_mt(event_name+'.png',
            [data_bw, data_sw], [greens_bw, greens_sw],
            [misfit_bw, misfit_sw], best_mt)

        plot_beachball(event_name+'_beachball.png', best_mt)

        print 'Finished\\n'

"""


WrapUp_GridSearch_DoubleCoupleMagnitudeDepth="""
    if comm.rank==0:
        print 'Saving results...\\n'

        best_misfit = {}
        best_mt = {}
        for depth in depths:
            best_misfit[depth] = results[depth].min()
            best_mt[depth] = grid.get(results[depth].argmin())

        filename = event_name+'_beachball_vs_depth.png'
        beachball_vs_depth(filename, best_mt)

        filename = event_name+'_misfit_vs_depth.png'
        misfit_vs_depth(filename, best_misfit)

        print 'Finished\\n'
"""


WrapUp_SerialGridSearch_DoubleCouple="""
    best_mt = grid.get(results.argmin())

    plot_data_greens_mt(event_name+'.png',
        [data_bw, data_sw], [greens_bw, greens_sw],
        [misfit_bw, misfit_sw], best_mt)

    plot_beachball(event_name+'_beachball.png', best_mt)

    print 'Finished\\n'

"""


WrapUp_TestGridSearch_DoubleCouple="""
    best_mt = grid.get(results.argmin())

    if run_figures:
        plot_data_greens_mt(event_name+'.png',
            [data_bw, data_sw], [greens_bw, greens_sw],
            [misfit_bw, misfit_sw], best_mt)

        plot_beachball(event_name+'_beachball.png', best_mt)


    if run_checks:
        def isclose(a, b, atol=1.e6, rtol=1.e-8):
            # the default absolute tolerance (1.e6) is several orders of 
            # magnitude less than the moment of an Mw=0 event

            for _a, _b, _bool in zip(
                a, b, np.isclose(a, b, atol=atol, rtol=rtol)):

                print '%s:  %.e <= %.1e + %.1e * %.1e' %\\
                    ('passed' if _bool else 'failed', abs(_a-_b), atol, rtol, abs(_b))
            print ''

            return np.all(
                np.isclose(a, b, atol=atol, rtol=rtol))

        if not isclose(
            best_mt,
            np.array([
                -1.92678437e+15,
                -1.42813064e+00,
                 1.92678437e+15,
                 2.35981928e+15,
                 6.81221149e+14,
                 1.66864422e+15,
                 ])
            ):
            raise Exception(
                "Grid search result differs from previous mtuq result")

        print 'SUCCESS\\n'
"""


WrapUp_TestGridSearch_DoubleCoupleMagnitudeDepth="""
    best_misfit = {}
    best_mt = {}
    for depth in depths:
        best_misfit[depth] = results[depth].min()
        best_mt[depth] = grid.get(results[depth].argmin())

    if run_figures:
        filename = event_name+'_beachball_vs_depth.png'
        beachball_vs_depth(filename, best_mt)

        filename = event_name+'_misfit_vs_depth.png'
        misfit_vs_depth(filename, best_misfit)

    if run_checks:
        pass

    print 'SUCCESS\\n'

"""


Main_BenchmarkCAP="""
    #
    # The benchmark starts now
    #

    print 'Reading data...\\n'
    data = read(path_data, format='sac', 
        event_id=event_name,
        tags=['units:cm', 'type:velocity']) 

    data.sort_by_distance()

    stations = data.get_stations()
    origins = data.get_origins()


    print 'Processing data...\\n'
    data_bw = data.map(process_bw, stations, origins)
    data_sw = data.map(process_sw, stations, origins)

    print 'Reading Greens functions...\\n'
    db = open_db(path_greens, format='FK', model=model)
    greens = db.get_greens_tensors(stations, origins)

    print 'Processing Greens functions...\\n'
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw, stations, origins)
    greens_sw = greens.map(process_sw, stations, origins)


    depth = int(origins[0].depth_in_m/1000.)+1
    name = '_'.join([model, str(depth), event_name])


    print 'Comparing waveforms...'

    for _i, mt in enumerate(grid):
        print '  %d of %d' % (_i+1, len(grid))

        cap_bw, cap_sw = get_synthetics_cap(
            data_bw, data_sw, paths[_i], name)

        mtuq_bw, mtuq_sw = get_synthetics_mtuq(
            data_bw, data_sw, greens_bw, greens_sw, mt)

        if run_figures:
            plot_data_synthetics('cap_vs_mtuq_'+str(_i)+'.png',
                cap_bw, cap_sw, mtuq_bw, mtuq_sw, 
                trace_labels=False)

        if run_checks:
            compare_cap_mtuq(
                cap_bw, cap_sw, mtuq_bw, mtuq_sw)

    if run_figures:
        # "bonus" figure comparing how CAP processes observed data with how
        # MTUQ processes observed data
        mtuq_sw, mtuq_bw = data_bw, data_sw

        cap_sw, cap_bw = get_data_cap(
            data_bw, data_sw, paths[0], name)

        plot_data_synthetics('cap_vs_mtuq_data.png',
            cap_bw, cap_sw, mtuq_bw, mtuq_sw, 
            trace_labels=False, normalize=False)

    print '\\nSUCCESS\\n'

"""


if __name__=='__main__':
    import os
    import re

    from mtuq.util import path_mtuq, replace
    os.chdir(path_mtuq())


    with open('examples/GridSearch.DoubleCouple.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(Imports)
        file.write(Docstring_GridSearch_DoubleCouple)
        file.write(PathsComments)
        file.write(Paths_Syngine)
        file.write(DataProcessingComments)
        file.write(DataProcessingDefinitions)
        file.write(MisfitComments)
        file.write(MisfitDefinitions)
        file.write(Grid_DoubleCouple)
        file.write(Main_GridSearch_DoubleCouple)
        file.write(WrapUp_GridSearch_DoubleCouple)


    with open('examples/GridSearch.DoubleCouple+Magnitude+Depth.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(
            replace(
            Imports,
            'grid_search_mt',
            'grid_search_mt_depth',
            'DoubleCoupleGridRandom',
            'DoubleCoupleGridRegular',
            'plot_beachball',
            'beachball_vs_depth, misfit_vs_depth',
            ))
        file.write(Docstring_GridSearch_DoubleCoupleMagnitudeDepth)
        file.write(PathsComments)
        file.write(Paths_Syngine)
        file.write(DataProcessingComments)
        file.write(DataProcessingDefinitions)
        file.write(MisfitDefinitions)
        file.write(Grid_DoubleCoupleMagnitudeDepth)
        file.write(Main_GridSearch_DoubleCoupleMagnitudeDepth)
        file.write(WrapUp_GridSearch_DoubleCoupleMagnitudeDepth)


    with open('examples/GridSearch.FullMomentTensor.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(
            replace(
            Imports,
            'DoubleCoupleGridRandom',
            'FullMomentTensorGridRandom',
            ))
        file.write(Docstring_GridSearch_FullMomentTensor)
        file.write(Paths_Syngine)
        file.write(DataProcessingComments)
        file.write(DataProcessingDefinitions)
        file.write(MisfitComments)
        file.write(MisfitDefinitions)
        file.write(Grid_FullMomentTensor)
        file.write(Main_GridSearch_DoubleCouple)
        file.write(WrapUp_GridSearch_DoubleCouple)


    with open('setup/chinook/examples/CapStyleGridSearch.DoubleCouple.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(
            replace(
            Imports,
            'syngine',
            'fk',
            ))
        file.write(Docstring_CapStyleGridSearch_DoubleCouple)
        file.write(
            replace(
            Paths_FK,
           r"path_greens=.*",
           r"path_greens= '/import/c1/ERTHQUAK/rmodrak/wf/FK_synthetics/scak'",
            ))
        file.write(
            replace(
            DataProcessingDefinitions,
            'pick_type=.*',
            "pick_type='from_fk_metadata',",
            'taup_model=.*,',
            'fk_database=path_greens,',
            ))
        file.write(MisfitDefinitions)
        file.write(Grid_DoubleCouple)
        file.write(
            replace(
            Main_GridSearch_DoubleCouple,
            'greens = get_greens_tensors\(stations, origins, model=model\)',
            'db = open_db(path_greens, format=\'FK\', model=model)\n        '
           +'greens = db.get_greens_tensors(stations, origins)',
            ))
        file.write(WrapUp_GridSearch_DoubleCouple)


    with open('setup/chinook/examples/CapStyleGridSearch.DoubleCouple+Magnitude+Depth.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(
            replace(
            Imports,
            'syngine',
            'fk',
            'grid_search_mt',
            'grid_search_mt_depth',
            'DoubleCoupleGridRandom',
            'DoubleCoupleGridRegular',
            'plot_beachball',
            'beachball_vs_depth, misfit_vs_depth',
            ))
        file.write(Docstring_CapStyleGridSearch_DoubleCouple)
        file.write(
            replace(
            Paths_FK,
           r"path_greens=.*",
           r"path_greens= '/import/c1/ERTHQUAK/ERTHQUAK/FK_synthetics/scak'",
            ))
        file.write(
            replace(
            DataProcessingDefinitions,
            'pick_type=.*',
            "pick_type='from_fk_metadata',",
            'taup_model=.*,',
            'fk_database=path_greens,',
            ))
        file.write(MisfitDefinitions)
        file.write(Grid_DoubleCoupleMagnitudeDepth)
        file.write(
            replace(
            Main_GridSearch_DoubleCoupleMagnitudeDepth,
            'greens = get_greens_tensors\(stations, origins, model=model\)',
            'db = open_db(path_greens, format=\'FK\', model=model)\n            '
           +'greens = db.get_greens_tensors(stations, origins)',
            ))
        file.write(WrapUp_GridSearch_DoubleCoupleMagnitudeDepth)


    with open('examples/SerialGridSearch.DoubleCouple.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(
            replace(
            Imports,
            'grid_search.mpi',
            'grid_search.serial',
            ))
        file.write(Docstring_SerialGridSearch_DoubleCouple)
        file.write(PathsComments)
        file.write(Paths_Syngine)
        file.write(DataProcessingComments)
        file.write(DataProcessingDefinitions)
        file.write(MisfitComments)
        file.write(MisfitDefinitions)
        file.write(Grid_DoubleCouple)
        file.write(Main_SerialGridSearch_DoubleCouple)
        file.write(WrapUp_SerialGridSearch_DoubleCouple)


    with open('tests/test_grid_search_mt.py', 'w') as file:
        file.write(
            replace(
            Imports,
            'grid_search.mpi',
            'grid_search.serial',
            'DoubleCoupleGridRandom',
            'DoubleCoupleGridRegular',
            ))
        file.write(Docstring_TestGridSearch_DoubleCouple)
        file.write(ArgparseDefinitions)
        file.write(Paths_FK)
        file.write(
            replace(
            DataProcessingDefinitions,
            'pick_type=.*',
            "pick_type='from_fk_metadata',",
            'taup_model=.*,',
            'fk_database=path_greens,',
            ))
        file.write(MisfitDefinitions)
        file.write(
            replace(
            Grid_DoubleCouple,
            'Random',
            'Regular',
            'npts=.*,',
            'npts_per_axis=5,',
            ))
        file.write(
            replace(
            Main_SerialGridSearch_DoubleCouple,
            'greens = get_greens_tensors\(stations, origins, model=model\)',
            'db = open_db(path_greens, format=\'FK\', model=model)\n    '
           +'greens = db.get_greens_tensors(stations, origins)',
            'verbose=True',
            'verbose=False',
            ))
        file.write(WrapUp_TestGridSearch_DoubleCouple)


    with open('tests/test_grid_search_mt_depth.py', 'w') as file:
        file.write(
            replace(
            Imports,
            'grid_search.mpi',
            'grid_search.serial',
            'grid_search_mt',
            'grid_search_mt_depth',
            'DoubleCoupleGridRandom',
            'DoubleCoupleGridRegular',
            'plot_beachball',
            'beachball_vs_depth, misfit_vs_depth',
            ))
        file.write(Docstring_TestGridSearch_DoubleCoupleMagnitudeDepth)
        file.write(ArgparseDefinitions)
        file.write(Paths_FK)
        file.write(
            replace(
            DataProcessingDefinitions,
            'pick_type=.*',
            "pick_type='from_fk_metadata',",
            'taup_model=.*,',
            'fk_database=path_greens,',
            ))
        file.write(MisfitDefinitions)
        file.write(Grid_TestDoubleCoupleMagnitudeDepth)
        file.write(Main_TestGridSearch_DoubleCoupleMagnitudeDepth)
        file.write(WrapUp_TestGridSearch_DoubleCoupleMagnitudeDepth)


    with open('tests/benchmark_cap.py', 'w') as file:
        file.write(
            replace(
            Imports,
            'syngine',
            'fk',
            'plot_data_greens_mt',
            'plot_data_synthetics',
            ))
        file.write(Docstring_BenchmarkCAP)
        file.write(ArgparseDefinitions)
        file.write(Paths_BenchmarkCAP)
        file.write(
            replace(
            Paths_FK,
            'data/examples/20090407201255351/weights.dat',
            'data/tests/benchmark_cap/20090407201255351/weights.dat',
            ))
        file.write(
            replace(
            DataProcessingDefinitions,
            'padding_length=.*',
            'padding_length=0,',
            'pick_type=.*',
            "pick_type='from_fk_metadata',",
            'taup_model=.*,',
            'fk_database=path_greens,',
            ))
        file.write(
            replace(
            MisfitDefinitions,
            'time_shift_max=.*',
            'time_shift_max=0.,',
            ))
        file.write(Grid_BenchmarkCAP)
        file.write(Main_BenchmarkCAP)


    with open('tests/test_graphics.py', 'w') as file:
        file.write(Imports)
        file.write(Docstring_TestGraphics)
        file.write(Paths_FK)
        file.write(
            replace(
            DataProcessingDefinitions,
            'pick_type=.*',
            "pick_type='from_fk_metadata',",
            'taup_model=.*,',
            'fk_database=path_greens,',
            ))
        file.write(MisfitDefinitions)
        file.write(Grid_TestGraphics)
        file.write(Main_TestGraphics)


