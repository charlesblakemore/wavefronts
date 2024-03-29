import os

from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from matplotlib.colors import LightSource
from matplotlib.gridspec import GridSpec

import scipy.interpolate as interpolate
import scipy.ndimage as ndimage

from iminuit import Minuit


plt.rcParams.update({'font.size': 16})



data_dir = os.path.abspath('../data/')
plot_dir = os.path.abspath('../plots/')


#############################################################




def _load_power_distribution(power_path, max_power=None):
    '''
    Internal subroutine to load the power distribution data saved by
    the Thorlabs WFS, in the default CSV format.

    INPUTS

        power_path - str, path to the (assumed .csv) data with the 
            power distribution from a particular wavefront snapshot

        max_power - float, an optional limit to apply when importing
            since the detector software occasionally likes to add 
            anomalously large numbers at the edges of the beam

    OUTPUTS

        power_matrix - np.matrix, the power in a 2D array, including a
            mask for the points where power==0.0, i.e. the detector 
            cut that pixel off due to low power.
    '''

    ### Read the entire text-encoded file into memory
    power_file = open(power_path).readlines()

    ### Loop over all the lines within the file and find the one
    ### defining the data itself
    found_data = False
    for lineind, line in enumerate(power_file):
        if 'POWER DISTRIBUTION [a.u.]' in line:
            dataset_ind = lineind + 3
            found_data = True
        if found_data:
            break

    ### Determine how many samples there are in X and Y
    lens_str = power_file[dataset_ind-2:dataset_ind]
    lens = list(map(int, np.genfromtxt(lens_str, delimiter=',')[:,-1]))

    ### Load the data into a numpy array
    power_data = power_file[dataset_ind:dataset_ind+lens[1]]
    power_arr = np.genfromtxt(power_data, delimiter=',')

    if max_power is not None:
        power_arr[np.where(power_arr > max_power)] = 0.0

    ### Crop around the actual data, and take a transpose so that the
    ### first axis is actually the x-axis ('ij' indexing, as matplotlib
    ### and numpy call it)
    power_arr = power_arr[:lens[1],:lens[0]].T

    ### Define a matrix with a mask where the value of the measured
    ### power identically zero, i.e. where the detector set an internal
    ### threshold value.
    power_matrix = np.ma.masked_where(power_arr==0.0, power_arr)

    return power_matrix



def _radial_crop(matrix, xcoord, ycoord, max_radius, center=None):
    '''
    Internal subroutine to crop the power distribution and/or 
    wavefront measurements from data saved by the Thorlabs WFS, 
    which sometimes does weird things and adds data that is not
    visible on the GUI (and likely not even there)

    INPUTS

        matrix - np.matrix, the matrix to crop

        xcoord - np.array, the horizontal (first axis of the matrix)
            data points, used to center the naive radial cropping

        ycoord - np.array, the vertical (second axis) data points
            to center the cropping

        max_radius - float, the cropping parameter to mask all data 
            outside of a certain radius value (in mm). Sometimes 
            the software adds numbers in the corners, outside of an
            annular region of all zeros around a measured beam

        center - 2-tuple of floats, origin about which to crop, 
            usually assumed to be zero

    OUTPUTS

        cropped_matrix - the new matrix, with values adjusted to 0.0 
            outside of the defined circle, and an updated internal 
            mask object for consistency
    '''

    if center is None:
        center = (0.0, 0.0)

    ### Build a meshgrid of offset coordinates for easy calculation
    ### of the radii and eventual cropping
    xx, yy = np.meshgrid(xcoord - center[0], \
                         ycoord - center[1], \
                         indexing='ij')

    ### Calculate the distance of each of the points from the imagined
    ### center of the cropping circle
    radii = np.sqrt(xx**2 + yy**2)

    ### Define a boolean array for the points inside the desired crop
    inside_circle = radii < max_radius

    ### Build the new data and mask arrays
    new_data = matrix.data * inside_circle
    new_mask = np.logical_or(matrix.mask, np.logical_not(inside_circle))

    ### Define a new numpy matrix object to return
    new_matrix = np.ma.masked_where(new_mask, new_data)

    return new_matrix




def _load_wavefront(meas_path):
    '''
    Internal subroutine to load the power distribution data saved by
    the Thorlabs WFS, in the default CSV format.

    INPUTS

        meas_path - str, path to the (assumed .csv) data with the 
            wavefront shape/phase

    OUTPUTS

        xcoord - np.ndarray, X-coords of the points at which the wavefront
            is sampled

        ycoord - np.ndarray, Y-coords of the sampled points

        wavefront_matrix - np.matrix, the wavefront in a 2D array, 
            including a mask for the points where wavefront==NaN, i.e. 
            the detector cut that pixel off due to low power. The
            wavefront is usually given in microns (um)
    '''

    ### Read the entire text-encoded file into memory
    wavefront_file = open(meas_path, errors='replace').readlines()

    ### Loop over all the lines within the file and find the one
    ### defining the data itself
    found_data = False
    for lineind, line in enumerate(wavefront_file):
        if 'y / x [mm]' in line:
            dataset_ind = lineind+1
            found_data = True
        if found_data:
            break

    ### Determine how many samples there are in X and Y
    lens_str = wavefront_file[dataset_ind-3:dataset_ind-1]
    lens = list(map(int, np.genfromtxt(lens_str, delimiter=',')[:,-1]))

    ### Extract the positions of the samples along one paricular axis
    xcoord = np.genfromtxt([wavefront_file[dataset_ind-1]], delimiter=',')[1:-1]

    ### Pull out the data, ignoring the 'xcoord' axis
    wavefront_data = wavefront_file[dataset_ind:dataset_ind+lens[1]]
    wavefront_arr = np.genfromtxt(wavefront_data, delimiter=',')

    ### Extract the ycoord axis, since the values of the y-coordinate
    ### are written to the CSV file in the same rows as the data itself
    ycoord = wavefront_arr[:,0]

    ### Cut off the extra columns, and take the transpose so that the
    ### first axis is the x-coordinate
    wavefront_arr = wavefront_arr[:,1:-1].T
    nan_inds = np.isnan(wavefront_arr)

    ### Define a matrix with a mask where the wavefront was not measured
    ### (due to the incident power being too low)
    wavefront_matrix = np.ma.masked_where(nan_inds, wavefront_arr)

    return xcoord, ycoord, wavefront_matrix 




def load_wfs_data(meas_name='', meas_path='', power_path='', \
                  meas_suffix='_meas.csv', power_suffix='_power.csv', \
                  load_power=True, max_power=None, max_radius=None, \
                  crop_center=None):
    '''
    Load data from Thorlabs WFS series detectors, making many assumptions
    about the way the data is saved, both with regard to the CSV format
    from the Thorlabs Wavefront Sensor desktop acpplication, as well as the 
    naming convention of various files.

    Optionally, direct file paths for both a wavefront measurement and a
    corresponding power measurement can be provided directly

    INPUTS

        meas_name - str, base word/phase to the wavefront and power
            measurement files

        meas_path - str, direct path to the (assumed .csv) data with the 
            wavefront shape/phase

        power_path - str, direct path to the (assumed .csv) data with the 
            wavefront intensity distribution

        meas_suffix - str, if a base name was provided, this is the suffix
            for the wavefront data specifically

        power_suffix - str, suffix for the power data

        load_power - boolean, option to load the power data or not

        max_power - float, maximum value to cutoff

        max_radius - float, radius of a circular crop applied to the power
            data, in mm (physical units)

        crop_center - 2-tuple of float (or other 2-index iterable), the 
            origin about which to perform the circular crop

    OUTPUTS

        xcoord - np.ndarray, X-coords of the points at which the wavefront
            is sampled

        ycoord - np.ndarray, Y-coords of the sampled points

        wavefront_matrix - np.matrix, the wavefront in a 2D array, 
            including a mask for the points where wavefront==NaN, i.e. 
            the detector cut that pixel off due to low power. The
            wavefront is usually given in microns (um)

        power_matrix - np.matrix, the power in a 2D array, including a
            mask for the points where power==0.0, i.e. the detector 
            cut that pixel off due to low power (optional)
    '''

    ### Teensy bit of argument handling
    if (not len(meas_name)) and (not len(meas_path) or not len(power_path)):
        raise ValueError('Need to provide a measurement name from '\
                         +'the default data directory, or a full path '\
                         +'to the desired file(s)\n')

    ### Build up the data paths if they weren't provided directly
    if not meas_path:
        meas_path = os.path.join(data_dir, meas_name+meas_suffix)
    if not power_path:
        power_path = os.path.join(data_dir, meas_name+power_suffix)

    ### Load the wavefront using the internally defined routine
    xcoord, ycoord, wavefront_matrix = _load_wavefront(meas_path)

    ### Make sure the coordinates are in ascending order
    xflip = False
    if xcoord[-1] < xcoord[0]:
        xcoord = xcoord[::-1]
        wavefront_matrix = wavefront_matrix[::-1,:]
        xflip = True
    yflip = False
    if ycoord[-1] < ycoord[0]:
        ycoord = ycoord[::-1]
        wavefront_matrix = wavefront_matrix[:,::-1]
        yflip = True

    ### If desired, load and return both the wavefront and the power,
    ### or optionally just the wavefront.
    if load_power:
        power_matrix = _load_power_distribution(power_path, \
                                                max_power=max_power)

        ### Flip the axes if required. This should usually execute a
        ### "yflip" if I remember correctly, but this general approach
        ### will work for any potential software updates
        if xflip:
            power_matrix = power_matrix[::-1,:]
        if yflip:
            power_matrix = power_matrix[:,::-1]

        ### If a max_radius was specified, crop the data as desired.
        if max_radius is not None:
            power_matrix = _radial_crop(power_matrix, xcoord, ycoord, \
                                        max_radius, center=crop_center)

        return xcoord, ycoord, wavefront_matrix, power_matrix
    else:
        return xcoord, ycoord, wavefront_matrix




def plot_3d_wavefront(xcoord, ycoord, wavefront_matrix, power_matrix=None,\
                      shade_with_power=False, invert=False, \
                      cmap='inferno', colorbar=True,\
                      fig=None, ax=None, figsize=(11,8), show=True):
    '''
    Plot data from Thorlabs WFS series detectors as 3D surfaces, where the 
    topology is always set by the phase of the wavefront to aid in 
    visualization, and the color can similarly demonstrate the phase, or 
    optionally can be set proportional to the intensity

    INPUTS

        xcoord - np.ndarray, X-coords of the points at which the wavefront
            is sampled

        ycoord - np.ndarray, Y-coords of the sampled points

        wavefront_matrix - np.matrix, the wavefront in a 2D array, 
            including a mask for the points where wavefront==NaN, i.e. 
            the detector cut that pixel off due to low power. The
            wavefront is usually given in microns (um)

        power_matrix - np.matrix, (optional), the power in a 2D array, 
            including a mask for the points where power==0.0, i.e. the 
            detector cut that pixel off due to low power

        shade_with_power - boolean, option to set color shading to be
            proportional to the intensity

        cmap - colormap object or str

        colorbar - boolean, if NOT shading with power, this option allows
            for a colorbar to help indicate the scale of the wavefront
            fluctuations/deformations

        fig - plt.figure, (optional), existing figure object on which to
            plot, if that was desired

        ax - plt.axes, (optional), existing axes on which to plot

        figsize - 2-tuple, size of the figure in inches

        show - boolean, flag to actually show the plots, in case the 
            user wanted to wait

    OUTPUTS
    
        fig - plt.figure, the figure object on which stuff is plotted

        ax - plt.axes, the axes object where it REALLY is plotted
    '''

    ### Mini argument handling for shading the wavefront
    if shade_with_power and power_matrix is None:
        raise ValueError('If you want to shade with power, provide a power '\
                         +'matrix for the given wavefront data.\n')

    if invert:
        wavefront_matrix *= -1.0

    ### Make the figure and axes objects if they weren't given as inputs
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.gca(projection='3d')

    ### Set some labels just to get it over with
    ax.set_xlabel('X-coordinate [mm]', labelpad=20)
    ax.set_ylabel('Y-coordinate [mm]', labelpad=20)
    ax.set_zlabel('Wavefront [um]', labelpad=20)

    ### Generate the meshgrids for x and y from the input axes
    xx, yy = np.meshgrid(xcoord, ycoord, indexing='ij')

    if shade_with_power and power_matrix is not None:

        ### Build an explicit array (instead of a matrix with a mask) for the
        ### LightSource shading operation
        not_zero_inds = power_matrix.data != 0.0
        power_arr_mod = power_matrix.data * not_zero_inds

        ### The LightSource class to be used ahead requires a callable 
        ### colormap object, not just a string with the name
        if type(cmap) == str:
            cmap = cm.get_cmap(cmap)

        ### Construct a LightSource object point down from the postive
        ### z-axis by setting azimuth=0 and elevation=90 (in degrees)
        ls = LightSource(0, 90)
        rgb = ls.shade(power_arr_mod, cmap=cmap, vert_exag=0.1, blend_mode='soft')
        surf = ax.plot_surface(xx, yy, wavefront_matrix, rstride=1, cstride=1, \
                               facecolors=rgb, linewidth=0, antialiased=False, \
                               shade=False)


    else:
        ### Plot a 3D surface where the coloring mimics the topology
        surf = ax.plot_surface(xx, yy, wavefront_matrix, rstride=1, cstride=1, \
                               cmap=cmap, linewidth=0, antialiased=False, \
                               shade=False)
        if colorbar:
            fig.colorbar(surf, shrink=0.5, aspect=15)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax




def plot_2d_power(xcoord, ycoord, power_matrix, \
                  cmap='inferno', colorbar=True, \
                  xlim=None, ylim=None, fig=None, ax=None, \
                  figsize=(11,8), show=True):
    '''
    Plot data from Thorlabs WFS series detectors as 3D surfaces, where the 
    topology is always set by the phase of the wavefront to aid in 
    visualization, and the color can similarly demonstrate the phase, or 
    optionally can be set proportional to the intensity

    INPUTS

        xcoord - np.ndarray, X-coords of the points at which the wavefront
            is sampled

        ycoord - np.ndarray, Y-coords of the sampled points

        power_matrix - np.matrix, the power in a 2D array, 
            including a mask for the points where power==0.0, i.e. the 
            detector cut that pixel off due to low power

        cmap - colormap object or str

        colorbar - boolean, if NOT shading with power, this option allows
            for a colorbar to help indicate the scale of the wavefront
            fluctuations/deformations

        fig - plt.figure, (optional), existing figure object on which to
            plot, if that was desired

        ax - plt.axes, (optional), existing axes on which to plot

        figsize - 2-tuple, size of the figure in inches

        show - boolean, flag to actually show the plots, in case the 
            user wanted to wait

    OUTPUTS
    
        fig - plt.figure, the figure object on which stuff is plotted

        ax - plt.axes, the axes object where it REALLY is plotted
    '''

    ### Make the figure and axes objects if they weren't given as inputs
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.gca()

    ax.set_title('Power distribution')

    ### Set some labels just to get it over with
    ax.set_xlabel('X-coordinate [mm]', labelpad=20)
    ax.set_ylabel('Y-coordinate [mm]', labelpad=20)

    extent = (np.min(xcoord), np.max(xcoord), np.min(ycoord), np.max(ycoord))

    ### Generate the meshgrids for x and y from the input axes
    xx, yy = np.meshgrid(xcoord, ycoord, indexing='ij')

    contour = plt.imshow(power_matrix.T, origin='lower', \
                         extent=extent, cmap=cmap)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if colorbar:
        fig.colorbar(contour, shrink=0.9, aspect=20)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax




def _gaussian_2d(x, y, A, x0, y0, sigmax, sigmay, theta, const):
    '''
    2-dimensional elliptical gaussian function, with an arbitrary 
    orientation in the xy-plane. The arguments {x, y} and function 
    parameters {x0, y0} are in reference to the standard cartesian
    plane, whereas sigmax refers to the width of the gaussian on 
    along the semimajor axis of the ellipse and sigmay along the 
    semiminor axis. An arbitrary constant is included

    INPUTS

        x - float, X-coordinate (function argument)

        y - float, Y-coordinate (function argument)

        A - float, peak value of the gaussian

        x0 - float, centroid location in X (function paramater)

        y0 - float, centroid location in Y (function parameter)

        sigmax - float, width of the 2D gaussian along the 
            semimajor axis of the ellipse

        sigmay - float, width of the 2D gaussian along the 
            semiminor axis of the ellipse

        theta - float, angle of the semimajor axis relative to 
            the X-axis in the usual cartesian plane, radians

        const - float, constant offset, often kept at 0

    OUTPUTS
    
        - float, value of the 2D gaussian at the requested
            coordinates {x, y}, with the given parameters
        
    '''

    ### Construct some coefficients to use in an easy functional
    ### form that includes an arbitrary angle

    ### This is like the "X" term, including the rotation
    a = np.cos(theta)**2 / (2.0*sigmax**2) \
                    + np.sin(theta)**2 / (2.0*sigmay**2)

    ### This is a "cross-term"
    b = np.sin(2.0*theta) / (4.0*sigmax**2) \
                    - np.sin(2.0*theta) / (4.0*sigmay**2)

    ### This is the "Y" term
    c = np.sin(theta)**2 / (2.0*sigmax**2) \
                    + np.cos(theta)**2 / (2.0*sigmay**2)

    return A*np.exp( -1.0*( a*(x-x0)**2 \
                            + 2.0*b*(x-x0)*(y-y0) \
                            + c*(y-y0)**2 ) ) \
            + const




def fit_2d_gaussian(xcoord, ycoord, data_matrix, fix_theta=False, \
                    theta_init=0.0, fix_const=True, const_init=0.0, \
                    print_fit=False, plot=False, figsize=(14, 6), \
                    resid_figsize=(8,6)):
    '''
    Routine to fit a 2-dimensional array to a 2D elliptical gaussian
    function, for instance to extract beam waist parameters or
    centroid locations in measured intensity distributions from the 
    Thorlabs wavefront sensors.

    INPUTS

        xcoord - ndarray, X-coordinates at which the data is sampled

        ycoord - ndarray, Y-coordinates at which the data is sample

        data_matrix - np.matrix, values of the data in a 2D matrix
            which may include a mask, over which we probably don't
            want to fit

        fix_theta - boolean, flag to fix the orientation of the
            semimajor axis

        theta_init - float, initial value of the angle, and also the
            value that theta gets fixed to if fixing is desired

        fix_const - boolean, flag to fix the constant offset for
            entire function

        const_init - float, initial value of the constant in the 
            fitting, and again, the value it gets fixed to

        print_fit - boolean, flag to print the Minuit.migrad() result

        plot - boolean, flag to plot the fitting result

        figsize - 2-tuple, size of the displayed figure in inches


    OUTPUTS
    
        popt - ndarray, best-fit paramater values, indexed as 
            follows: [A, x0, y0, sigmax, sigmay, theta, const]

        _gaussian_2d - fitting function used for easy user
            interaction
        
    '''

    ### Find the max value of the data, as well as the location
    ### of the max value as an estimate of the centroid location
    maxval = np.amax(data_matrix)
    max_indices = np.unravel_index(data_matrix.argmax(), data_matrix.shape)

    ### Build the meshgrids from the input arrays
    xx, yy = np.meshgrid(xcoord, ycoord, indexing='ij')

    ### Extract the mask, then construct the logical inverse of 
    ### this array for sub-selecting data to fit
    mask = data_matrix.mask
    mask_not = np.logical_not(mask)

    ### Extract the raw data as a ndarray
    data = data_matrix.data

    ### Degrees of freedom in the fit
    npts = np.sum(mask_not)

    ### A derpy scale object as I figure out how to better handle
    ### data without variance
    scale = np.mean(data[mask_not])

    ### Construct a least-squares cost function
    def cost(A, x0, y0, sigmax, sigmay, theta, const):
        func_val = _gaussian_2d(xx, yy, A, x0, y0, sigmax, \
                                sigmay, theta, const)
        resid = (data - func_val)**2
        return (1.0 / (npts - 1.0)) * np.sum(resid[mask_not])

    ### Assemble the Minuit object with cost function and
    ### initial parameter value guesses
    m = Minuit(cost, 
               A = maxval, \
               x0 = xcoord[max_indices[0]], \
               y0 = ycoord[max_indices[1]], \
               sigmax = 0.1*np.abs(np.max(xcoord) - np.min(xcoord)), \
               sigmay = 0.1*np.abs(np.max(ycoord) - np.min(ycoord)), \
               theta = theta_init, \
               const = const_init
               )

    ### Set some limits to make sure the fitting is well-behaved
    m.limits['A'] = (0, np.inf)
    m.limits['x0'] = (np.min(xcoord), np.max(xcoord))
    m.limits['y0'] = (np.min(ycoord), np.max(ycoord))
    m.limits['sigmax'] = (0, np.inf)
    m.limits['sigmay'] = (0, np.inf)
    m.limits['theta'] = (0, np.pi)
    m.limits['const'] = (-maxval, maxval)

    ### Fix any values that the user wants fixed
    if fix_theta:
        m.fixed['theta'] = True
    if fix_const:
        m.fixed['const'] = True

    ### Default values for number of standard deviations to report
    ### in the parameter error value, as well as the verbosity
    ### of the result printing
    m.errordef = 1
    m.print_level = 0

    ### Minimize the cost function
    result = m.migrad(ncall=100000)
    if print_fit:
        print(result)

    ### Construct the output
    popt = np.array( [m.values['A'], m.values['x0'], m.values['y0'], \
                      m.values['sigmax'], m.values['sigmay'], \
                      m.values['theta'], m.values['const']] )

    if plot:
        ### Define the figure object and the two subplots
        fig = plt.figure(figsize=figsize)

        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        fig2 = plt.figure(figsize=resid_figsize)
        ax3 = fig2.add_subplot(111, projection='3d')


        # gs = GridSpec(3, 4, figure=fig)
        # ax1 = fig.add_subplot(gs[0:2,0:2], projection='3d')
        # ax2 = fig.add_subplot(gs[0:2,2:4], projection='3d')
        # ax3 = fig.add_subplot(gs[2,:], projection='3d')

        ax3.set_box_aspect((1, 1, 0.5))

        ### Plot the input data
        surf = ax1.plot_surface(xx, yy, data_matrix, \
                                rstride=1, cstride=1, \
                                cmap='inferno', linewidth=0, \
                                antialiased=False, shade=False)

        ### Plot the reconstructed Gaussian
        fit_matrix = _gaussian_2d(xx, yy, *popt)
        fit = ax2.plot_surface(xx, yy, fit_matrix, \
                               rstride=1, cstride=1, \
                               cmap='inferno', linewidth=0, \
                               antialiased=False, shade=False)

        ### Plot the residuals
        resid_matrix = data_matrix - fit_matrix
        resid_matrix *= 1.0 / popt[0]
        resid = ax3.plot_surface(xx, yy, resid_matrix, \
                                 rstride=1, cstride=1, \
                                 cmap='coolwarm', linewidth=0, \
                                 antialiased=False, shade=False)

        large_waist = np.max(np.abs(popt[3:5]))
        resid_plot_lim = 3.0*large_waist

        ax3.set_xlim3d(-1.0*resid_plot_lim, resid_plot_lim)
        ax3.set_ylim3d(-1.0*resid_plot_lim, resid_plot_lim)

        ### Add some axis labels
        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel('X-coord [mm]', labelpad=10)
            ax.set_ylabel('Y-coord [mm]', labelpad=10)

        ax3.set_zlabel('$\\Delta$ / Peak', labelpad=15)

        ### Label the two subplots appropriately
        ax1.set_title('Measured', fontsize=16)
        ax2.set_title('Reconstructed', fontsize=16)
        ax3.set_title('Residuals', fontsize=16)

        ### Align the Z-axis limits, since X- and Y-axis limts are 
        ### the same by construction
        zlim1 = ax1.get_zlim3d()
        zlim2 = ax2.get_zlim3d()
        if zlim1[1] >= zlim2[1]:
            ax2.set_zlim3d(zlim1)
        else:
            ax1.set_zlim3d(zlim2)

        fig.tight_layout()
        fig2.tight_layout()

        ### Event handler so that side-by-side 3D subplots maintain
        ### the same orientation and axis limits
        def on_move(event):
            if event.inaxes == ax1:
                if ax1.button_pressed in ax1._rotate_btn:
                    ax2.view_init(elev=ax1.elev, azim=ax1.azim)
                    # ax3.view_init(elev=ax1.elev, azim=ax1.azim)

                elif ax1.button_pressed in ax1._zoom_btn:
                    ax2.set_xlim3d(ax1.get_xlim3d())
                    ax2.set_ylim3d(ax1.get_ylim3d())
                    ax2.set_zlim3d(ax1.get_zlim3d())

                    # ax3.set_xlim3d(ax1.get_xlim3d())
                    # ax3.set_ylim3d(ax1.get_ylim3d())

            elif event.inaxes == ax2:
                if ax2.button_pressed in ax2._rotate_btn:
                    ax1.view_init(elev=ax2.elev, azim=ax2.azim)
                    # ax3.view_init(elev=ax2.elev, azim=ax2.azim)

                elif ax2.button_pressed in ax2._zoom_btn:
                    ax1.set_xlim3d(ax2.get_xlim3d())
                    ax1.set_ylim3d(ax2.get_ylim3d())
                    ax1.set_zlim3d(ax2.get_zlim3d())

                    # ax3.set_xlim3d(ax2.get_xlim3d())
                    # ax3.set_ylim3d(ax2.get_ylim3d())

            # elif event.inaxes == ax3:
            #     if ax3.button_pressed in ax3._rotate_btn:
            #         ax1.view_init(elev=ax3.elev, azim=ax3.azim)
            #         ax2.view_init(elev=ax3.elev, azim=ax3.azim)

            #     # elif ax2.button_pressed in ax2._zoom_btn:
            #     #     ax1.set_xlim3d(ax3.get_xlim3d())
            #     #     ax1.set_ylim3d(ax3.get_ylim3d())

            #     #     ax2.set_xlim3d(ax3.get_xlim3d())
            #     #     ax2.set_ylim3d(ax3.get_ylim3d())

            else:
                return
            fig.canvas.draw_idle()

        ### Connect the event handler to the figure and canvas
        c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)

        plt.show()

    return popt, _gaussian_2d




def _pad_wavefront(wavefront_matrix, pad_length=5):

    '''
    Routine to resample a wavefront surface over a new set of points.

    INPUTS

        wavefront_matrix - matrix with data to be padded

        pad_length - int, number of indices to pad

    OUTPUTS

        padded_data - np.ndarry, 
        
    '''

    data_copy = np.copy(wavefront_matrix.data)
    data_copy[wavefront_matrix.mask] = 0.0

    ## Pad the data by looping over the mask rows and finding where
    ## the edge is. The value of the wavefront at the edge is 
    ## propagated outward along the axes, with some attempt to make
    ## an average of the X and Y extrapolations
    for i in range(wavefront_matrix.shape[0]):
        for j in range(wavefront_matrix.shape[1]):
            if not wavefront_matrix.mask[i,j]:
                for k in range(pad_length):
                    try:
                        data_copy[i,j-k] = wavefront_matrix[i,j]
                    except IndexError:
                        break
                break
        for j in range(wavefront_matrix.shape[1])[::-1]:
            if not wavefront_matrix.mask[i,j]:
                for k in range(pad_length):
                    try:
                        data_copy[i,j+k] = wavefront_matrix[i,j]
                    except IndexError:
                        break
                break

    for j in range(wavefront_matrix.shape[1]):
        for i in range(wavefront_matrix.shape[0]):
            if not wavefront_matrix.mask[i,j]:
                for k in range(pad_length):
                    try:
                        if data_copy[i-k,j]:
                            data_copy[i-k,j] = np.mean(\
                                    [wavefront_matrix[i,j], data_copy[i-k,j]])
                        else:
                            data_copy[i-k,j] = wavefront_matrix[i,j]
                    except IndexError:
                        break
                break
        for i in range(wavefront_matrix.shape[0])[::-1]:
            if not wavefront_matrix.mask[i,j]:
                for k in range(pad_length):
                    try:
                        if data_copy[i+k,j]:
                            data_copy[i+k,j] = np.mean(\
                                        [wavefront_matrix[i,j], data_copy[i+k,j]])
                        else:
                            data_copy[i+k,j] = wavefront_matrix[i,j]
                    except IndexError:
                        break
                break

    return data_copy



def smooth_wavefront(wavefront_matrix, sigma=1.0, pad_length=5):

    '''
    Routine to resample a wavefront surface over a new set of points.

    INPUTS

        wavefront_matrix - ndarray, X-coordinates at which the 
            data is sampled

        sigma - float, kernel for the gaussian blurring

    OUTPUTS
        
    '''

    ### Make sure we're padding more than we're blurring, otherwise
    ### the interpolator may do weird things at the end of the 
    ### wavefront, since the data array has those values as zeros
    ### in the absence of padding
    if pad_length < sigma:
        pad_length = int(np.ceil(sigma)) + 1

    data_copy = _pad_wavefront(wavefront_matrix, pad_length=pad_length)

    ### Blur the padded data and then re-apply the mask
    blur_arr = ndimage.gaussian_filter(data_copy, 1)

    wavefront_arr_n = blur_arr * np.logical_not(wavefront_matrix.mask)
    wavefront_matrix_n = np.ma.masked_where(\
                            wavefront_matrix.mask, wavefront_arr_n)

    return wavefront_matrix_n




def resample_wavefront(xcoord, ycoord, wavefront_matrix, \
                       new_xcoord, new_ycoord, sigma=1.0, \
                       smoothing=0.0, pad_length=5):
    '''
    Routine to resample a wavefront surface over a new set of points.

    INPUTS

        xcoord - ndarray, X-coordinates at which the data is sampled

        ycoord - ndarray, Y-coordinates at which the data is sampled

        wavefront_matrix - np.matrix, values of the wavefront phase
            in a 2D matrix, likely with a mask

        x0 - float, the centroid position on the X-axis

        y0 - float, the centroid position on the Y-axis

        new_xcoord - ndarray, new X-coordinates to sample

        new_ycoord - ndarray, new Y-coordinates to sample

        sigma - float, size of the gaussian blurring kernel to apply
            before interpolating

        smoothing - float, the smoothing factor applied to the 
            bivariate spline interpolator

        pad_length - int, number of indices to pad to help the 
            interpolator be well-behaved at the edges

    OUTPUTS
        
    '''

    data_copy = _pad_wavefront(wavefront_matrix, pad_length=pad_length)

    ### Blur the padded data and then re-apply the mask
    blur_arr = ndimage.gaussian_filter(data_copy, sigma)

    ### Interpolate the wavefront mask so we can resample this as well
    mask_interp = interpolate.RectBivariateSpline(\
                        xcoord, ycoord, wavefront_matrix.mask, \
                        kx=1, ky=1, s=0.0)

    ### Interpolate the wavefront itself
    interp_func = interpolate.RectBivariateSpline(\
                        xcoord, ycoord, blur_arr, \
                        kx=3, ky=3, s=smoothing)

    ### Build the new mask by evaluating the mask_interp function
    ### and then converting the floats to booleans
    mask_n_float = mask_interp(new_xcoord, new_ycoord)
    mask_n = np.zeros(mask_n_float.shape, dtype=bool)
    mask_n[mask_n_float > 0.9] = True

    ### Define a matrix to be consistent in object type etc. Make the 
    ### values covered by the mask equal to np.nan since the wavefront
    ### can actually cross through 0
    wavefront_arr_n = interp_func(new_xcoord, new_ycoord) * np.logical_not(mask_n)
    wavefront_matrix_n = np.ma.masked_where(mask_n, wavefront_arr_n)
    wavefront_matrix_n[mask_n] = np.nan

    return wavefront_matrix_n




