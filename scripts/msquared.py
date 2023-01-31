import os
import wavefronts

import numpy as np
import matplotlib.pyplot as plt

from iminuit import Minuit


plt.rcParams.update({'font.size': 16})


# meas_base = '20220822_input_beam_collimation'
# meas_list = \
#     [
#      'telescope_out_fullsize_z57cm', \
#      'telescope_out_fullsize_z71cm_1mirror', \
#      'telescope_out_fullsize_z158_5cm_1mirror', \
#      'telescope_out_fullsize_z249_5cm_1mirror', \
#      'telescope_out_fullsize_z358_5cm_2mirror', \
#      'telescope_out_fullsize_z478_5cm_3mirror'
#     ]
# axial_positions = np.array([57.0, 71.0, 158.5, 249.5, 358.5, 478.5])*1e-2


meas_base = '20230124_input_collimation'
meas_list = \
    [
     'trapping_beam_telescope_output_nopinhole_later_33cm', \
     'trapping_beam_telescope_output_nopinhole_later_68cm', \
     'trapping_beam_telescope_output_nopinhole_later_111cm', \
     'trapping_beam_telescope_output_nopinhole_later_186cm', \
    ]
axial_positions = np.array([33.0, 68.0, 11.0, 186.0])*1e-2

# shade_with_power = False
# colorbar = True

fix_waist_loc = False
print_fit = True

#############################################################
##############  FIT TO FIND THE BEAM WAISTS  ################
#############################################################


# xwaists = [[], []]
# ywaists = [[], []]

xwaists = []
ywaists = []

angles = []

for meas in meas_list:

    meas_name = os.path.join(meas_base, meas)

    xcoord, ycoord, wavefront_matrix, power_matrix = \
        wavefronts.load_wfs_data(meas_name=meas_name)

    smooth_power_matrix = wavefronts.smooth_wavefront(\
                                power_matrix, sigma=1)

    # popt, pcov = wavefronts.fit_2d_gaussian(\
    #                 xcoord, ycoord, power_matrix, \
    #                 plot=True, \
    #                 print_fit=False, \
    #                 fix_theta=True, \
    #                 theta_init=0.0, \
    #                 save_fig=True, \
    #                 fig_name=meas+'_power_fit.svg')

    wavefronts.plot_2d_power(xcoord, ycoord, power_matrix, \
                         # xlim=(-2.5, 2.5), ylim=(-2.5, 2.5),
                         show=True, colorbar=True, \
                        )

    # wavefronts.plot_2d_power(xcoord, ycoord, smooth_power_matrix, \
    #                      # xlim=(-2.5, 2.5), ylim=(-2.5, 2.5),
    #                      show=True, colorbar=True, \
    #                     )

    xwaists.append(2.0*popt[3])
    ywaists.append(2.0*popt[4])

    angles.append(popt[5])

print(angles)
xwaists = np.array(xwaists);  ywaists = np.array(ywaists)



#############################################################
#############  CONSIDER DIVERGENCE BEHAVIOR  ################
#############################################################

def gaussian_waist_evolution(z, z0, w0, Msq, wavelength=1064.0e-9):
    return np.sqrt( w0**2 + Msq**2 * (wavelength / (np.pi * w0))**2 * (z - z0)**2 )

### Construct a least-squares cost function
npts = 2.0*len(xwaists)
def cost(z0, w0, Msq):
    func_val = gaussian_waist_evolution(axial_positions, z0, w0, Msq)
    resid = (xwaists - func_val)**2 + (ywaists - func_val)**2
    return (1.0 / (npts - 1.0)) * np.sum(resid)

### Assemble the Minuit object with cost function and
### initial parameter value guesses
m = Minuit(cost, 
           z0 = axial_positions[np.argmin(np.concatenate((xwaists, ywaists)))], \
           w0 = np.min(np.concatenate((xwaists, ywaists))), \
           Msq = 1.1, \
           )

### Set some limits to make sure the fitting is well-behaved
m.limits['z0'] = (-1.0*np.inf, np.inf)
m.limits['w0'] = (0.0, np.max(np.concatenate((xwaists, ywaists))))
m.limits['Msq'] = (1.0, np.inf)

### Fix any values that the user wants fixed
if fix_waist_loc:
    m.fixed['z0'] = True

### Default values for number of standard deviations to report
### in the parameter error value, as well as the verbosity
### of the result printing
m.errordef = 1
m.print_level = 0

### Minimize the cost function
result = m.migrad(ncall=100000)
if print_fit:
    print(result)




fig, ax = plt.subplots(1,1)
ax.plot(axial_positions, xwaists, '-o')
ax.plot(axial_positions, ywaists, '-o')
ax.plot(axial_positions, \
        gaussian_waist_evolution(axial_positions, \
                                 m.values['z0'], \
                                 m.values['w0'], \
                                 m.values['Msq'], \
                                 ), \
        lw=2, ls='--', color='r'
        )
ax.set_ylim(0.0, 1.2*ax.get_ylim()[1])
ax.set_xlabel('Axial Position [m]')
ax.set_ylabel('Waist [mm]')

fig.tight_layout()

plt.show()

