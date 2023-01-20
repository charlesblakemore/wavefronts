import os
import wavefronts

import numpy as np
import matplotlib.pyplot as plt


meas_base = '20220822_input_beam_collimation'

meas_list = \
    [
     'telescope_out_fullsize_z57cm', \
     'telescope_out_fullsize_z71cm_1mirror', \
     'telescope_out_fullsize_z158_5cm_1mirror', \
     'telescope_out_fullsize_z249_5cm_1mirror', \
     'telescope_out_fullsize_z358_5cm_2mirror', \
     'telescope_out_fullsize_z478_5cm_3mirror'
    ]

shade_with_power = False
colorbar = True


#############################################################

axial_positions = np.array([57.0, 71.0, 158.5, 249.5, 358.5, 478.5])*1e-2

xwaists = []
ywaists = []

angles = []

for meas in meas_list:

    meas_name = os.path.join(meas_base, meas)

    xcoord, ycoord, wavefront_matrix, power_matrix = \
        wavefronts.load_wfs_data(meas_name=meas_name)

    popt, _ = wavefronts.fit_2d_gaussian(\
                    xcoord, ycoord, power_matrix, plot=False, \
                    fix_theta=False, theta_init=0.0)

    xwaists.append(2.0*popt[3])
    ywaists.append(2.0*popt[4])

    angles.append(popt[5])

print(angles)

fig, ax = plt.subplots(1,1)
ax.plot(axial_positions, xwaists, '-o')
ax.plot(axial_positions, ywaists, '-o')
ax.set_ylim(0.0, ax.get_ylim()[1])
ax.set_xlabel('Axial Position [m]')
ax.set_ylabel('Waist [mm]')

fig.tight_layout()

plt.show()

