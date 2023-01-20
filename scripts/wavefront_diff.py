import os
import wavefronts
import numpy as np


meas_base = '20220822_input_beam_collimation'
# meas_name1 = os.path.join(meas_base, 'telescope_out_fullsize_z57cm')
meas_name1 = os.path.join(meas_base, 'telescope_out_fullsize_z71cm_1mirror')
meas_name2 = os.path.join(meas_base, 'telescope_out_fullsize_z158_5cm_1mirror')

shade_with_power = False
colorbar = True

plot_fits = False

#############################################################


xcoord1, ycoord1, wavefront_matrix1, power_matrix1 = \
    wavefronts.load_wfs_data(meas_name=meas_name1)

xcoord2, ycoord2, wavefront_matrix2, power_matrix2 = \
    wavefronts.load_wfs_data(meas_name=meas_name2)

# wavefront_matrix2 = wavefront_matrix2[:,::-1]
# power_matrix2 = power_matrix2[:,::-1]


popt1, func = wavefronts.fit_2d_gaussian(xcoord1, ycoord1, \
                                         power_matrix1, plot=plot_fits)
popt2, _ = wavefronts.fit_2d_gaussian(xcoord2, ycoord2, \
                                      power_matrix2, plot=plot_fits)

wavefront_matrix1_n = \
    wavefronts.resample_wavefront(xcoord1-popt1[1], ycoord1-popt1[2], \
                                  wavefront_matrix1, xcoord1, ycoord1, \
                                  smoothing=1e-5)
wavefront_matrix2_n = \
    wavefronts.resample_wavefront(xcoord2-popt2[1], ycoord2-popt2[2], \
                                  wavefront_matrix2, xcoord1, ycoord1, \
                                  smoothing=1e-5)



fig1, ax1 = wavefronts.plot_3d_wavefront(xcoord1, ycoord1, \
                                       wavefront_matrix1_n, #power_matrix, \
                                       shade_with_power=shade_with_power, \
                                       show=False, cmap='plasma')

fig2, ax2 = wavefronts.plot_3d_wavefront(xcoord1, ycoord1, \
                                       wavefront_matrix2_n, #power_matrix, \
                                       shade_with_power=shade_with_power, \
                                       show=False, cmap='plasma')




# wavefront_diff = np.ma.array(wavefront_matrix1_n[:,::-1] - wavefront_matrix2_n, \
#                     mask=(wavefront_matrix1_n.mask[:,::-1] * wavefront_matrix2_n.mask))

wavefront_diff = np.ma.array(wavefront_matrix1_n - wavefront_matrix2_n, \
                    mask=(wavefront_matrix1_n.mask * wavefront_matrix2_n.mask))


fig3, ax3 = wavefronts.plot_3d_wavefront(xcoord1, ycoord1, \
                                       wavefront_diff, #power_matrix, \
                                       shade_with_power=shade_with_power, \
                                       show=True, cmap='plasma')



