import os
import wavefronts

import numpy as np

import matplotlib.pyplot as plt


### Code is shipped with an example wavefront. Assuming the user puts
### data in the data directory (for automatic path finding etc), then they
### will want to comment this line out, as it changes the default
### parent directory that the code looks in
# wavefronts.data_dir = os.path.abspath('../example_data')


# meas_name = 'example'


# meas_name = '20221026_7_5um_gbead/no_bead/reference_beam_no_bead'
# meas_name = '20221026_7_5um_gbead/no_bead/trapping_beam_no_bead'

# meas_name = '20221026_7_5um_gbead/lowering/z0_trapping_beam'
# meas_name = '20221026_7_5um_gbead/lowering/z5000_trapping_beam'
# meas_name = '20221026_7_5um_gbead/lowering/z10000_trapping_beam'
# meas_name = '20221026_7_5um_gbead/lowering/z15000_trapping_beam'

# meas_name = '20221026_7_5um_gbead/xy_test/xy_test_2'

# meas_name = '20220822_input_beam_collimation/telescope_out_fullsize_z57cm'
# meas_name = '20220822_input_beam_collimation/telescope_out_fullsize_z478_5cm_3mirror'

# meas_name = '20221130_output_beam_collimation/initial_state/initial'
# meas_name = '20221130_output_beam_collimation/initial_state/initial_no_tele'
# meas_name = '20221130_output_beam_collimation/recollimation_lens/new_lens'
# meas_name = '20221130_output_beam_collimation/recollimation_lens/new_lens_shim'
# meas_name = '20221130_output_beam_collimation/recollimation_lens/new_lens_shim_vacuum'

# meas_name = '20230119_beam_alignment/input_beam_initial_48cm'
# meas_name = '20230123_objective_collimation/trapping_beam_objective_output_44cm'
# meas_name = '20230123_objective_collimation/trapping_beam_objective_output_recoll_44cm'
# meas_name = '20230123_objective_collimation/trapping_beam_objective_output_recoll_nextday_44cm'
# meas_name = '20230123_objective_collimation/trapping_beam_objective_output_recoll2_101cm'

# meas_name = '20230124_input_collimation/trapping_beam_telescope_output_nopinhole_later_33cm'

# meas_name = '20230125_output_collimation/trapping_beam_telescope_output_nopinhole_filter_186cm'
meas_name = '20230125_output_collimation/output_beam_fullsize_89cm'

shade_with_power = True
colorbar = True

invert_curvature = False

# max_power = 5000
max_power = 15000

max_radius = None
crop_center = None

# max_radius = 3.5
# crop_center = (0.0, 0.0)

# max_radius = 1.0
# crop_center = (-0.6, 0.5)


#############################################################


xcoord, ycoord, wavefront_matrix, power_matrix = \
    wavefronts.load_wfs_data(meas_name=meas_name, max_power=max_power, \
                             max_radius=max_radius, crop_center=crop_center)

smooth_wavefront_matrix = wavefronts.smooth_wavefront(\
                                wavefront_matrix, sigma=1)
smooth_power_matrix = wavefronts.smooth_wavefront(\
                                power_matrix, sigma=1)

extent = (np.min(xcoord), np.max(xcoord), np.min(ycoord), np.max(ycoord))

# wavefronts.fit_2d_gaussian(xcoord, ycoord, power_matrix, \
#                            print_fit=True, plot=True, fix_theta=True)

wavefronts.fit_2d_gaussian(xcoord, ycoord, smooth_power_matrix, \
                           print_fit=True, plot=False, fix_theta=True)

wavefronts.plot_2d_power(xcoord, ycoord, smooth_power_matrix, colorbar=True, \
                         # xlim=(-2.5, 2.5), ylim=(-2.5, 2.5),
                         show=False, cmap='inferno',\
                        )

# fig, ax = wavefronts.plot_3d_wavefront(xcoord, ycoord, \
#                                        wavefront_matrix, \
#                                        power_matrix, \
#                                        shade_with_power=shade_with_power, \
#                                        invert=invert_curvature, \
#                                        show=True, cmap='plasma')

fig, ax = wavefronts.plot_3d_wavefront(xcoord, ycoord, \
                                       smooth_wavefront_matrix, \
                                       smooth_power_matrix, \
                                       shade_with_power=shade_with_power, \
                                       invert=invert_curvature, \
                                       show=True, cmap='inferno')



