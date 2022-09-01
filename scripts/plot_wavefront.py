import os
import wavefronts


### Code is shipped with an example wavefront. Assuming the user puts
### data in the data directory (for automatic path finding etc), then they
### will want to comment this line out, as it changes the default
### parent directory that the code looks in
wavefronts.data_dir = os.path.abspath('../example_data')


meas_name = 'example'

shade_with_power = True
colorbar = True


#############################################################


xcoord, ycoord, wavefront_matrix, power_matrix = \
    wavefronts.load_wfs_data(meas_name=meas_name)


fig, ax = wavefronts.plot_3d_wavefront(xcoord, ycoord, \
                                       wavefront_matrix, power_matrix, \
                                       shade_with_power=shade_with_power, \
                                       show=True, cmap='plasma')



