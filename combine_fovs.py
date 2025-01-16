import tifffile
import numpy as np
import pims
import time
import sys

directory = '/mnt/cephalotus_4/chris/store/'

in_filename_list = ['max_round3_gtac.ome.tif', 'max_round3_gtac_238toend.ome.tif']
fov_ranges = [np.arange(0, 238), np.arange(238, 2515)]
# in_filename_list = ['max_round6_gtac.ome.tif']

start_processing = time.time()
full_round_image = np.zeros((2515, 4, 2304, 2304), dtype=np.uint16)

for fov_range, in_filename in zip(fov_ranges, in_filename_list):
    frames = tifffile.imread(directory + in_filename)
    print(frames.shape)
    
    full_round_image[fov_range,:,:,:] = frames[fov_range - fov_range[0],:,:,:]

tifffile.imwrite(directory + 'round3.ome.tif', full_round_image, ome=True, photometric='minisblack', metadata={'axes': 'TCYX'})

total_duration = time.time() - start_processing
print(f"Total time: {total_duration:.2f} seconds")