from tifffile import imwrite
import numpy as np
import pims
import time
import sys

# Main function with input and output filenames as arguments
def main(in_filename, out_filename, num_fovs=None):
    # Start total duration timer
    start_total = time.time()

    # Open pims with ND2 file to get image dimensions
    with pims.open(in_filename) as images:
        # Start timer for reading dimensions
        start_reading = time.time()

        # Get image dimensions
        channels = images.sizes['c']
        width = images.sizes['x']
        height = images.sizes['y']
        fovs = images.sizes['m']
        zs = images.sizes['z']

        images.iter_axes = 'm'
        images.bundle_axes = 'czyx'

        # Limit FOVs if specified
        if num_fovs:
            fovs = min(num_fovs, fovs)

        # Stop timer for reading dimensions
        duration_reading = time.time() - start_reading
        print(f"Time to read dimensions: {duration_reading:.2f} seconds")

        # Preallocate the results array to hold the selected FOVs: shape (fovs, channels, height, width)
        all_fovs_array = np.zeros((fovs, channels, height, width), dtype=np.uint16)

        # Start timer for processing FOVs
        start_processing = time.time()

        # Process each FOV individually and store results directly in all_fovs_array
        count = 0
        start_batch = time.time()
        for m in range(fovs):
            fov_data = images[m]  # Load single FOV
            all_fovs_array[m] = np.max(fov_data, axis=1)  # Max project across Z and store directly in the output array
            if count > 8:
                duration_batch = time.time() - start_batch
                print(f"Time per 10 FOVs to FOV {m}: {duration_batch:.2f} seconds")
                start_batch = time.time()
                count = 0
            else:
                count += 1

        # Stop timer for processing FOVs
        duration_processing = time.time() - start_processing
        print(f"Time to process FOVs: {duration_processing:.2f} seconds")

    # Start timer for writing TIFF
    start_writing = time.time()

    # Open TIFF writer with OME-TIFF format
    imwrite(out_filename, all_fovs_array, ome=True, photometric='minisblack', metadata={'axes': 'TCYX'})

    # Stop timer for writing TIFF
    duration_writing = time.time() - start_writing
    print(f"Time to write TIFF: {duration_writing:.2f} seconds")

    # Stop total duration timer
    total_duration = time.time() - start_total
    print(f"Total time: {total_duration:.2f} seconds")

# Run main function with command-line arguments or specify filenames here
if __name__ == "__main__":
    in_filename = sys.argv[1] if len(sys.argv) > 1 else '/path/image.nd2'
    out_filename = sys.argv[2] if len(sys.argv) > 2 else '/path/max_image.ome.tif'
    num_fovs = int(sys.argv[3]) if len(sys.argv) > 3 else None  # Optional number of FOVs to process
    main(in_filename, out_filename, num_fovs)
