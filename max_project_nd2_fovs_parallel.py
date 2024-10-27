from tifffile import imwrite
import numpy as np
import pims
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import sys
import psutil
import os

# Function to max project a single FOV
def process_fov(fov, m, all_fovs_array):
    all_fovs_array[m] = np.max(fov, axis=1)  # Max projection across the Z-axis

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

        # Calculate memory-per-thread, set max_workers based on memory and CPU count
        overhead = 1.5
        data_type_size = 2  # bytes for uint16
        memory_per_thread = height * width * channels * zs * data_type_size * overhead  # in bytes
        print('Estimated memory per thread ', memory_per_thread / 1e9, ' GB')
        available_memory = psutil.virtual_memory().available
        print('Available Memory', available_memory / 1e9, 'GB')
        optimal_workers = min(available_memory // memory_per_thread, os.cpu_count())
        print('Optimal Workers', optimal_workers)
        max_workers = max(1, optimal_workers)  # Ensure at least 1 worker

        # Start timer for processing FOVs
        start_processing = time.time()

        # Loop through FOVs in batches
        batch_size = max_workers
        for start_index in range(0, fovs, batch_size):
            start_batch = time.time()
            end_index = min(start_index + batch_size, fovs)
            batch = [images[m] for m in range(start_index, end_index)]

            duration_load = time.time() - start_batch
            print(f"Time to process FOVs {start_index} to {end_index}: {duration_load:.2f} seconds")

            # Process the current batch in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_fov, batch[i], start_index + i, all_fovs_array) for i in range(len(batch))}

                # Wait for all futures in the batch to complete
                for future in as_completed(futures):
                    future.result()
            duration_batch = time.time() - start_batch
            print(f"Time to process FOVs: {duration_batch:.2f} seconds")
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
    in_filename = sys.argv[1] if len(sys.argv) > 1 else '/mnt2/nepenthes/Chris/acx_tangential_starmap/yc19lbp2/round9.nd2'
    out_filename = sys.argv[2] if len(sys.argv) > 2 else '/home/kebschulllab/max_proj/max_round9.ome.tif'
    num_fovs = int(sys.argv[3]) if len(sys.argv) > 3 else None  # Optional number of FOVs to process
    main(in_filename, out_filename, num_fovs)
