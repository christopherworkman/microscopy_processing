from tifffile import imwrite
import numpy as np
from nd2reader import ND2Reader
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import sys
import psutil
import os

# Function to process a single FOV and return the result
def process_fov(v, in_filename, all_fovs_array):
    with ND2Reader(in_filename) as images:
        # Set the iterating and bundling axes for ND2Reader within each thread
        images.iter_axes = 'v'
        images.bundle_axes = 'czyx'
        
        # Retrieve the FOV by index, process it, and perform max projection
        fov = images[v].astype(np.uint16)
        print(f"Processing FOV {v} with shape: {fov.shape}")

        all_fovs_array[v] = np.max(fov, axis=1)  # Max project and directly store in the shared array

# Main function with input and output filenames as arguments
def main(in_filename, out_filename, num_fovs=None):
    # Start total duration timer
    start_total = time.time()

    # Open ND2Reader to access input images
    with ND2Reader(in_filename) as images:
        # Start timer for reading dimensions
        start_reading = time.time()
        
        # Get image dimensions
        channels = images.sizes['c']
        width = images.sizes['x']
        height = images.sizes['y']
        fovs = images.sizes['v']
        zs = images.sizes['z']
        
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

        # Memory per thread (adjust based on testing)
        overhead = 5
        data_type_size = 2  # bytes for uint16
        memory_per_thread = height * width * channels * zs * data_type_size * overhead # in bytes
        print('Estimated memory per thread ', memory_per_thread / 1e9, ' GB')
        available_memory = psutil.virtual_memory().available
        print('Available Memory', available_memory / 1e9, 'GB')
        optimal_workers = min(available_memory // memory_per_thread, os.cpu_count())
        print('Optimal Workers', optimal_workers)
        max_workers = max(1 , optimal_workers)  # Ensure at least 1 worker

        # Parallel processing of each FOV with ordered writing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks to process each FOV and store futures with their respective indices
            futures = {executor.submit(process_fov, v, in_filename, all_fovs_array): v for v in range(fovs)}  

            # Collect results as each FOV completes
            for future in as_completed(futures):
                # v, fov_max = future.result()
                future.result()
                # all_fovs_array[v] = fov_max  # Store result by FOV index for ordered writing

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
