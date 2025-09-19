from tifffile import imwrite
import numpy as np
import time
import sys
import nd2  # pure-Python reader; no Java

def max_project_nd2(in_filename, num_fovs=None):
    """
    Returns a numpy array shaped (FOV, C, Y, X), uint16.
    FOVs are taken from ND2 axis 'P' (positions) if present, else 'M', else 'T'.
    Always max-projects over Z and T.
    """
    t0 = time.time()
    with nd2.ND2File(in_filename) as f:
        # Ordered axis labels (e.g., ('P','C','Z','Y','X') or ('T','C','Z','Y','X'), etc.)
        axes_order = tuple(f.sizes)           # ordered keys
        sizes      = dict(f.sizes)            # axis -> length

        # Pick the FOV axis
        if sizes.get('P', 0) > 1:
            fov_axis = 'P'
        elif sizes.get('M', 0) > 1:
            fov_axis = 'M'
        elif sizes.get('T', 0) > 1:
            fov_axis = 'T'  # fallback: treat time as FOVs
        else:
            fov_axis = None

        n_fovs = sizes.get(fov_axis, 1) if fov_axis else 1
        if num_fovs:
            n_fovs = min(n_fovs, int(num_fovs))

        axis_index = {ax: i for i, ax in enumerate(axes_order)}
        darr = f.to_dask()  # dask array in the same axis order as axes_order

        out_list = []
        start_batch = time.time()
        for i in range(n_fovs):
            # slice FOV axis (if any)
            indexer = [slice(None)] * len(axes_order)
            if fov_axis:
                indexer[axis_index[fov_axis]] = i
            sub = darr[tuple(indexer)]  # e.g. (C,Z,Y,X) or (T,C,Z,Y,X), etc.

            # Track current axes after possible FOV slicing
            cur_axes = list(axes_order)
            if fov_axis:
                cur_axes.pop(axis_index[fov_axis])

            # Reduce over T, then Z (if present)
            if 'T' in cur_axes:
                t_idx = cur_axes.index('T')
                sub   = sub.max(axis=t_idx)
                cur_axes.pop(t_idx)
            if 'Z' in cur_axes:
                z_idx = cur_axes.index('Z')
                sub   = sub.max(axis=z_idx)
                cur_axes.pop(z_idx)

            # Ensure we have (C,Y,X)
            if 'C' in cur_axes:
                c_idx = cur_axes.index('C')
                if c_idx != 0:
                    sub = sub.moveaxis(c_idx, 0)
                    cur_axes.insert(0, cur_axes.pop(c_idx))
            else:
                sub = sub[None, ...]
                cur_axes.insert(0, 'C')

            # Make sure Y and X exist and are last two in order (C,Y,X)
            if 'Y' not in cur_axes and 'X' in cur_axes:
                sub = sub[:, None, ...]
                cur_axes.insert(1, 'Y')
            if 'X' not in cur_axes and 'Y' in cur_axes:
                sub = sub[..., None]
                cur_axes.append('X')

            cur_axes_str = ''.join(cur_axes)
            order = [cur_axes_str.index('C'), cur_axes_str.index('Y'), cur_axes_str.index('X')]
            sub = sub.transpose(order)

            arr = sub.compute() if hasattr(sub, "compute") else np.asarray(sub)
            if arr.dtype != np.uint16:
                arr = arr.astype(np.uint16, copy=False)

            out_list.append(arr)

            if (i + 1) % 10 == 0 or (i + 1) == n_fovs:
                print(f"[ND2] Processed {i+1}/{n_fovs} FOVs in {time.time()-start_batch:.2f}s")
                start_batch = time.time()

        stack = np.stack(out_list, axis=0)  # (FOV, C, Y, X)
        print(f"[ND2] axes={axes_order} sizes={sizes} -> output {stack.shape} in {time.time()-t0:.2f}s")
        return stack

def main(in_filename, out_filename, num_fovs=None):
    start_total = time.time()
    stack = max_project_nd2(in_filename, num_fovs=num_fovs)
    # Save using your original convention: FOV as 'T'
    imwrite(out_filename, stack, ome=True, photometric='minisblack',
            metadata={'axes': 'TCYX'})
    print(f"Saved {out_filename}")
    print(f"Total time: {time.time() - start_total:.2f}s")

if __name__ == "__main__":
    in_filename  = sys.argv[1] if len(sys.argv) > 1 else '/path/image.nd2'
    out_filename = sys.argv[2] if len(sys.argv) > 2 else '/path/max_image.ome.tif'
    num_fovs     = int(sys.argv[3]) if len(sys.argv) > 3 else None
    main(in_filename, out_filename, num_fovs)
