from tifffile import imwrite
import numpy as np
import time, sys, os, tempfile
import nd2  # pure-Python reader; no Java

def stream_max_project_nd2(in_filename, out_filename, num_fovs=None):
    """
    Stream max-projected FOVs to disk without holding the whole stack in RAM.
    Output OME-TIFF has axes 'TCYX' (T = FOV).
    """
    t0 = time.time()
    with nd2.ND2File(in_filename) as f:
        axes_order = tuple(f.sizes)   # e.g. ('P','C','Z','Y','X') or ('T','C','Z','Y','X')
        sizes      = dict(f.sizes)

        # Choose FOV axis: P > M > T (fallback)
        if sizes.get('P', 0) > 1:
            fov_axis = 'P'
        elif sizes.get('M', 0) > 1:
            fov_axis = 'M'
        elif sizes.get('T', 0) > 1:
            fov_axis = 'T'
        else:
            fov_axis = None

        total_fovs = sizes.get(fov_axis, 1) if fov_axis else 1
        if num_fovs:
            total_fovs = min(total_fovs, int(num_fovs))

        # Basic geometry
        C = sizes.get('C', 1)
        Y = sizes.get('Y', None)
        X = sizes.get('X', None)
        if Y is None or X is None:
            # Peek one tiny slice to infer Y/X (rarely needed)
            arr0 = np.asarray(f.to_dask()[tuple(0 if (ax in ('P','M','T','Z','C') and sizes.get(ax,1)>1) else slice(None)
                                               for ax in axes_order)])
            if Y is None: Y = arr0.shape[-2]
            if X is None: X = arr0.shape[-1]

        # Create dask array once (no custom chunking; single-threaded compute)
        darr = f.to_dask()
        axis_index = {ax: i for i, ax in enumerate(axes_order)}

        # Disk-backed buffer (T, C, Y, X) where T == total_fovs
        tmpdir = tempfile.gettempdir()
        memmap_path = os.path.join(tmpdir, os.path.basename(out_filename) + ".mmap")
        mmap = np.memmap(memmap_path, mode='w+', dtype=np.uint16, shape=(total_fovs, C, Y, X))

        start_batch = time.time()
        for i in range(total_fovs):
            # Slice this FOV
            indexer = [slice(None)] * len(axes_order)
            if fov_axis:
                indexer[axis_index[fov_axis]] = i
            sub = darr[tuple(indexer)]   # e.g. (C,Z,Y,X) or (T,C,Z,Y,X), etc.

            # Track axes for this subarray
            cur_axes = list(axes_order)
            if fov_axis:
                cur_axes.pop(axis_index[fov_axis])

            # Max over T and Z (if present)
            if 'T' in cur_axes:
                t_idx = cur_axes.index('T')
                sub   = sub.max(axis=t_idx)
                cur_axes.pop(t_idx)
            if 'Z' in cur_axes:
                z_idx = cur_axes.index('Z')
                sub   = sub.max(axis=z_idx)
                cur_axes.pop(z_idx)

            # Ensure (C,Y,X)
            if 'C' in cur_axes:
                c_idx = cur_axes.index('C')
                if c_idx != 0:
                    sub = sub.moveaxis(c_idx, 0)
                    cur_axes.insert(0, cur_axes.pop(c_idx))
            else:
                sub = sub[None, ...]
                cur_axes.insert(0, 'C')

            if 'Y' not in cur_axes and 'X' in cur_axes:
                sub = sub[:, None, ...]
                cur_axes.insert(1, 'Y')
            if 'X' not in cur_axes and 'Y' in cur_axes:
                sub = sub[..., None]
                cur_axes.append('X')

            cur_axes_str = ''.join(cur_axes)
            order = [cur_axes_str.index('C'), cur_axes_str.index('Y'), cur_axes_str.index('X')]
            sub = sub.transpose(order)

            # Compute (single-threaded by default) and cast
            arr = sub.compute() if hasattr(sub, "compute") else np.asarray(sub)
            if arr.dtype != np.uint16:
                arr = arr.astype(np.uint16, copy=False)

            # Guard shape & write to memmap
            if arr.shape != (C, Y, X):
                arr = arr[..., :Y, :X]
                if arr.shape[1] < Y or arr.shape[2] < X:
                    pad_y = Y - arr.shape[1]
                    pad_x = X - arr.shape[2]
                    arr = np.pad(arr, ((0,0),(0,pad_y),(0,pad_x)), mode='edge')

            mmap[i] = arr

            if (i + 1) % 10 == 0 or (i + 1) == total_fovs:
                print(f"[ND2] Processed {i+1}/{total_fovs} FOVs in {time.time()-start_batch:.2f}s")
                start_batch = time.time()

        # Flush and write final OME-TIFF from memmap (no big RAM usage)
        mmap.flush()
        imwrite(
            out_filename, mmap, ome=True, photometric='minisblack',
            metadata={'axes': 'TCYX'}, compression='LZW', bigtiff=True
        )
        del mmap
        try:
            os.remove(memmap_path)
        except OSError:
            pass

    print(f"Saved {out_filename}")
    print(f"Total time: {time.time() - t0:.2f}s")

def main(in_filename, out_filename, num_fovs=None):
    stream_max_project_nd2(in_filename, out_filename, num_fovs=num_fovs)

if __name__ == "__main__":
    in_filename  = sys.argv[1] if len(sys.argv) > 1 else '/path/image.nd2'
    out_filename = sys.argv[2] if len(sys.argv) > 2 else '/path/max_image.ome.tif'
    num_fovs     = int(sys.argv[3]) if len(sys.argv) > 3 else None
    main(in_filename, out_filename, num_fovs)
