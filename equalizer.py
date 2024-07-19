import zarr
import os
from typing import Union
import json
import numcodecs
try:
    import cupy as xp
    # Check if a GPU is available
    if xp.cuda.runtime.getDeviceCount() == 0:
        raise ImportError("No GPU found, falling back to NumPy.")
except (ImportError, xp.cuda.runtime.CUDARuntimeError):
    import numpy as xp
from sklearn.mixture import GaussianMixture

def equalize(zarrpath: Union[str, os.PathLike], divisor_range: int = 12, min_range: int = 100) -> None:
    assert divisor_range > 3, "Divisor range too small"
    assert min_range > 50, "Min range too small"

    with zarr.open(zarrpath, mode="r") as z:
        assert all(dim > min_range for dim in z[0].shape), "Min range too big"

        box_min = [dim // 2 - dim // divisor_range for dim in z[0].shape]
        box_max = [dim // 2 + dim // divisor_range for dim in z[0].shape]

        for i, (min_val, max_val) in enumerate(zip(box_min, box_max)):
            if (max_val - min_val) < min_range:
                box_min[i] = z[0].shape[i] // 2 - min_range // 2
                box_max[i] = z[0].shape[i] // 2 + min_range // 2

        chunk = xp.asarray(z[0][box_min[0]:box_max[0], box_min[1]:box_max[1], box_min[2]:box_max[2]]).astype(xp.float32)
        chunk = chunk.get() if hasattr(chunk, 'get') else chunk

        import numpy as np

        gmm = GaussianMixture(n_components=2, verbose=1)
        gmm.fit(chunk.reshape(-1, 1))

        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.reshape(2, 1, 1).flatten())

        # Zip and sort means and standard deviations with respect to means
        zipped = sorted(zip(means, stds), key=lambda x: x[0])
        sorted_means, sorted_stds = zip(*zipped)

        # Compute the required values
        value1 = sorted_means[0] # - sorted_stds[0]/2. removing this for air, probably not needed
        value2 = sorted_means[1] + sorted_stds[1]/2. # taking one half standard deviation more for papyrus

        dir_path = os.path.dirname(zarrpath)
        base_name = os.path.basename(zarrpath)
        new_path = os.path.join(dir_path, base_name[:-5] + '_equalized.zarr')

        def transform_function(volume, min_value=value1, max_value=value2):
            np.clip(volume, min_value, max_value+np.finfo(volume.dtype).tiny, out=volume)
            volume -= min_value
            volume *= np.iinfo(z[0].dtype).max / (max_value - min_value)
            np.clip(volume, 0, np.iinfo(z[0].dtype).max+np.finfo(volume.dtype).tiny, out=volume)
            return volume.astype(z[0].dtype)

        nz = zarr.open(new_path, mode='w')
        
        with open(os.path.join(zarrpath, '.zattrs')) as f:
            metadata = json.load(f)
        with open(os.path.join(new_path, '.zattrs'), 'w') as f:
            json.dump(metadata, f)

        compressor = numcodecs.Blosc(cname='zstd', clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE)

        for array_name in z.array_keys():
            source_array = z[array_name]
            nz_array = nz.create_dataset(
                array_name,
                shape=source_array.shape,
                chunks=source_array.chunks,
                dtype=source_array.dtype,
                compressor=compressor,
            )
            nz_array.attrs.put(source_array.attrs.asdict())

            chunk_sizes = source_array.chunks
            shape = source_array.shape

            # Manually iterate over the chunks
            for i in range(0, shape[0], chunk_sizes[0]):
                for j in range(0, shape[1], chunk_sizes[1]):
                    for k in range(0, shape[2], chunk_sizes[2]):
                        chunk = source_array[
                            i:i+chunk_sizes[0],
                            j:j+chunk_sizes[1],
                            k:k+chunk_sizes[2]
                        ].astype(np.float32)
                        nz_array[
                            i:i+chunk_sizes[0],
                            j:j+chunk_sizes[1],
                            k:k+chunk_sizes[2]
                        ] = transform_function(chunk)

    print("Equalization completed successfully.")
