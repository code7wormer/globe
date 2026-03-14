import xarray as xr
import numpy as np

ds = xr.open_dataset('temperature.nc')
print("Variables:", list(ds.data_vars))
print("Coords:", list(ds.coords))
print("Dims:", dict(ds.dims))
print()
for var in ds.data_vars:
    print(f"--- {var} ---")
    arr = ds[var].values
    print("  dtype:", arr.dtype)
    print("  shape:", arr.shape)
    print("  min:", np.nanmin(arr))
    print("  max:", np.nanmax(arr))
    print("  NaN count:", np.isnan(arr).sum(), "/", arr.size)
    print()

print("Time range:", ds.time.values.min(), "to", ds.time.values.max())
