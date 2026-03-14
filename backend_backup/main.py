from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

app = FastAPI()

# Allow frontend to fetch from different origin (if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Serve static files
os.makedirs("./static/tiles", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Path to NetCDF
NC_FILE = "./temperature.nc"

def generate_image_for_date(year: int, month: int, day: int) -> str:
    """
    Selects the closest timestep in the dataset for the given year/month/day
    and saves a transparent-background heatmap PNG for Cesium overlay.
    """
    ds = xr.open_dataset(NC_FILE)

    # Find closest time index using fractional-year arithmetic
    if np.issubdtype(ds.time.dtype, np.floating):
        target_frac = year + (month - 0.5) / 12.0
        idx = np.argmin(np.abs(ds.time.values - target_frac))
    else:
        ds['time_dt'] = ds.time
        target_date = pd.Timestamp(year=year, month=month, day=day)
        idx = np.argmin(np.abs(ds['time_dt'].values - target_date))

    # Select temperature slice (anomaly) and reconstruct absolute temperature
    # by adding the climatological baseline for the requested month (0-indexed)
    anomaly = ds.temperature.isel(time=idx).values          # (lat, lon)
    clim    = ds.climatology.isel(month_number=month-1).values  # (lat, lon)
    # The anomaly is very small (±2-3 degrees) compared to the ±40 degree absolute scale,
    # so year-to-year changes are hard to see. Multiply anomaly by 3 to exaggerate it visually.
    temp_slice = (anomaly * 3.0) + clim  # emphasized absolute temperature

    # Build a masked array so matplotlib treats NaNs as transparent
    data = np.ma.masked_invalid(temp_slice)

    # Render with matplotlib: transparent background, no axes/borders
    # Custom vivid colormap: blue → cyan → green → yellow → red (no white)
    cmap = LinearSegmentedColormap.from_list(
        'vivid_temp',
        ['#0000cc', '#00aaff', '#00dd88', '#00ff00', '#aaff00', '#ffcc00', '#ff4400', '#cc0000'],
    )
    cmap.set_bad(alpha=0)  # NaN cells → fully transparent

    fig, ax = plt.subplots(figsize=(3.6, 1.8), dpi=100)
    fig.patch.set_alpha(0)
    ax.imshow(
        data,
        cmap=cmap,
        origin='lower',
        aspect='auto',
        vmin=-15,   # °C — shifted up so Greenland/Arctic easily hits saturated deep blue
        vmax=45,    # °C — shifted up so Sahara is red and India is orange
    )
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    filename = f"temp_{year}-{month:02d}-{day:02d}.png"
    output_path = os.path.join("./static/tiles", filename)
    fig.savefig(output_path, dpi=100, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return filename

@app.get("/generate_image")
def generate_image(year: int = Query(...), month: int = Query(...), day: int = Query(...)):
    filename = generate_image_for_date(year, month, day)
    return PlainTextResponse(f"/static/tiles/{filename}")

app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")