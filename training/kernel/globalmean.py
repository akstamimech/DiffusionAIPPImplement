# --- repo root on sys.path (this file moved into a sub-package) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# ---
import arcpy
import numpy as np
from pathlib import Path

folder = Path(r"C:\Users\Aksha\OneDrive\Year 6\Thesis\Datasets\NAIP_dataset\selected_tiles")

total_sum = 0.0
total_count = 0

for tif in folder.glob("*.tif"):
    arr = arcpy.RasterToNumPyArray(str(tif)).astype(float)

    valid = np.isfinite(arr)
    total_sum += arr[valid].sum()
    total_count += valid.sum()

global_mean = total_sum / total_count
print(global_mean)