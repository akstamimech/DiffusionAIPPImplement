from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv

dataset_id_range = 200
path = r"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\csv"
for id in range(0, dataset_id_range + 1):


    file = f"{path}/map_{id}_multiblob.csv"

    data = np.loadtxt(file, delimiter=",", skiprows=1)

    # columns: map_id, x, y
    pts = data[:, 1:3]  # shape (N, 2)

    xmin, xmax = 0.0, 100.0
    ymin, ymax = 0.0, 100.0
    step = 1.0          
    radius = 6.0        
    r2 = radius**2

    xs = np.arange(xmin, xmax + 1e-9, step)
    ys = np.arange(ymin, ymax + 1e-9, step)
    X, Y = np.meshgrid(xs, ys, indexing="xy")  # shape (Ny, Nx)


    min_d2 = np.full(X.shape, np.inf)
    counts = np.zeros(X.shape, dtype=np.int32)
    for px, py in pts:
        counts += (((X - px)**2 + (Y - py)**2) <= r2)



    heatmap = counts

    # plt.imshow(
    #     heatmap,
    #     origin="lower",
    #     extent=[xs.min(), xs.max(), ys.min(), ys.max()],
    #     interpolation="nearest",
    #     aspect="equal",
    #     cmap="hot"
    # )

    # plt.colorbar(label="number of trees within radius 4")
    # # plt.scatter(pts[:,0], pts[:,1], s=8, color="blue")
    # plt.title(f"multiblob Map {id} - Density Heatmap")
    # plt.xlabel("X coordinate")
    # plt.ylabel("Y coordinate")
    # plt.show()

    heatmap = (min_d2 <= r2).astype(np.uint8)  # 1 = occupied, 0 = empty



    # plt.imshow(
    #     heatmap,
    #     origin="lower",
    #     extent=[xs.min(), xs.max(), ys.min(), ys.max()],
    #     interpolation="nearest",
    #     aspect="equal",
    # )
    # plt.scatter(pts[:,0], pts[:,1], s=8)
    # plt.show()

    out_file = f"{path}/map_{id}_multiblob_grid_counts.csv"
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "count"])

        for x, y, c in zip(X.ravel(), Y.ravel(), counts.ravel()):
            writer.writerow([x, y, c])