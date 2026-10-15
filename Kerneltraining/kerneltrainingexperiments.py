import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import pandas as pd
import matplotlib.pyplot as plt 
from tqdm import tqdm

step = 2
map_range = 40
#ground truth data subsampling
# csv_path = r"C:\Users\Aksha\OneDrive\Year 6\Thesis\Datasets\NAIP_dataset\selected_tiles_csv"
csv_path = r"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\csv"


length_scales = []
stand_devs = []
nus = [] 




for map_id in tqdm(range(1, map_range), desc="Processing maps"):
    # csv_file = rf"{csv_path}\map_{map_id}_NAIP_grid_counts.csv"
    csv_file = rf"{csv_path}\map_{map_id}_multiblob_normalized_grid_counts.csv"
    df = pd.read_csv(csv_file)
    df_sub = df[(df["x"] % step == 0 ) & (df["y"] % step == 0 )].copy()

    X_train = df_sub[["x", "y"]].to_numpy(dtype=np.float64)
    y_train = df_sub["count"].to_numpy(dtype=np.float64)
    prior_threshold = 0.3
    y_residual = y_train - prior_threshold
    signal_var = np.var(y_train)
    #regression
    Matern_kernel = ConstantKernel(signal_var, constant_value_bounds="fixed") * Matern(length_scale=10.0, length_scale_bounds=(1e-2, 1e3), nu=1.5)
    gp = GaussianProcessRegressor(kernel=Matern_kernel, n_restarts_optimizer=0, alpha=1e-4, normalize_y=False)

    gp.fit(X_train, y_train)

    fitted_kernel = gp.kernel_
    length_scales.append(fitted_kernel.k2.length_scale)
    constant_value = fitted_kernel.k1.constant_value
    stand_devs.append(np.sqrt(constant_value))
    nus.append(fitted_kernel.k2.nu)

    print(gp.kernel_)


    # xs = np.sort(df["x"].unique())
    # ys = np.sort(df["y"].unique())
    # X, Y = np.meshgrid(xs, ys)
    # X_test = np.column_stack([X.ravel(), Y.ravel()])

    # mu, std = gp.predict(X_test, return_std=True)
    # mu_map = mu.reshape(X.shape)
    # std_map = std.reshape(X.shape)
    # true_map = df["count"].to_numpy(dtype=np.float64).reshape(X.shape)   


    # fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # im0 = axes[0].imshow(true_map, origin="lower", extent=[0, 100, 0, 100])
    # axes[0].set_title("True map")
    # plt.colorbar(im0, ax=axes[0])

    # im1 = axes[1].imshow(mu_map, origin="lower", extent=[0, 100, 0, 100])
    # axes[1].set_title("GP predicted mean")
    # plt.colorbar(im1, ax=axes[1])

    # im2 = axes[2].imshow(std_map, origin="lower", extent=[0, 100, 0, 100])
    # axes[2].set_title("GP predicted std")
    # plt.colorbar(im2, ax=axes[2])

    # plt.tight_layout()
    # plt.show()



print("median signal std:", np.median(stand_devs))
print("median signal var:", np.median(np.square(stand_devs)))
print("median length scale:", np.median(length_scales))
print("median nu:", np.median(nus))
