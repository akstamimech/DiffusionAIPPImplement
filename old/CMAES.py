import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
from matplotlib.patches import Rectangle
from gaussianprocesstraining import utility_function, sampler, create_plots_and_gifs, kalman_update, initialize_gp

horizon = 4


def grid_search(X, Y, cx, cy, utility_history, grid_distance = 4.0):
    grid_points = []
    for x in np.arange(X.min(), X.max() + 1e-9, grid_distance):
        for y in np.arange(Y.min(), Y.max() + 1e-9, grid_distance):
            grid_points.append((x, y))

    util_values = []
    for xg, yg in grid_points:
        idx = np.where(
            np.isclose(X.ravel(), xg) &
            np.isclose(Y.ravel(), yg)
        )[0]

        if len(idx) == 0:
            continue

        idx = idx[0]
        util_value = utility_history[-1][idx]
        util_values.append((util_value, (xg, yg)))
        # print(f"Utility at ({xg}, {yg}): {util_value:.4f}")
    
    #grid search loop for receding horizon:
    for i in range(horizon):
        util_values.sort(key=lambda x: x[0]/np.hypot(x[1][0] - cx, x[1][1] - cy), reverse=True)
    return util_values[0] # Return top utility values and their corresponding coordinates


if __name__ == "__main__":
    path = r"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\csv"
    data = np.loadtxt(rf"{path}\map_1_grid_counts.csv", delimiter=",", skiprows=1)


    gp, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = initialize_gp()

    # discretized points
    pts = data[:, 0:3]
    tol = 1e-9
    mask = (
        np.isclose(np.mod(pts[:, 0], step), 0.0, atol=tol)
        & np.isclose(np.mod(pts[:, 1], step), 0.0, atol=tol)
    )
    pts = pts[mask]

    N = X_test.shape[0]
    mu = mean.copy()
    P = cov.copy()
    R = 1e-6

    mu_history = []
    P_history = []
    step_numbers = []
    grad_history = []
    pos_history = []
    utility_history = []
    sorted_util_values_list = []

    mu_history.append(mu.copy())
    P_history.append(P.copy())
    step_numbers.append(0)
    utility_threshold = 4.0

    initial_utility = utility_function(mu, P, utility_threshold)
    utility_history.append(initial_utility)


    save_every = 5
    lateral_coverage = step * 2
    samplestep = 4.0
    timealloted = 150

   

    # initial stating position and gradient
    cx, cy = 50.0, 50.0
    grad_x, grad_y = 0.0, 0.0
    pos_history.append((cx, cy))


    # store initial gradient field
    initial_var_field = np.diag(P_history[-1]).reshape(X.shape)
    gy0, gx0 = np.gradient(initial_var_field, Y[:, 0], X[0, :])
    grad_history.append((gx0, gy0))

    for ts in range(0, timealloted):
        cx = np.clip(cx + grad_x + np.ceil(np.random.uniform(-step, step)), xmin, xmax)
        cy = np.clip(cy + grad_y + np.ceil(np.random.uniform(-step, step)), ymin, ymax)
        cx = step * np.round(cx / step)
        cy = step * np.round(cy / step)
        pos_history.append((cx, cy))

        print(cx, cy)
        step_numbers.append(ts + 1)

        fov = [
            (x, y)
            for x in np.arange(cx - lateral_coverage, cx + lateral_coverage + 1e-9, step)
            for y in np.arange(cy - lateral_coverage, cy + lateral_coverage + 1e-9, step)
        ]

        sensor = np.zeros((len(fov), N))

        for i, (x_meas, y_meas) in enumerate(fov):
            idx = np.where(
                np.isclose(X_test[:, 0], x_meas) &
                np.isclose(X_test[:, 1], y_meas)
            )[0]

            if len(idx) == 0:
                continue

            idx = idx[0]
            sensor[i, idx] = 1.0

        measurement_list = []
        for x_fov, y_fov in fov:
            idx = np.where(
                np.isclose(pts[:, 0], x_fov) &
                np.isclose(pts[:, 1], y_fov)
            )[0]

            if idx.size > 0:
                measurement_list.append(pts[idx[0], 2])
            else:
                measurement_list.append(0.0)

        z_meas = np.array(measurement_list)

        mu, P = kalman_update(mu, P, sensor, z_meas, R)

        

        mu_history.append(mu.copy())
        P_history.append(P.copy())
        utility = utility_function(mu, P, utility_threshold, beta = 4.0)
        utility_history.append(utility)
        sorted_util_value = grid_search(X, Y, cx, cy, utility_history)
        sorted_util_values_list.append(sorted_util_value)
        
        util, (target_x, target_y) = grid_search(X, Y, cx, cy, utility_history)
        grad_x, grad_y = (target_x - cx) / np.linalg.norm((target_x - cx)), (target_y - cy) / np.linalg.norm((target_y - cy))

        if np.isnan(grad_x) or np.isnan(grad_y):
            grad_x, grad_y = 0.0, 0.0

        # grad, grad_field, target = sampler(cx, cy, X, Y, P_history, samplestep)
        # grad_x, grad_y = grad
        # target_x, target_y = target

        # grad_history.append(grad_field)

        print("step vector:", grad_x, grad_y)
        print("target:", target_x, target_y)
        print(sorted_util_value)


    print(sorted_util_values_list)
    create_plots_and_gifs(path, mu_history, P_history, step_numbers, grad_history, pos_history, utility_history, sorted_util_values_list, X, Y, xs, ys, cx, cy, lateral_coverage, xmin, xmax, ymin, ymax)
