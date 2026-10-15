import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process.kernels import ConstantKernel, Matern

# Grid
step = 2.0
xmin, xmax = 0.0, 100.0
ymin, ymax = 0.0, 100.0

xs = np.arange(xmin, xmax + 1e-9, step)
ys = np.arange(ymin, ymax + 1e-9, step)

Xg, Yg = np.meshgrid(xs, ys)
X_test = np.column_stack([Xg.ravel(), Yg.ravel()])

# Learned kernel
kernel = 0.109**2 * Matern(length_scale=4.79, nu=1.5)

# Prior covariance
P = kernel(X_test)

# Prior variance map
prior_var = np.diag(P).reshape(len(ys), len(xs))

plt.figure(figsize=(6, 5))
plt.imshow(
    prior_var,
    origin="lower",
    extent=[xmin, xmax, ymin, ymax],
    cmap="viridis",
)
plt.colorbar(label="prior variance")
plt.title("GP Prior Variance Before Updates")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()