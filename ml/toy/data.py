import torch
import numpy as np
import h5py


# du/dx = 2a * du/dt + 3a * u
# u(x, 0) = 6e^(-ax)
# u(x, t) = 6e^(-ax - 2t)
# a ~ log-normal(0, 1)

num_a_training = 1000
num_a_test = 1000
num_a_val = 100
num_points = 100
x = np.linspace(0, 1, num_points)
t = np.linspace(0, 1, num_points)
x_mesh, t_mesh = np.meshgrid(x, t)

with h5py.File("data.h5", "w") as hf:
    u_true_dset = hf.create_dataset(
        "training_dataset",
        (num_a_training * num_points * num_points, 4),
        dtype="f8",
        compression="gzip",
        compression_opts=9,
    )
    u_test_dset = hf.create_dataset(
        "test_dataset",
        (num_a_test * num_points * num_points, 4),
        dtype="f8",
        compression="gzip",
        compression_opts=9,
    )
    u_val_dset = hf.create_dataset(
        "val_dataset",
        (num_a_val * num_points * num_points, 4),
        dtype="f8",
        compression="gzip",
        compression_opts=9,
    )

    for i in range(num_a_training):
        a = np.random.lognormal(1)

        u_true = 6 * np.exp(-a * x_mesh - 2 * t_mesh)
        a_part_true = np.full_like(u_true, a)
        tmp_true = np.vstack(
            (a_part_true.flatten(), x_mesh.flatten(), t_mesh.flatten(), u_true.flatten())
        ).T
        u_true_dset[i * num_points**2 : (i + 1) * num_points**2, :] = tmp_true

    for i in range(num_a_test):
        a = np.random.lognormal(1)

        u_true = 6 * np.exp(-a * x_mesh - 2 * t_mesh)
        a_part_true = np.full_like(u_true, a)
        tmp_true = np.vstack(
            (a_part_true.flatten(), x_mesh.flatten(), t_mesh.flatten(), u_true.flatten())
        ).T
        u_test_dset[i * num_points**2 : (i + 1) * num_points**2, :] = tmp_true

    for i in range(num_a_val):
        a = np.random.lognormal(1)

        u_true = 6 * np.exp(-a * x_mesh - 2 * t_mesh)
        a_part_true = np.full_like(u_true, a)
        tmp_true = np.vstack(
            (a_part_true.flatten(), x_mesh.flatten(), t_mesh.flatten(), u_true.flatten())
        ).T
        u_val_dset[i * num_points**2 : (i + 1) * num_points**2, :] = tmp_true
