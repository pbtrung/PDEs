import torch
import numpy as np


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params += param
        print(f"{name}: {param}")
    print(f"Total Trainable Params: {total_params}\n\n")


def rmse(x, y):
    return np.sqrt(np.mean(np.square(x - y)))


def mse(x, y):
    return np.mean(np.square(x - y))


def mse(x, y):
    return np.mean(np.square(x - y))
