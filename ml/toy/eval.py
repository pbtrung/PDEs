import sys

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, entropy

from classes import NN, PDEDataset
import utils


# CUDA must be available
assert torch.cuda.is_available(), "CUDA is not available"
device = torch.device("cuda:0")


def eval(model, inputs, outputs, batch_size):
    mse_loss = nn.MSELoss()

    num_batches = len(outputs) // batch_size
    total_loss = 0.0
    for i in range(num_batches):
        input_batch = inputs[i * batch_size : (i + 1) * batch_size, :]
        input_batch = input_batch.to(device)
        output_batch = outputs[i * batch_size : (i + 1) * batch_size]
        output_batch = output_batch.to(device)

        u_pred = model(input_batch)
        u_pred = u_pred.squeeze(1)
        output_batch = output_batch.squeeze(1)
        loss = mse_loss(u_pred, output_batch)
        total_loss += loss.item()

        if i % 10 == 0 or i == num_batches - 1:
            print(f"Batch {i:3d}, loss: {loss:.6f}")

    print(f"total_loss: {total_loss / num_batches:.6f}")


def plot(output_dir, inputs, outputs, model, start_idx):
    a_input = inputs[start_idx:-1:10000, 0].detach().cpu().numpy()
    output = outputs[start_idx:-1:10000].detach().cpu().numpy()

    pred = model(inputs[start_idx:-1:10000, :].to(device))
    pred = pred.detach().cpu().numpy()

    print(f"x = {inputs[start_idx, 1]:.6f}, t = {inputs[start_idx, 2]:.6f}")
    print(f"MSE: {utils.mse(output, pred):.6f}")
    print(f"RMSE: {utils.rmse(output, pred):.6f}")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(a_input, output, s=5, label="truth")
    plt.scatter(a_input, pred, s=5, label="pred")
    plt.xlabel(f"a, x = {inputs[start_idx, 1]:.6f}, t = {inputs[start_idx, 2]:.6f}")
    plt.ylabel("u(a, x, t)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(a_input, np.log(np.abs(output - pred)), s=5, label="diff")
    plt.xlabel(f"a, x = {inputs[start_idx, 1]:.6f}, t = {inputs[start_idx, 2]:.6f}")
    plt.ylabel("log(diff)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot.pdf")
    plt.close()


def plot_mse(output_dir, inputs, outputs, start_idx, epoch):
    output = outputs[start_idx:-1:10000].detach().cpu().numpy()

    mse = []
    for i in range(epoch):
        fname = f"{output_dir}/{output_dir}_{i:05d}.pt"
        model = torch.load(fname).to(device)

        pred = model(inputs[start_idx:-1:10000, :].to(device))
        pred = pred.detach().cpu().numpy()

        i_mse = utils.mse(output, pred)
        mse.append(i_mse)
        if i % 100 == 0 or i == epoch - 1:
            print(f"MSE: {i_mse:.6f}")

    plt.plot(range(epoch), np.log(mse), label="log(mse)")
    plt.xlabel("epoch")
    plt.ylabel("log(mse)")
    plt.legend()
    plt.title(f"x = {inputs[start_idx, 1]:.6f}, t = {inputs[start_idx, 2]:.6f}")
    plt.savefig(f"{output_dir}/mse.pdf")
    plt.close()


def plot_pdf(output_dir, inputs, outputs, start_idx, epoch):
    output = outputs[start_idx:-1:10000].detach().cpu().numpy()

    fname = f"{output_dir}/{output_dir}_{epoch:05d}.pt"
    model = torch.load(fname).to(device)
    pred = model(inputs[start_idx:-1:10000, :].to(device))
    pred = pred.detach().cpu().numpy()

    # Kernel Density Estimation
    kde_output = gaussian_kde(output[:, 0])
    kde_pred = gaussian_kde(pred[:, 0])

    x_eval = np.linspace(np.min(output[:, 0]) - 1, np.max(output[:, 0]) + 1, 1000)
    pdf_output = kde_output(x_eval)
    pdf_pred = kde_pred(x_eval)

    print(f"Shannon entropy: {entropy(pdf_pred, pdf_output):.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(x_eval, pdf_output, label="truth")
    plt.plot(x_eval, pdf_pred, label="pred")
    plt.xlabel("Sample Value")
    plt.ylabel("Density")
    plt.title(f"PDF, x = {inputs[start_idx, 1]:.6f}, t = {inputs[start_idx, 2]:.6f}")
    plt.legend()
    plt.savefig(f"{output_dir}/pdf.pdf")
    plt.close()


if __name__ == "__main__":
    output_dir = sys.argv[1]
    epoch = int(sys.argv[2])
    start_idx = int(sys.argv[3])

    fname = f"{output_dir}/{output_dir}_{epoch:05d}.pt"
    model = torch.load(fname).to(device)

    dataset = PDEDataset("data.h5", "test_dataset")
    inputs = dataset.get_inputs()
    outputs = dataset.get_outputs()
    batch_size = int(0.5e6)
    eval(model, inputs, outputs, batch_size)

    plot(output_dir, inputs, outputs, model, start_idx)

    plot_mse(output_dir, inputs, outputs, start_idx, epoch)
    plot_pdf(output_dir, inputs, outputs, start_idx, epoch)
