import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from classes import NN, PDEDataset
import utils


# CUDA Check and Device Selection
assert torch.cuda.is_available(), "CUDA is not available."
device = torch.device("cuda")

output_dir = "pinn"
writer = SummaryWriter(output_dir)


def train_pinn(model, epochs, batch_size, dataset):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=15,
        prefetch_factor=2,
        persistent_workers=True,
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_ic_loss = 0.0
        total_bc_loss = 0.0
        total_pde_loss = 0.0
        total_loss = 0.0

        start = time.time()
        start_batch = time.time()
        sum_diff = 0.0
        for batch, (input_true_batch, u_true_batch) in enumerate(dataloader):

            input_true_batch = input_true_batch.to(device)
            u_true_batch = u_true_batch.to(device)

            # IC loss
            input_ic = torch.hstack((input_true_batch[:, 0:2], torch.zeros_like(u_true_batch)))
            u_ic_pred = model(input_ic)
            u_ic_true = 6 * torch.exp(-input_true_batch[:, 0] * input_true_batch[:, 1])
            u_ic_true = u_ic_true.unsqueeze(1)
            ic_loss = criterion(u_ic_true, u_ic_pred)

            # BC loss
            input_bc = torch.hstack(
                (
                    input_true_batch[:, 0].unsqueeze(1),
                    torch.zeros_like(u_true_batch),
                    input_true_batch[:, 2].unsqueeze(1),
                )
            )
            u_bc_pred = model(input_bc)
            u_bc_true = 6 * torch.exp(-2 * input_true_batch[:, 2])
            u_bc_true = u_bc_true.unsqueeze(1)
            bc_loss = criterion(u_bc_true, u_bc_pred)

            # PDE loss
            input_true_batch.requires_grad_(True)
            u_pred = model(input_true_batch)
            u_pred = u_pred.squeeze(1)

            (u_x,) = torch.autograd.grad(u_pred.sum(), input_true_batch, create_graph=True)
            u_x = u_x[:, 1]
            (u_t,) = torch.autograd.grad(u_pred.sum(), input_true_batch, create_graph=True)
            u_t = u_t[:, 2]

            pde_loss = criterion(
                u_x, 2 * input_true_batch[:, 0] * u_t + 3 * input_true_batch[:, 0] * u_pred
            )

            loss = ic_loss + bc_loss + pde_loss

            total_ic_loss += ic_loss.item()
            total_bc_loss += bc_loss.item()
            total_pde_loss += pde_loss.item()
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end_batch = time.time()
            diff = end_batch - start_batch - sum_diff
            sum_diff += diff

            if batch % 4 == 0 or batch == len(dataloader) - 1:
                print(
                    f"Epoch {epoch}, Batch {batch:2d}, Took {diff:.4f}s, ic_loss: {ic_loss.item():.6f}, bc_loss: {bc_loss.item():.6f}, pde_loss: {pde_loss.item():.6f}, loss: {loss.item():.6f}"
                )

        end_epoch = time.time()
        print(
            f"Epoch {epoch}, Took {end_epoch-start:.4f}s, Average ic_loss = {total_ic_loss / len(dataloader):.6f}, Average bc_loss = {total_bc_loss / len(dataloader):.6f}, Average pde_loss = {total_pde_loss / len(dataloader):.6f}, Average loss = {total_loss / len(dataloader):.6f}"
        )

        writer.add_scalar("ic_loss", total_ic_loss / len(dataloader), epoch)
        writer.add_scalar("bc_loss", total_bc_loss / len(dataloader), epoch)
        writer.add_scalar("pde_loss", total_pde_loss / len(dataloader), epoch)
        writer.add_scalar("loss", total_loss / len(dataloader), epoch)
        writer.flush()

        torch.save(model, f"{output_dir}/pinn_{epoch:05d}.pt")
        time.sleep(2)


if __name__ == "__main__":

    model = NN(input_dim=3, hidden_layers=3, hidden_dim=1000, output_dim=1).to(device)
    utils.count_parameters(model)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!\n\n")
        model = nn.DataParallel(model)

    epochs = 10000
    batch_size = int(1.2e6)
    dataset = PDEDataset("data.h5", "training_dataset")
    train_pinn(model, epochs, batch_size, dataset)
