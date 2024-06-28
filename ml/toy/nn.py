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

output_dir = "nn"
writer = SummaryWriter(output_dir)


def train_nn(model, epochs, batch_size, dataset):
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
        total_loss = 0.0

        start = time.time()
        start_batch = time.time()
        sum_diff = 0.0
        for batch, (input_true_batch, u_true_batch) in enumerate(dataloader):

            input_true_batch = input_true_batch.to(device)
            u_true_batch = u_true_batch.to(device)
            u_pred_true = model(input_true_batch)

            loss = criterion(u_pred_true, u_true_batch)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end_batch = time.time()
            diff = end_batch - start_batch - sum_diff
            sum_diff += diff

            if batch % 4 == 0 or batch == len(dataloader) - 1:
                print(f"Epoch {epoch}, Batch {batch:2d}, Took {diff:.4f}s, loss: {loss.item():.6f}")

        end_epoch = time.time()
        print(
            f"Epoch {epoch}, Took {end_epoch-start:.4f}s, Average loss = {total_loss / len(dataloader):.6f}"
        )

        writer.add_scalar("loss", total_loss / len(dataloader), epoch)
        writer.flush()

        torch.save(model, f"{output_dir}/nn_{epoch:05d}.pt")
        time.sleep(2)


if __name__ == "__main__":

    model = NN(input_dim=3, hidden_layers=3, hidden_dim=1200, output_dim=1).to(device)
    utils.count_parameters(model)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!\n\n")
        model = nn.DataParallel(model)

    epochs = 10000
    batch_size = int(6e6)
    dataset = PDEDataset("data.h5", "training_dataset")
    train_nn(model, epochs, batch_size, dataset)
