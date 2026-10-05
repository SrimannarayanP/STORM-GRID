# train.py


from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim

import torch, logging


from surge_dataset import HydroDataset
from surge_unet import SurgeUNet


logging.basicConfig(level = logging.INFO, format = "%(asctime)s - [%(levelname)s] - %(message)s")


class PhysicsInformedLoss(nn.Module):

    """Evals both the statistical err against NOAA's ground truth & the physical validity of the predicted fluid dynamics."""

    def __init__(self, lambda_phy = 0.1):
        super().__init__()

        self.mse = nn.MSELoss() # Mean Squared Err
        self.lambda_phy = lambda_phy

    def forward(self, pred_surge, true_surge):
        # Std supervised learning err against NOAA data
        data_loss = self.mse(pred_surge, true_surge)

        # H2O surfaces don't have 90-degree spikes. They pool & flow cont-ly. We use finite diffs to calc the spatial gradients (dx, dy) of the prediction. If the
        # network predicts a sharp, physically impossible cliff in H2O level, these derivatives will explode, penalizing the model's weights.
        dx = torch.abs(pred_surge[:, :, :, 1:] - pred_surge[:, :, :, :-1])
        dy = torch.abs(pred_surge[:, :, 1:, :] - pred_surge[:, :, :-1, :])

        physics_loss = torch.mean(dx) + torch.mean(dy)
        total_loss = data_loss + (self.lambda_phy*physics_loss)

        return total_loss, data_loss, physics_loss


def train_surrogate():
    # Init infra
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f"Compute engine: {device}")

    model = SurgeUNet(in_channels = 3, out_channels = 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    criterion = PhysicsInformedLoss(lambda_phy = 0.15)

    # Mount the data by loading a single tensor you guaranteed to attach.
    dataset = HydroDataset('../data_pipeline/data/training_tensors')
    dataloader = DataLoader(dataset, batch_size = 16, shuffle = True, pin_memory = True)

    epochs = 5

    logging.info("Init-ing Physics-Informed Backpropagation...")

    for epoch in range(epochs):
        model.train()

        running_loss = running_d_loss = running_p_loss = 0.0

        for batch_idx, (x_features, y_label) in enumerate(dataloader):
            # Move the matrices to CPU/GPU
            x_features = x_features.to(device)
            y_label = y_label.to(device)

            # 0 the gradients from the prev step
            optimizer.zero_grad()

            prediction = model(x_features) # Forward pass: Predict the flood map

            loss, d_loss, p_loss = criterion(prediction, y_label) # Compute PINN loss.

            loss.backward() # Backward pass: Calc derivatives for all the weights.

            optimizer.step() # Adjust weights to minimize loss.

            # To find the true mean of the entire epoch.
            running_loss += loss.item()
            running_d_loss += d_loss.item()
            running_p_loss += p_loss.item()

        num_batches = len(dataloader)
        epoch_loss = running_loss/num_batches
        epoch_d_loss = running_d_loss/num_batches
        epoch_p_loss = running_p_loss/num_batches

        # Checkpoint the weights to the disk.
        torch.save(model.state_dict(), 'surge_unet_weights.pt')

        logging.info("Model weights saved to surge_unet_weights.pt")

        logging.info(f"Epoch {epoch + 1}/{epochs} | Total loss: {running_loss:.4f} (Data: {d_loss.item():.4f}, Physics: {p_loss.item():.4f})")

    logging.info("Training test complete. Arch is fully operational.")


train_surrogate()