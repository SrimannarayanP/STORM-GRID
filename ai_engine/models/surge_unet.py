# surge_unet.py


from torch.utils.checkpoint import checkpoint

import torch.nn as nn

import logging, time, torch


class DoubleConv(nn.Module):

    """Std dual convolutional block used in U-Net archs. Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):

        return self.conv(x)


class ConvLSTMCell(nn.Module):

    """2D Convolutional LSTM Cell. Replaces matrix multiplications with convolutions to preserve spatial gradients."""

    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        padding = kernel_size//2

        # Stacking the gates (Input, Forget, Cell, Output) into a single conv layer for VRAM efficiency.
        self.conv = nn.Conv2d(in_channels = input_dim + hidden_dim, out_channels = 4*hidden_dim, kernel_size = kernel_size, padding = padding, bias = True)

    def forward(self, input_tensor, curr_state):
        h_curr, c_curr = curr_state

        combined = torch.cat([input_tensor, h_curr], dim = 1) # Concat spatial input with hidden state along the channel dim
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim = 1) # Split the tensor into the 4 LSTM gates

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f*c_curr + i*g
        h_next = o*torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        height, width = image_size

        return (torch.zeros(batch_size, self.hidden_dim, height, width, device = device), torch.zeros(batch_size, self.hidden_dim, height, width, device = device))


class RecurrentSurgeUNet(nn.Module):

    """Spatiotemporal Physics-Informed CNN. Embeds a ConvLSTM at the bottleneck level."""

    def __init__(self, in_channels = 6, out_channels = 3):
        super().__init__()

        # Encoder (Downsampling: 256 -> 128 -> 64 -> 32)
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2) # 256 -> 128
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2) # 128 -> 64
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2) # 64 -> 32
        # Bottleneck (Deepest learned rep of storm)
        self.lstm_bottleneck = ConvLSTMCell(input_dim = 256, hidden_dim = 256, kernel_size = 3)
        # Decoder (Upsampling: 32 -> 64 -> 128 -> 256 with properly aligned skip conns)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2) # 32 -> 64
        self.up_conv1 = DoubleConv(384, 128) # u1 (128) + d3 (256) = 384
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2) # 64 -> 128
        self.up_conv2 = DoubleConv(192, 64) # u2 (64) + d2 (128) = 192
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size = 2, stride = 2) # 128 -> 256
        self.up_conv3 = DoubleConv(96, 64) # u3 (32) + d1 (64) = 96

        # Final output layer (Maps 64 features down to 1 surge depth channel)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size = 1)

    def single_step_forward(self, curr_forces, curr_fluid, h_curr, c_curr):
        """Predicts the 40-sec residual delta (d_zeta, du, dv) added to curr_fluid."""

        x_t = torch.cat([curr_forces, curr_fluid], dim = 1) # (B, 6, 256, 256)

        d1 = self.down1(x_t) # (B, 64, 256, 256)
        p1 = self.pool1(d1) # (B, 64, 128, 128)
        d2 = self.down2(p1) # (B, 128, 128, 128)
        p2 = self.pool2(d2) # (B, 128, 64, 64)
        d3 = self.down3(p2) # (B, 256, 64, 64)
        p3 = self.pool3(d3) # (B, 256, 32, 32)

        h_next, c_next = self.lstm_bottleneck(p3, (h_curr, c_curr)) # (B, 256, 32, 32)

        u1 = self.up1(h_next) # (B, 128, 64, 64)
        u1 = torch.cat([u1, d3], dim = 1) # (B, 384, 64, 64)
        u1 = self.up_conv1(u1) # (B, 128, 64, 64)
        u2 = self.up2(u1) # (B, 64, 128, 128)
        u2 = torch.cat([u2, d2], dim = 1) # (B, 192, 128, 128)
        u2 = self.up_conv2(u2) # (B, 64, 128, 128)
        u3 = self.up3(u2) # (B, 32, 256, 256)
        u3 = torch.cat([u3, d1], dim = 1) # (B, 96, 256, 256) 
        u3_out = self.up_conv3(u3) # (B, 64, 256, 256)

        # Residual micro-step update: tanh bounds single-step change so 90-step rollout stays stable.
        delta = 0.002*torch.tanh(self.final_conv(u3_out))
        next_fluid = curr_fluid + delta

        return next_fluid, h_next, c_next

    def forward(self, forcing_sequence, initial_fluid_state, hidden_state = None, return_hidden = False):
        """
            forcing_sequence: (Batch, Seq_Len, 3, H, W) -> [bathy, wind_u, wind_v] over time
            initial_fluid_state: (Batch, 3, H, W) -> [h, u, v]
        """

        b, seq_len, c, h, w = forcing_sequence.size()

        if hidden_state is None:
            h_curr, c_curr = self.lstm_bottleneck.init_hidden(batch_size = b, image_size = (h//8, w//8), device = forcing_sequence.device)
        else:
            h_curr, c_curr = hidden_state

        curr_fluid = initial_fluid_state

        if not curr_fluid.requires_grad:
            curr_fluid = curr_fluid.detach().requires_grad_(True)

        fluid_outputs = []

        for t in range(seq_len):
            curr_forces = forcing_sequence[:, t, :, :, :]

            curr_fluid, h_curr, c_curr = checkpoint(self.single_step_forward, curr_forces, curr_fluid, h_curr, c_curr, use_reentrant = False)

            fluid_outputs.append(curr_fluid.unsqueeze(1))

        out_seq = torch.cat(fluid_outputs, dim = 1)

        if return_hidden:

            return out_seq, (h_curr, c_curr)

        return out_seq


if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s - [%(levelname)s] - %(message)s")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f"Running isolated 90-step arch & VRAM verification on: {device}")

    model = RecurrentSurgeUNet(in_channels = 6, out_channels = 3).to(device)

    # Batch = 2, steps = 90, channels = 3, 256x256
    dummy_forcing = torch.randn(2, 90, 3, 256, 256, device = device)
    dummy_init_fluid = torch.rand(2, 3, 256, 256, device = device) + 0.1

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()

    pred_seq = model(dummy_forcing, dummy_init_fluid)
    loss = pred_seq.mean()
    loss.backward()

    if device.type == 'cuda':
        torch.cuda.synchronize()

        peak_vram_gb = torch.cuda.max_memory_allocated(device)/(1024**3)
    else:
        peak_vram_gb = 0.0

    elapsed = time.perf_counter() - t0

    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

    logging.info(f"Output shape: {list(pred_seq.shape)} (Expected: [2, 90, 3, 256, 256])")
    logging.info(f"90-Step Forward + Backward Time: {elapsed:.2f}s | Peak VRAM: {peak_vram_gb:.2f}GB | Grad NormSum: {grad_norm:.4f}")