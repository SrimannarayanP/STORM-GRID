# eval_rollout.py


from pathlib import Path

from config import CHECKPOINTS_DIR, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, MAX_SURGE, MAX_TOPO, MAX_VEL, MAX_WIND, PROCESSED_DIR, TRAINING_TENSORS_DIR
from models.surge_unet import RecurrentSurgeUNet

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import logging, matplotlib, torch


matplotlib.use('Agg') # Headless rendering for WSL2

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - [%(levelname)s] - %(message)s")


def eval_24h_rollout(rollout_hrs = 24, sub_steps_per_hr = 12, start_t = 0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    chkpts = sorted(CHECKPOINTS_DIR.glob('surge_unet_weights_epoch_*.pt'))

    if not chkpts:
        raise FileNotFoundError(f"No trained weights found in {CHECKPOINTS_DIR}. Run train.py 1st.")

    latest_chkpt = chkpts[-1]

    tensor_files = sorted(TRAINING_TENSORS_DIR.glob('structured_surge_*.nc'))

    if not tensor_files:
        raise FileNotFoundError(f"No struct-ed tensors found in {TRAINING_TENSORS_DIR}.")

    test_file = tensor_files[0]

    logging.info(f"Loading weights: {latest_chkpt.name} | Eval-ing on: {test_file.name}")

    model = RecurrentSurgeUNet(in_channels = 6, out_channels = 3).to(device)
    model.load_state_dict(torch.load(latest_chkpt, map_location = device, weights_only = True))
    model.eval()

    with xr.open_dataset(test_file, engine = 'h5netcdf') as ds:
        topo = np.nan_to_num(ds['topo'].values, nan = 0.0)
        surge_h = np.nan_to_num(ds['surge_h'].values, nan = 0.0)
        surge_u = np.nan_to_num(ds['surge_u'].values, nan = 0.0)
        surge_v = np.nan_to_num(ds['surge_v'].values, nan = 0.0)
        wind_u = np.nan_to_num(ds['wind_u'].values, nan = 0.0)
        wind_v = np.nan_to_num(ds['wind_v'].values, nan = 0.0)

    # Wet ocean mask (where ocean floor topo < 0m) so land 0s don't artificially deflate err matrices.
    ocean_mask = topo < 0.0

    static_bathy = torch.from_numpy(topo).unsqueeze(0).unsqueeze(0).float().to(device)
    curr_fluid = torch.from_numpy(np.stack([surge_h[start_t], surge_u[start_t], surge_v[start_t]], axis = 0)).unsqueeze(0).float().to(device)

    scale_forces = torch.tensor([MAX_TOPO, MAX_WIND, MAX_WIND], device = device).view(1, 3, 1, 1)
    scale_fluid = torch.tensor([MAX_SURGE, MAX_VEL, MAX_VEL], device = device).view(1, 3, 1, 1)

    curr_fluid_scaled = curr_fluid/scale_fluid
    
    snapshots = {}

    horizons_to_plot = [1, 6, 12, 24]

    hidden_state = None

    with torch.no_grad():
        for h in range(1, rollout_hrs + 1):
            t0 = start_t + h - 1
            t1 = start_t + h

            w_t0 = torch.from_numpy(np.stack([wind_u[t0], wind_v[t0]], axis = 0)).unsqueeze(0).float().to(device)
            w_t1 = torch.from_numpy(np.stack([wind_u[t1], wind_v[t1]], axis = 0)).unsqueeze(0).float().to(device)

            forcing_seq = []

            for step in range(1, sub_steps_per_hr + 1):
                alpha = step/sub_steps_per_hr

                curr_w = (1.0 - alpha)*w_t0 + alpha*w_t1
                curr_f = torch.cat([static_bathy, curr_w], dim = 1)
                forcing_seq.append((curr_f/scale_forces).unsqueeze(1))

            forcing_seq = torch.cat(forcing_seq, dim = 1)

            # Auto-regressive step: Model receives its own prev prediction as curr_fluid_scaled
            pred_seq_scaled, hidden_state = model(forcing_seq, curr_fluid_scaled, hidden_state = hidden_state, return_hidden = True)
            curr_fluid_scaled = pred_seq_scaled[:, -1, :, :, :]

            pred_h_m = (curr_fluid_scaled[0, 0].cpu().numpy())*MAX_SURGE
            true_h_m = surge_h[t1]

            # Calc err strictly over wet ocean pixels in cm
            diff_cm = (pred_h_m - true_h_m)[ocean_mask]*100.0
            rmse_cm = float(np.sqrt(np.mean(diff_cm**2)))
            mae_cm = float(np.mean(np.abs(diff_cm)))

            if h in horizons_to_plot or h%6 == 0:
                logging.info(f"Lead time +{h:02d}h | Ocean RMSE: {rmse_cm:6.2f}cm | Ocean MAE: {mae_cm:6.2f}cm")

            if h in horizons_to_plot:
                snapshots[h] = (true_h_m, pred_h_m, np.abs(pred_h_m - true_h_m)*100.0, rmse_cm)

    fig, axes = plt.subplots(len(horizons_to_plot), 3, figsize = (15, 16), constrained_layout = True)
    
    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

    for row_idx, h in enumerate(horizons_to_plot):
        true_grid, pred_grid, err_grid_cm, rmse_cm = snapshots[h]

        true_masked = np.ma.masked_where(~ocean_mask, true_grid)
        pred_masked = np.ma.masked_where(~ocean_mask, pred_grid)
        err_masked = np.ma.masked_where(~ocean_mask, err_grid_cm)

        vmin = min(float(np.min(true_masked)), float(np.min(pred_masked)))
        vmax = max(float(np.max(true_masked)), float(np.max(pred_masked)))

        im0 = axes[row_idx, 0].imshow(true_masked, origin = 'lower', extent = extent, cmap = 'viridis', vmin = vmin, vmax = vmax)
        axes[row_idx, 0].set_title(f"+{h}h Ground Truth ADCIRC (m)")
        plt.colorbar(im0, ax = axes[row_idx, 0], fraction = 0.046)

        im1 = axes[row_idx, 1].imshow(pred_masked, origin = 'lower', extent = extent, cmap = 'viridis', vmin = vmin, vmax = vmax)
        axes[row_idx, 1].set_title(f"+{h}h PINN Rollout (m)")
        plt.colorbar(im1, ax = axes[row_idx, 1], fraction = 0.046)

        im2 = axes[row_idx, 2].imshow(err_masked, origin = 'lower', extent = extent, cmap = 'magma', vmin = vmin)
        axes[row_idx, 2].set_title(f"+{h}h Abs err (cm) | RMSE: {rmse_cm:.2f}cm")
        plt.colorbar(im2, ax = axes[row_idx, 2], fraction = 0.046)

    out_plot = PROCESSED_DIR/'rollout_24h_eval.png'
    plt.savefig(out_plot, dpi = 200)
    plt.close(fig)

    logging.info(f"Saved 24-hr autoregressive rollout diagnostic plot to: {out_plot}")


if __name__ == '__main__':
    eval_24h_rollout()