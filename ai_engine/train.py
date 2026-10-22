# train.py


from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim

import logging, math, torch


from config import (CHECKPOINTS_DIR, GRID_RES, LAT_MAX, LAT_MIN, MAX_SURGE, MAX_TOPO, MAX_VEL, MAX_WIND, TRAINING_TENSORS_DIR)
from models.surge_dataset import HydroDataset
from models.surge_unet import RecurrentSurgeUNet


logging.basicConfig(level = logging.INFO, format = "%(asctime)s - [%(levelname)s] - %(message)s")


class AutoregressivePINNLoss(nn.Module):

    """Enforces the 2D Depth-Integrated Shallow H2O Equations with planetary rotation."""

    def __init__(self, dx = 390.625, dy = 390.625, dt = 40.0, dry_tolerance = 0.05):
        super().__init__()

        self.mse = nn.MSELoss() # Mean Squared Err
        # dx & dy in m (e.g. 100 km/256 pixels = ~390.625 m/pixel)
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.dry_tolerance = dry_tolerance # 5 cm threshold for "dry land"
        self.gravity = 9.81
        self.manning_n = 0.02 # Bottom friction coefficient for smooth earth
        self.rho_air = 1.225
        self.rho_water = 1025.0
        self.cd = 0.0012

        omega = 7.2921e-5 # This is Earth's angular velocity (rad/s)
        
        lat_degrees = torch.linspace(LAT_MIN, LAT_MAX, GRID_RES) # Gen the lat gradient matching the 18.0N to 22.0N bounding box from mesh_interpolator 
        lat_radians = lat_degrees*(math.pi/180.0)
        # Coriolis param tensor (f = 2*omega*sin(lat)). Reshape to (1, 1, Height(Y), 1) to broadcast correctly across (Batch, Channels, Y, X)
        f_tensor = 2*omega*torch.sin(lat_radians).view(1, 1, GRID_RES, 1)

        self.register_buffer('coriolis_f', f_tensor) # Reg buffer ensures this non-trainable tensor is pushed to the GPU with the model.

    def compute_pde_loss(self, pred_scaled, prev_scaled, forces_scaled):
        # Unscale to physical units (m, m/s)
        zeta_pred = pred_scaled[:, 0:1, :, :]*MAX_SURGE
        u_pred = pred_scaled[:, 1:2, :, :]*MAX_VEL
        v_pred = pred_scaled[:, 2:3, :, :]*MAX_VEL

        zeta_prev = prev_scaled[:, 0:1, :, :]*MAX_SURGE
        u_prev = prev_scaled[:, 1:2, :, :]*MAX_VEL
        v_prev = prev_scaled[:, 2:3, :, :]*MAX_VEL

        # Bathymetry/topography: -ve in ocean (e.g. -2000m), +ve on land (e.g. +10m)
        topo = forces_scaled[:, 0:1, :, :]*MAX_TOPO # -ve in ocean, +ve on land
        wind_u = forces_scaled[:, 1:2, :, :]*MAX_WIND
        wind_v = forces_scaled[:, 2:3, :, :]*MAX_WIND

        # Total H2O col depth H = zeta - topo (e.g. 0m - (-1000m) = 1000m ocean depth)
        H_prev = zeta_prev - topo
        # Wet/Dry Mask. Creates a bin tensor (1.0 for H2O, 0.0 for land). Must use .detach() so PyTorch does not attempt to calc gradients through the bool threshold.
        wet_mask = (H_prev > self.dry_tolerance).float().detach()

        H_pred = torch.clamp(zeta_pred - topo, min = 0.1)

        # Mass conservation (Continuity Eq) [d(zeta)/dt + d(H*u)/dx + d(H*v)/dy = 0]
        hu = H_pred*u_pred
        hv = H_pred*v_pred

        dzeta_dt = (zeta_pred - zeta_prev)/self.dt

        d_hu_dx = torch.gradient(hu, spacing = (self.dx,), dim = 3)[0] # dim = 3 is X (Width)
        d_hv_dy = torch.gradient(hv, spacing = (self.dy,), dim = 2)[0] # dim = 2 is Y (Height)

        # Normalize continuity for local depth H_pred so 2000m deep-ocean pixels don't weight 4000000x heavier than 1m coastal pixels.
        mass_residual = (dzeta_dt + d_hu_dx + d_hv_dy)/H_pred

        # Spatial derivatives for momentum eqs (free-surface gradient uses zeta_pred, not total depth)
        du_dx = torch.gradient(u_pred, spacing = (self.dx,), dim = 3)[0]
        du_dy = torch.gradient(u_pred, spacing = (self.dy,), dim = 2)[0]
        dv_dx = torch.gradient(v_pred, spacing = (self.dx,), dim = 3)[0]
        dv_dy = torch.gradient(v_pred, spacing = (self.dy,), dim = 2)[0]        
        dzeta_dx = torch.gradient(zeta_pred, spacing = (self.dx,), dim = 3)[0]
        dzeta_dy = torch.gradient(zeta_pred, spacing = (self.dy,), dim = 2)[0]

        # Momentum conservation (x-axis). Accounts for gravity, wind shear & bottom friction forces.
        # Bottom friction force
        vel_mag = torch.sqrt(u_pred**2 + v_pred**2 + 1e-6)
        wind_mag = torch.sqrt(wind_u**2 + wind_v**2 + 1e-6)

        H_4_3 = (H_pred**(4.0/3.0)) + 1e-4
        denom_wind = (self.rho_water*H_pred) + 1.0

        bottom_friction_x = (self.gravity*(self.manning_n**2)*u_pred*vel_mag)/H_4_3
        bottom_friction_y = (self.gravity*(self.manning_n**2)*v_pred*vel_mag)/H_4_3

        # Wind stress force
        wind_stress_x = (self.rho_air*self.cd*wind_u*wind_mag)/denom_wind
        wind_stress_y = (self.rho_air*self.cd*wind_v*wind_mag)/denom_wind

        du_dt = (u_pred - u_prev)/self.dt
        dv_dt = (v_pred - v_prev)/self.dt
        # It uses -f (Coriolis deflection pushing the flow rightward)
        mom_x_residual = du_dt + (u_pred*du_dx) + (v_pred*du_dy) + (self.gravity*dzeta_dx) - (self.coriolis_f*v_pred) - wind_stress_x + bottom_friction_x
        # It uses +f (Coriolis deflection pushing the flow rightward)
        mom_y_residual = dv_dt + (u_pred*dv_dx) + (v_pred*dv_dy) + (self.gravity*dzeta_dy) + (self.coriolis_f*u_pred) - wind_stress_y + bottom_friction_y

        masked_mass = mass_residual*wet_mask
        
        masked_mom_x = mom_x_residual*wet_mask
        masked_mom_y = mom_y_residual*wet_mask

        return torch.mean(masked_mass**2) + torch.mean(masked_mom_x**2) + torch.mean(masked_mom_y**2)

    def compute_data_loss(self, final_pred_scaled, target_scaled):

        return self.mse(final_pred_scaled, target_scaled)


def train_checkpointed_surrogate():
    # Init infra
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f"Compute engine: {device}")

    sub_steps_per_hour = 12
    dt_seconds = 3600.0/sub_steps_per_hour
    pde_weight = 1e-2
    
    model = RecurrentSurgeUNet(in_channels = 6, out_channels = 3).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    # dt = 40 secs to satisfy CFL (Courant-Friedrichs-Lewy) limit.
    criterion = AutoregressivePINNLoss(dx = 390.625, dy = 390.625, dt = dt_seconds, dry_tolerance = 0.05).to(device)

    scale_tensor_forces = torch.tensor([MAX_TOPO, MAX_WIND, MAX_WIND], device = device).view(1, 3, 1, 1)
    fluid_scale_tensor = torch.tensor([MAX_SURGE, MAX_VEL, MAX_VEL], device = device).view(1, 3, 1, 1)

    try:
        dataset = HydroDataset(TRAINING_TENSORS_DIR)
        dataloader = DataLoader(dataset, batch_size = 2, shuffle = True, pin_memory = (device.type == 'cuda'))
    except FileNotFoundError as e:
        logging.critical(e)

        return

    epochs = 5

    for epoch in range(epochs):
        model.train()

        running_total_loss = running_data_loss = 0.0

        for batch_idx, (static_bathy, wind_t0, wind_t1, fluid_t0, target_h_t1) in enumerate(dataloader):
            # Move the matrices to CPU/GPU
            static_bathy = static_bathy.to(device, non_blocking = True)
            wind_t0 = wind_t0.to(device, non_blocking = True)
            wind_t1 = wind_t1.to(device, non_blocking = True)
            fluid_t0 = fluid_t0.to(device, non_blocking = True)
            target_h_t1 = target_h_t1.to(device, non_blocking = True)

            # 0 the gradients from the prev step
            optimizer.zero_grad()

            forcing_seq = []

            for step in range(1, sub_steps_per_hour + 1):
                alpha = step/sub_steps_per_hour
                curr_wind = (1.0 - alpha)*wind_t0 + alpha*wind_t1
                curr_forces = torch.cat([static_bathy, curr_wind], dim = 1)

                # Scale immediately before seq gen
                forcing_seq.append((curr_forces/scale_tensor_forces).unsqueeze(1))

            # Seq shape: (Batch, 90, 3, 256, 256)
            forcing_seq = torch.cat(forcing_seq, dim = 1)
            
            fluid_t0_scaled = fluid_t0/fluid_scale_tensor
            pred_fluid_seq_scaled = model(forcing_seq, fluid_t0_scaled)

            total_pde_loss = 0.0

            curr_fluid_scaled = fluid_t0_scaled

            for t in range(sub_steps_per_hour):
                next_fluid_scaled = pred_fluid_seq_scaled[:, t, :, :, :]

                curr_forces_scaled = forcing_seq[:, t, :, :, :]

                total_pde_loss += criterion.compute_pde_loss(next_fluid_scaled, curr_fluid_scaled, curr_forces_scaled)

                curr_fluid_scaled = next_fluid_scaled

            # Avg PDE loss per micro-step so it stays invariant to sub_steps_per_hour.
            mean_pde_loss = total_pde_loss/sub_steps_per_hour

            final_fluid_state_scaled = pred_fluid_seq_scaled[:, -1, :, :, :]
            
            target_scaled = target_h_t1/fluid_scale_tensor

            data_loss = criterion.compute_data_loss(final_fluid_state_scaled, target_scaled)

            total_loss = data_loss + (pde_weight*mean_pde_loss)

            total_loss.backward() # Backpropagate

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

            optimizer.step()

            running_total_loss += total_loss.item()
            running_data_loss += data_loss.item()

            logging.info(f"Batch [{batch_idx + 1}/{len(dataloader)}] | Total Loss: {total_loss.item():.6f} | Data MSE: {data_loss.item():.6f} | PDE Mean: {mean_pde_loss.item():.6f}")

        epoch_loss = running_total_loss/len(dataloader)
        
        epoch_data_mse = running_data_loss/len(dataloader)

        logging.info(f"*** Epoch {epoch + 1}/{epochs} Complete | Avg Total Loss: {epoch_loss:.6f} | Avg Data MSE: {epoch_data_mse:.6f} ***")

        torch.save(model.state_dict(), CHECKPOINTS_DIR/f'surge_unet_weights_epoch_{epoch + 1}.pt')


if __name__ == '__main__':
    train_checkpointed_surrogate()