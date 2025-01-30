import numpy as np
import pandas as pd
import os

# Parameters
nx, ny = 60, 42  # Grid points in x and y for 60 m x 42 m area
lx, ly = 60.0, 42.0  # Length of the domain in meters
dx, dy = lx / (nx - 1), ly / (ny - 1)  # Grid spacing
nt = 500  # Number of time steps
dt = 0.001  # Time step size
nu = 1.46e-5  # Kinematic viscosity of air at 1 atm and freezing temperatures (m^2/s)
rho = 1.275  # Density of air at 1 atm and freezing temperatures (kg/m^3)

# Load wind data from CSV
input_file = "data/meas/202501_wind_data.csv"
output_dir = "data/ns_output/"
os.makedirs(output_dir, exist_ok=True)

df_wind = pd.read_csv(input_file, parse_dates=["timestamp"])

# Initialize variables
u_grid = np.zeros((ny, nx))  # Velocity in x direction
v_grid = np.zeros((ny, nx))  # Velocity in y direction
p = np.zeros((ny, nx))  # Pressure field
b = np.zeros((ny, nx))  # Source term for pressure Poisson equation

# Apply boundary conditions
def apply_boundary_conditions(u, v):
    u[0, :] = 0  # Ground (bottom boundary)
    v[0, :] = 0
    v[-1, :] = v[-2, :]  # Free-slip at top
    u[-1, :] = u[-2, :]
    u[:, -1] = u[:, -2]  # Outflow on the right
    v[:, -1] = v[:, -2]

# Pressure Poisson equation
def pressure_poisson(p, b, dx, dy):
    for _ in range(50):
        pn = p.copy()
        p[1:-1, 1:-1] = (
            (dy**2 * (pn[1:-1, 2:] + pn[1:-1, :-2]) +
             dx**2 * (pn[2:, 1:-1] + pn[:-2, 1:-1]) -
             b[1:-1, 1:-1] * dx**2 * dy**2) /
            (2 * (dx**2 + dy**2))
        )
    return p

# Build the source term
def build_b(b, u, v, dx, dy):
    b[1:-1, 1:-1] = (
        rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
                         (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) -
               ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx))**2 -
               2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy) *
                    (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)) -
               ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy))**2)
    )

# Process wind data
grouped = df_wind.groupby("timestamp")
for timestamp, group in grouped:
    u_grid.fill(0)
    v_grid.fill(0)
    
    for _, row in group.iterrows():
        x_idx, y_idx = int(row["x"] / 3), int(row["y"] / 3)  # Convert positions to grid indices
        if 0 <= x_idx < nx and 0 <= y_idx < ny:
            u_grid[y_idx, x_idx] = row["wind_x"]
            v_grid[y_idx, x_idx] = row["wind_y"]
    
    apply_boundary_conditions(u_grid, v_grid)
    build_b(b, u_grid, v_grid, dx, dy)
    p = pressure_poisson(p, b, dx, dy)
    
    # Save results to CSV
    output_file = os.path.join(output_dir, f"ns_output_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv")
    output_df = pd.DataFrame({
        "x": group["x"].values,
        "y": group["y"].values,
        "wind_x": group["wind_x"].values,
        "wind_y": group["wind_y"].values,
        "pressure": [p[int(y/3), int(x/3)] for x, y in zip(group["x"], group["y"])]
    })
    output_df.to_csv(output_file, index=False)

print("Navier-Stokes processing complete. Output saved to:", output_dir)
