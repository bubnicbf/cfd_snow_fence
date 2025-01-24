import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx, ny = 195, 140  # Grid points in x and y for 195' x 140' area
lx, ly = 195.0, 140.0  # Length of the domain in feet
dx, dy = lx / (nx - 1), ly / (ny - 1)  # Grid spacing
nt = 500  # Number of time steps
dt = 0.001  # Time step size
nu = 0.1  # Kinematic viscosity
rho = 1.0  # Density

# Initialize variables
u = np.zeros((ny, nx))  # Velocity in x direction
v = np.zeros((ny, nx))  # Velocity in y direction
p = np.zeros((ny, nx))  # Pressure field
b = np.zeros((ny, nx))  # Source term for pressure Poisson equation

# Boundary conditions
def apply_boundary_conditions(u, v, p):
    u[0, :] = 0  # Bottom wall
    u[-1, :] = 0  # Top wall
    u[:, 0] = 0  # Left wall
    u[:, -1] = 0  # Right wall

    v[0, :] = 0  # Bottom wall
    v[-1, :] = 0  # Top wall
    v[:, 0] = 0  # Left wall
    v[:, -1] = 0  # Right wall

    u[-1, 1:-1] = 1  # Lid-driven top boundary

# Pressure Poisson equation
def pressure_poisson(p, b, dx, dy):
    for _ in range(50):  # Iterative solver
        pn = p.copy()
        p[1:-1, 1:-1] = (
            (dy**2 * (pn[1:-1, 2:] + pn[1:-1, :-2]) +
             dx**2 * (pn[2:, 1:-1] + pn[:-2, 1:-1]) -
             b[1:-1, 1:-1] * dx**2 * dy**2) /
            (2 * (dx**2 + dy**2))
        )
        # Boundary conditions for pressure
        p[:, 0] = p[:, 1]  # dp/dx = 0 at left wall
        p[:, -1] = p[:, -2]  # dp/dx = 0 at right wall
        p[0, :] = p[1, :]  # dp/dy = 0 at bottom wall
        p[-1, :] = 0  # p = 0 at top wall
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

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()

    apply_boundary_conditions(u, v, p)

    build_b(b, un, vn, dx, dy)
    p = pressure_poisson(p, b, dx, dy)

    # Update velocity
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx *
                     (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                     vn[1:-1, 1:-1] * dt / dy *
                     (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
                     dt / (2 * rho * dx) *
                     (p[1:-1, 2:] - p[1:-1, :-2]) +
                     nu * (dt / dx**2 *
                           (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] +
                            un[1:-1, :-2]) +
                           dt / dy**2 *
                           (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] +
                            un[:-2, 1:-1])))

    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx *
                     (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                     vn[1:-1, 1:-1] * dt / dy *
                     (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
                     dt / (2 * rho * dy) *
                     (p[2:, 1:-1] - p[:-2, 1:-1]) +
                     nu * (dt / dx**2 *
                           (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] +
                            vn[1:-1, :-2]) +
                           dt / dy**2 *
                           (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] +
                            vn[:-2, 1:-1])))

# Plot the results
plt.figure(figsize=(11, 7))
plt.quiver(np.linspace(0, lx, nx), np.linspace(0, ly, ny), u, v, scale=20, pivot="midle")
plt.title("Velocity Field")
plt.xlabel("X (feet)")
plt.ylabel("Y (feet)")
plt.show()
