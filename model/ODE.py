import torch
import torch.nn as nn

def apply_neumann_boundary_conditions(X):
    # Apply Neumann boundary conditions (zero gradient)
    X[:, 0, :] = X[:, 1, :]       # Zero gradient at the top boundary
    X[:, -1, :] = X[:, -2, :]     # Zero gradient at the bottom boundary
    X[:, :, 0] = X[:, :, 1]       # Zero gradient at the left boundary
    X[:, :, -1] = X[:, :, -2]     # Zero gradient at the right boundary
    return X

def compute_advection(X, v_x, v_y, dx, dy):
    gradient = torch.zeros_like(X)
    # Compute gradients for the inner region
    gradient[:, 1:-1, 1:-1] = (
        v_x[:, 1:-1, 1:-1] * (X[:, 1:-1, 2:] - X[:, 1:-1, :-2]) / (2 * dx) +
        v_y[:, 1:-1, 1:-1] * (X[:, 2:, 1:-1] - X[:, :-2, 1:-1]) / (2 * dy)
    )
    return apply_neumann_boundary_conditions(gradient)  # Return advection term with Neumann BCs

def compute_laplacian(X, dx, dy):
    laplacian_X = torch.zeros_like(X)
    # Compute finite difference Laplacian for the inner region
    laplacian_X[:, 1:-1, 1:-1] = (
        (X[:, 2:, 1:-1] - 2 * X[:, 1:-1, 1:-1] + X[:, :-2, 1:-1]) / (dx ** 2) +
        (X[:, 1:-1, 2:] - 2 * X[:, 1:-1, 1:-1] + X[:, 1:-1, :-2]) / (dy ** 2)
    )
    return apply_neumann_boundary_conditions(laplacian_X)  # Return Laplacian with Neumann BCs
