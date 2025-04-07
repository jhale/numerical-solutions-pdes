# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # A first finite element code
#
# In this notebook we will develop a one-dimensional finite Galerkin finite
# element code.
#
# ## Basic algorithm
#
# Recall that using the notions in {doc}`part1` and during class we derived the
# following expressions for the entries of the finite element stiffness matrix
# $\mathbf{K}$ and load vector $\mathbf{f}$
#
# $$
# K_{ij} &:= \sum_{k = 0}^{N - 1} \int_{K_k} \nabla \phi_i \cdot \nabla \phi_j \; \mathrm{d}x, \\
# f_{j} &:= \sum_{k = 0}^{N - 1} \int_{K_k} f \phi_j \; \mathrm{d}x.
# $$
#
# It was proposed that the most effective way to construct these linear algebra
# objects would be to:
# 1. loop over the cells $K_k = (x_i, x_{i+1})$ of the mesh $\mathcal{T}_h$.
# 2. determine which pair of finite element basis functions are active.
# 3. calculate the cell local contribution.
# 4. *assemble* (add) the cell local contribution to the stiffness matrix.
#
# ### Exercise 1
#
# For a general cell $K_k$ of size $h$ derive an explicit expression for the
# cell local contribution $\hat{\bf{K}} \in \mathbb{R}^{2 \times 2}$ to the
# stiffness matrix $\mathbf{K}$ using the local-to-global mapping approach
# shown in class.
#
# *Answer*
#
# Write your answer using Markdown here.
#
# ### Exercise 2
#
# Code the function `cell_stiffness` which returns the stiffness matrix for a
# cell with vertices $a$ and $b$ with $b > a$.
#
# %%
import numpy as np
import numpy.typing as npt
from typing import NamedTuple


def cell_stiffness(a: float, b: float) -> npt.NDArray[np.float64]:
    """Calculate the local stiffness matrix for a cell with vertices a and b."""
    inv_h = 1.0 / (b - a)
    return inv_h * np.array([[1.0, -1.0], [-1.0, 1.0]])


# $$ [markdown]
#
# ## Mesh
#
# The mesh in the code will be composed of two data structures:
# 1. the *geometry* which will contain the positions of the vertices of the
# mesh.
# 2. the *topology*, which will contain the cell-to-vertex connectivity.
#
# ```{note}
# To avoid object-oriented programming, but to still keep the code tidy, we
# will use Python's namedtuple feature which allows tuples to have *named
# fields*.
# ```
#
# Consider the case when we want to create a mesh with $N = 4$ cells. Here the
# geometry will be a one-dimensional numpy array containing
#
# $$
# \text{geometry} =
# \begin{bmatrix}
# 0.0 &
# 0.25 &
# 0.5 &
# 0.75 &
# 1.0
# \end{bmatrix}
# $$
#
# and the topology a two-dimensional numpy array containing
#
# $$
# \text{topology} =
# \begin{bmatrix}
# 0 & 1 \\
# 1 & 2 \\
# 2 & 3 \\
# 3 & 4
# \end{bmatrix}
# $$
#
# ### Exercise 3
#
# Generalise the function `create_unit_interval_mesh` to arbitrary input
# `num_cells`.
#
# %%

# Define the mesh type
Mesh = NamedTuple(
    "Mesh", (("geometry", npt.NDArray[np.float64]), ("topology", npt.NDArray[np.int32]))
)


def create_unit_interval_mesh(num_cells: int) -> Mesh:
    """
    Generate a 1D uniform mesh on the unit interval.

    Args:
        num_cells: Number of cells.

    Returns:
        A 1D uniform mesh on the unit interval.
    """
    geometry = np.linspace(0.0, 1.0, num_cells + 1)

    left = np.arange(num_cells, dtype=np.int32)
    right = left + 1
    topology = np.stack((left, right), axis=1)

    return Mesh(geometry=geometry, topology=topology)


num_cells = 4
mesh = create_unit_interval_mesh(num_cells)

if num_cells == 4:
    print(f"Mesh: {mesh}")
    assert np.all(np.isclose(mesh.geometry, [0.0, 0.25, 0.5, 0.75, 1.0]))
    assert np.all(np.isclose(mesh.topology, [[0, 1], [1, 2], [2, 3], [3, 4]]))

print(cell_stiffness(*mesh.geometry[mesh.topology[1]]))
