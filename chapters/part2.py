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
# Instead of calculating each entry of $K_{ij}$ we discussed that the most
# straightforward way to *assemble* the stiffness matrix is to:
# 1. Loop over the global cells $K_k = (x_k, x_{k+1})$ of the mesh $\mathcal{T}_h$.
# 2. Calculate the cell local contribution $\mathbf{K}_K \in \mathbb{R}^{2
# \times 2}$.
# 3. Determine which pair of finite element basis functions are active on the
#    cell.
# 4. *Assemble* (add) the cell local contribution to the stiffness matrix at
#    the location of the active basis functions.
#
# The load vector assembly follows similarly.
#
# ### Exercise 1
#
# For a general cell $K$ derive an explicit expression for the cell local
# contribution $\mathbf{K}_K \in \mathbb{R}^{2 \times 2}$ in terms of $h$ to the
# stiffness matrix $\mathbf{K}$. Use the local-to-global mapping approach shown
# in class.
#
# *Answer*
#
# Write your answer using Markdown here.
#
# ### Exercise 2
#
# Complete the function `cell_stiffness` which returns the stiffness matrix for
# a cell with vertices $a$ and $b$ with $b > a$.
#
# %%
import numpy as np
import numpy.typing as npt
from typing import NamedTuple, Callable


def cell_stiffness(a: float, b: float) -> npt.NDArray[np.float64]:
    """Calculate the local stiffness matrix for a cell with vertices a and b."""
    assert b > a
    inv_h = 1.0 / (b - a)
    return inv_h * np.array([[1.0, -1.0], [-1.0, 1.0]])


# %% [markdown]
#
# ## Mesh
#
# The mesh will be composed of two data structures:
# 1. the *geometry* which will contain the positions of the vertices of the
# mesh. The index on the first dimension (rows) is the mesh vertex number. The
# index on the second dimension (columns) of the geometry is the related to the
# coordinate index.
# 2. the *topology*, which will contain the cell-to-vertex connectivity. The
# index on the first dimension (rows) is the cell number. The index on the
# second dimension (columns) are the indices of the vertices of the cell.
#
# Consider the case when we want to create a mesh with $N = 4$ cells. Here the
# geometry will be a two-dimensional numpy array containing
#
# $$
# \text{geometry} =
# \begin{bmatrix}
# 0.0 \\
# 0.25 \\
# 0.5 \\
# 0.75 \\
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
# ```{note}
# To avoid object-oriented programming, but to still keep the code tidy, we
# will use Python's namedtuple feature which allows tuples to have *named
# fields*.
# ```
#
# %%

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
    geometry = np.linspace(0.0, 1.0, num_cells + 1).reshape(-1, 1)

    left = np.arange(num_cells, dtype=np.int32)
    right = left + 1
    topology = np.stack((left, right), axis=1)

    return Mesh(geometry=geometry, topology=topology)


num_cells = 4
mesh = create_unit_interval_mesh(num_cells)

if num_cells == 4:
    print(f"Mesh: {mesh}")
    assert np.all(np.isclose(mesh.geometry, [[0.0], [0.25], [0.5], [0.75], [1.0]]))
    assert np.all(np.isclose(mesh.topology, [[0, 1], [1, 2], [2, 3], [3, 4]]))

# %% [markdown]
# For example, for global cell 1 we can get the connected global vertices using
# %%
print(mesh.topology[1])

# %% [markdown]
# and the position of the vertices using
# %%
print(mesh.geometry[mesh.topology[1]])

# %% [markdown]
# and assemble the local stiffness matrix contribution. The `*` unpacks the
# entries of the two element array into two separate variables `a` and `b`.
# %%
print(cell_stiffness(*mesh.geometry[mesh.topology[1]]))

# %% [markdown]
#
# ## Degree of freedom map
#
# The *degree of freedom map* `dof_map` will be an array contain information
# about the connection between the local basis functions (degrees of freedom)
# on the local cell and the global basis functions (degrees of freedom). On the
# first dimension (rows) the index is the mesh cell number. On the second
# dimension (columns) the index is the local degree of freedom number.
#
# For the $P_1$ finite element space $V_h$ each vertex is assigned one global
# degree of freedom. Each cell then has two global degrees of freedom. To
# ensure $C^0$ continuity, vertices shared by a cell must share global degrees
# of freedom.
#
# This is infact nothing more than the existing `mesh.topology` array! We
# simply make a copy and continue, but we explicitly use the right array, in
# the right place.
#
# ```{note}
# This explicit separation between mesh topology, geometry and solution degrees
# of freedom is not necessary for this simple problem, but does mimic closer
# how a real finite element code such as [FEniCSx](https://fenicsproject.org)
# or [Firedrake](https://firedrakeproject.org) is organised.
# ```
# %%
dof_map = mesh.topology.copy()

# %% [markdown]
# For example, we can get the global degree of freedom for global cell 1
# associated with local degree of freedom 0 using
# %%
print(dof_map[1, 0])

# %% [markdown]
# ## Assembly of $\mathbf{K}$
#
# We now have everything we need to assemble $\mathbf{K}$. Because of the local
# construction of the basis functions $\mathbf{K}$ will be sparse, i.e. the
# majority of its entries will be zero.
#
# The package `scipy.sparse` contains various efficient sparse data structures.
# For simplicity I recommend building the sparse matrix in the [List of
# Lists](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html)
# (LOL) format and then converting to the [Compressed Sparse
# Row](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html)
# (CSR) format for efficient solution.
#
# ```{note} The
# [COOrdinate](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.html#scipy.sparse.coo_array)
# format is more efficient for construction than LOL. Constructing directly in
# CSR is possible and is also the most efficient, but is significantly more complex.
# ```
#
# ### Exercise 4
#
# Complete the function `assemble_stiffness`.
# %%
import scipy.sparse


def assemble_stiffness(
    mesh: Mesh, dof_map: npt.NDArray[np.int32], cell_stiffness_fn: Callable
) -> scipy.sparse.lil_matrix:
    num_dofs = mesh.geometry.shape[0]

    K = scipy.sparse.lil_matrix((num_dofs, num_dofs))

    # Loop over the cells of the mesh
    for cell in range(0, num_cells):
        # Remove this pass statement when you begin coding here
        # pass
        # Step 1: Calculate the stiffness matrix on this cell
        K_cell = cell_stiffness_fn(*mesh.geometry[mesh.topology[cell]])

        # Step 2: Extract the local to global degree of freedom mapping for the
        # cell
        dofs = dof_map[cell]

        # Step 3: Scatter to the sparse matrix
        # Hint: K[np.ix_(dofs, dofs)]
        K[np.ix_(dofs, dofs)] = K_cell

    return K.tocsr()


K = assemble_stiffness(mesh, dof_map, cell_stiffness)
if num_cells == 4:
    K_dense = K.todense()
    print(K_dense)
    assert np.all(np.isclose(K_dense, K_dense.T))

# %% [markdown]
# ## Assembly of $\mathbf{f}$
#
# We will now assemble the load vector $\mathbf{f}$.
#
# If we take $f(x) = \sin(x)$ then we cannot straightforwardly calculate the
# cell local contribution $\mathbf{f}_{K}$ to the load vector $\mathbf{f}$
# symbolically. A standard approach is to use quadrature, which allows the
# approximation of an integral on the unit interval
#
# $$
# \int_0^1 g(\hat{x}) \approx \sum_{i = 0}^{n - 1} w_i g(\hat{x}^q_i)
# $$
#
# where the $w_i$ are known as the quadrature weights and the $\hat{x}_i^q$ as
# quadrature points. We will use a two-point rule $n = 2$ with points
# $\hat{x}^q = \frac{1}{2} \pm \frac{1}{2\sqrt{3}}$ and weights $w_1 = w_2 =
# 1$.
#
# ```{note}
# In a proper finite element code *both* the element contributions of the
# stiffness matrix and load vector are calculated numerically through
# [quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature).
# ```
#
# This rule can be defined in code as
# %%
quadrature_points = np.array(
    [
        (1.0 / 2.0) - (1.0 / (2.0 * np.sqrt(3.0))),
        (1.0 / 2.0) + (1.0 / (2.0 * np.sqrt(3.0))),
    ]
)
quadrature_weights = np.ones(2, dtype=np.float64)

# %% [markdown]
# ### Exercise 6
#
#

# %% [markdown]
# Write your answer using Markdown here.
#
# *Answer*

# %% [markdown]
# ### Exercise 7
#
# Complete the function `cell_load` which returns the load vector for a cell
# with vertices $a$ and $b$ with $b > a$.
