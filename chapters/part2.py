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
# ```{note}
# This notebook is incomplete and will be completed as an exercise in class.
# ```
#
# ## Basic algorithm
#
# Recall that using the notions in {doc}`part1` and during class we derived the
# following expressions for the entries of the finite element stiffness matrix
# $\mathbf{K}$ and load vector $\mathbf{f}$
#
# $$
# K_{ij} &:= \sum_{K \in \mathcal{T}_h} \int_{K} \nabla \phi_i \cdot \nabla \phi_j \; \mathrm{d}x, \\
# f_{j} &:= \sum_{K \in \mathcal{T}_h} \int_{K} f \phi_j \; \mathrm{d}x.
# $$
#
# Instead of calculating each entry of $K_{ij}$ we discussed that the most
# straightforward way to *assemble* the stiffness matrix is to:
# 1. Loop over the global cells $K$ of the mesh $\mathcal{T}_h$.
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
# For a general cell $K$ derive an explicit symbolic expression for the cell
# local contribution $\mathbf{K}_K \in \mathbb{R}^{2 \times 2}$ in terms of
# $h$. Use the local-to-global mapping approach shown in class.
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
import scipy.sparse
import matplotlib.pyplot as plt
from typing import NamedTuple, Callable


def cell_stiffness(a: float, b: float) -> npt.NDArray[np.float64]:
    """Calculate the stiffness matrix contribution for a cell with vertices a and b.

    Args:
        a: Position of first vertex.
        b: Position of second vertex.

    Return:
        The stiffness matrix contribution for the cell.
    """
    assert b > a
    raise NotImplementedError


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
# \end{bmatrix},
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
# \end{bmatrix}.
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
    raise NotImplementedError

    # return Mesh(geometry=geometry, topology=topology)


num_cells = 4
c = 1.0 * np.pi
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
# The *degree of freedom map* `dofmap` will be an array contain information
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
# We package up the `mesh`, `dofmap` and also make a record of the `size` of
# the function space in a new `NamedTuple` object `FunctionSpace`.
#
# ```{note}
# This explicit separation between mesh topology, geometry and solution degrees
# of freedom is not necessary for this simple problem, but does mimic closer
# how a real finite element code such as [FEniCSx](https://fenicsproject.org)
# or [Firedrake](https://firedrakeproject.org) is organised.
# ```
# %%
FunctionSpace = NamedTuple(
    "FunctionSpace",
    (
        ("mesh", Mesh),
        ("dofmap", npt.NDArray[np.int32]),
        ("size", npt.NDArray[np.int64]),
    ),
)
fs = FunctionSpace(mesh=mesh, dofmap=mesh.topology.copy(), size=mesh.geometry.shape[0])

# %% [markdown]
# For example, we can get the global degree of freedom for global cell 1
# associated with local degree of freedom 0 using
# %%
print(fs.dofmap[1, 0])

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


def assemble_stiffness(
    fs: FunctionSpace, cell_stiffness_fn: Callable
) -> scipy.sparse.lil_matrix:
    """Write a documentation string"""
    K = scipy.sparse.lil_matrix((fs.size, fs.size))

    # Loop over the cells of the mesh
    for cell in range(0, fs.mesh.topology.shape[0]):
        # Remove this pass statement when you begin coding here
        pass
        # Step 1: Calculate the stiffness matrix on this cell

        # Step 2: Extract the local to global degree of freedom mapping for the
        # cell

        # Step 3: Scatter to the sparse matrix
        # Hint: K[np.ix_(dofs, dofs)] will select the right elements in K

    return K


K = assemble_stiffness(fs, cell_stiffness)
if num_cells == 4:
    print(K)
    K_dense = K.todense()
    assert np.all(np.isclose(K_dense, K_dense.T))

# %% [markdown]
# ## Assembly of $\mathbf{f}$
#
# We will now assemble the load vector $\mathbf{f}$.
#
# If we take $f(x)$ as a general function then we cannot usually symbolically
# calculate the cell local contribution $\mathbf{f}_{K}$ to the load vector
# $\mathbf{f}$. A standard approach is to use quadrature, which allows the
# approximation of an integral on the unit interval
#
# $$
# \int_0^1 g(\hat{x}) \approx \sum_{i = 0}^{n - 1} w_i g(\hat{x}^q_i),
# $$
#
# where the $w_i$ are known as the quadrature weights and the $\hat{x}_i^q$ as
# quadrature points. We will use a two-point rule $n = 2$ on $[0, 1]$ with
# points $\hat{x}^q_0 = \frac{1}{2} - \frac{1}{2\sqrt{3}}$ and $\hat{x}^q_1 =
# \frac{1}{2} + \frac{1}{2\sqrt{3}}$ associated with weights $w_1 = w_2 =
# \frac{1}{2}$.
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
quadrature_weights = 0.5 * np.ones(2, dtype=np.float64)

# %% [markdown]
# ### Exercise 6
#
# Using the local-to-global approach and a quadrature rule with $n$ points
# derive the cell local contribution $\mathbf{f}_K$ to the load vector
# $\mathbf{f}$.
#
# %% [markdown]
# Write your answer using Markdown here.
#
# *Answer*


# %% [markdown]
# ### Exercise 7
#
# Complete the functions `phi_hat` and `cell_load`. The latter returns the load
# vector for a cell with vertices $a$ and $b$ with $b > a$.
#
# %%
def phi_hat(x_hat: float) -> npt.NDArray[np.float64]:
    """
    Calculate the local P1 finite element basis functions.

    Args:
        x_hat: Position in local element coordinate system.

    Returns:
        An array containing the evaluation of the local basis functions.
    """
    raise NotImplementedError


def cell_load(a: float, b: float) -> npt.NDArray[np.float64]:
    """Write a documentation string"""
    f_cell = np.zeros(2, dtype=np.float64)

    for point, weight in zip(quadrature_points, quadrature_weights):
        # Remove this pass when you begin coding your solution
        pass

    return f_cell


# %% [markdown]
# So for the fourth cell we can assemble the local stiffness matrix
# contribution
# %%
print(cell_load(*mesh.geometry[mesh.topology[1]]))

# %% [markdown]
# ### Exercise 8
#
# Complete the function `assemble_load`.
# %%


def assemble_load(fs: FunctionSpace, cell_load_fn: Callable) -> npt.NDArray[np.float64]:
    f = np.zeros(fs.size)

    # Loop over the cells of the mesh
    for cell in range(0, mesh.topology.shape[0]):
        # Remove this pass statement when you begin coding here
        pass
        # Step 1: Calculate the stiffness matrix on this cell

        # Step 2: Extract the local to global degree of freedom mapping for the
        # cell

        # Step 3: Scatter to the vector

    return f


f = assemble_load(fs, cell_load)
if num_cells == 4:
    print(f)

# %% [markdown]
# ## Applying Dirichlet conditions
#
# We have assembled the matrix on the space $V_h$, but recall in the
# specification of the weak form of the problem we used the space
# $\mathring{V}_h$, that contains only the basis functions associated with the
# interior degrees of freedom. We can transfer the problem to the space
# $\mathring{V}_h$ by modifying the linear system in place. For the degrees of
# freedom associated with vertices on the boundary (here, always $0$ and $N -
# 1$) we shall modify the stiffness matrix $\mathbf{K}$ by
#
# 1. placing $0$ on the corresponding rows,
# 2. placing $0$ on the corresponding columns,
# 3. inserting $1$ on the corresponding diagonals,
#
# For the force vector $\mathbf{f}$ we place place $0$ on the corresponding
# rows.
#
# %%


def apply_boundary_conditions(
    dofs: npt.NDArray[np.int32], K: scipy.sparse.lil_matrix, f: npt.NDArray[np.float64]
):
    """Apply boundary conditions on dofs to the linear system (K, f).

    Note: K and f are modified in-place.

    Args:
        dofs: the degrees of freedom to apply boundary conditions to.
        K: the stiffness matrix.
        f: the force vector.
    """
    num_dofs = K.shape[0]

    for dof in dofs:
        # Zero the row
        K.rows[dof] = []
        K.data[dof] = []

        # Loop over all the rows
        for row in range(num_dofs):
            # Does this row have an entry on the column associated with dof?
            if dof in K.rows[row]:
                idx = K.rows[row].index(dof)
                K.rows[row].pop(idx)
                K.data[row].pop(idx)

        K[dof, dof] = 1.0
        f[dof] = 0.0


boundary_dofs = np.array([0, mesh.geometry.shape[0] - 1], dtype=np.int32)
apply_boundary_conditions(boundary_dofs, K, f)

if num_cells == 4:
    print(K)
    print(f)

# %% [markdown]
# ## Solving
# We can now solve the system $\mathbf{K} \mathbf{u}_h = \mathbf{f}$ to find
# the vector of previously unknown coefficients $\mathbf{u}_h$ of the finite
# element solution $u_h$.
# %%
K_csr = K.tocsr()
u = scipy.sparse.linalg.spsolve(K_csr, f)

plt.plot(mesh.geometry, u, "o-")
plt.plot(mesh.geometry, np.sin(c * mesh.geometry), "-")
plt.xlabel(r"$x$")
plt.ylabel(r"$u$")
plt.show()

# %% [markdown]
# ## Further exercises
#
# ### Exercise 9
#
# Make a new function `cell_stiffness_quadrature` to compute the stiffness
# matrix using a quadrature approach. Comment on the necessary order of the
# quadrature rule to exactly compute the integrand. Pass this up to your
# assembler and re-run, making sure you get the same result.
#
# %%

# %% [markdown]
# ### Exercise 10
#
# Write a function `solve` which takes the `num_cells` (number of cells) as an
# argument. Return the solution from the function. Plot a sequence of solutions
# on increasingly fine meshes.
#
# %%

# %% [markdown]
# ### Exercise 11
#
# Modify the `solve` function to additionally return the error between the
# interpolant of the solution and the solution in the natural norm.
#
# $$
# e_h = \lVert I_h u - u_h \rVert^2_{H^1_0} = (\mathbf{u} - \mathbf{u}_h)^T \mathbf{K} (\mathbf{u} - \mathbf{u}_h)
# $$
#
# where $\mathbf{u}$ is the vector of coefficients of the interpolant of the
# exact solution $u$.
#
# On a sequence of refined meshes compute the associated error. Plot the error
# $e_h$ against $h$ on a log-log plot, and calculate the slope. What do you
# observe?
# %%
