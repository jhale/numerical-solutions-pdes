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
# $\mathbf{A}$ and load vector $\mathbf{f}$
#
# $$
# A_{ij}^{(k)} &:= \sum_{k = 0}^{N - 1} \int_{K_k}} \nabla \phi_i \cdot \nabla \phi_j \; \mathrm{d}x, \\
# f_{j}^{(k)} &:= \sum_{k = 0}^{N - 1} \int_{K_k}} f \phi_j \; \mathrm{d}x.
# $$
#
# Instead of calculating each entry of $A_{ij}$ we discussed that the most
# straightforward way to *assemble* the stiffness matrix is to:
# 1. Loop over the global cells $K_k = [x_k, x_{k+1}]$ of the mesh $\mathcal{T}_h$.
# 2. Calculate the cell local contribution $\mathbf{A}^{(k)} \in \mathbb{R}^{2
# \times 2}$.
# 3. Determine which pair of finite element basis functions are active on the
#    cell.
# 4. *Assemble* (add/accumulate) the cell local contribution to the stiffness matrix at
#    the location of the active global basis functions.
#
# The load vector assembly follows similarly.
#
# ### Exercise 1
#
# For a general cell $K_k$ derive an explicit expression for the cell local
# contribution $\mathbf{A}^{(k)} \in \mathbb{R}^{2 \times 2}$ in terms of $h$ to the
# stiffness matrix $\mathbf{A}$. Use the local-to-global mapping approach shown
# in class.
#
# *Answer*
#
# Write your answer using Markdown here.
#
# ### Exercise 2
#
# Complete the function `cell_stiffness` which returns the stiffness matrix for
# a cell with vertices $x_k$ and $x_{k+1}$ with $x_{k + 1} > x_{k}$.
#
# %%
import numpy as np
import numpy.typing as npt
import scipy.sparse
import matplotlib.pyplot as plt
from typing import NamedTuple, Callable


def cell_stiffness(x_k: float, x_kp1: float) -> npt.NDArray[np.float64]:
    """Calculate the local stiffness matrix for a cell with vertices x_k and x_{k + 1}."""
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

num_cells = 4
c = 3.0 * np.pi  # Can be set to n\pi with n \in \mathbb{N}^{+}
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
FunctionSpace = NamedTuple(
    "FunctionSpace",
    (
        ("mesh", Mesh),
        ("dofmap", npt.NDArray[np.int32]),
        ("size", int),
    ),
)
fs = FunctionSpace(mesh=mesh, dofmap=mesh.topology.copy(), size=mesh.geometry.shape[0])

# %% [markdown]
# For example, we can get the global degree of freedom for global cell 1
# associated with local degree of freedom 0 using
# %%
print(fs.dofmap[1, 0])

# %% [markdown]
# ## Assembly of $\mathbf{A}$
#
# We now have everything we need to assemble $\mathbf{A}$. Because of the local
# construction of the basis functions $\mathbf{A}$ will be sparse, i.e. the
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
    A = scipy.sparse.lil_matrix((fs.size, fs.size))

    # Loop over the cells of the mesh
    for cell in range(0, fs.mesh.topology.shape[0]):
        pass
        # Step 1: Calculate the stiffness matrix on this cell

        # Step 2: Extract the local to global degree of freedom mapping for
        # the cell

        # Step 3: Scatter to the sparse matrix
        # Hint: A[np.ix_(dofs, dofs)] will select the right elements in A to add into

    return A


A = assemble_stiffness(fs, cell_stiffness)
if num_cells == 4:
    print(A)
    A_dense = A.todense()
    assert np.all(np.isclose(A_dense, A_dense.T))

# %% [markdown]
# ## Assembly of $\mathbf{f}$
#
# We will now assemble the load vector $\mathbf{f}$.
#
# If we take $f(x)$ as a general function then we cannot usually symbolically
# calculate the cell local contribution $\mathbf{f}^{k}$ to the load vector
# $\mathbf{f}$. A standard approach is to use quadrature, which allows the
# approximation of an integral on the unit interval through its weighted point
# evaluation:
#
# $$
# \int_0^1 g(\hat{x}) \approx \sum_{i = 0}^{n - 1} w_i g(\hat{x}^q_i),
# $$
#
# where the $w_i$ are known as the quadrature weights and the $\hat{x}_i^q$ as
# quadrature points. We will use a two-point rule $n = 2$ on $[0, 1]$ with
# points $\hat{x}^q = \frac{1}{2} \pm \frac{1}{2\sqrt{3}}$ and weights $w_1 =
# w_2 = 1/2$.
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
# ### Exercise 5
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
# ### Exercise 6
#
# Complete the function `cell_load` which returns the load vector for a cell $K_k$
# with vertices $x_k$ and $x_{k + 1}$ with $b > a$.
# %%
def phi_hat(x_hat: float) -> npt.NDArray[np.float64]:
    """
    Calculate the local P1 finite element basis functions.

    Args:
        x_hat: Position in local element coordinate system.

    Returns:
        An array containing the evaluation of the local basis functions at
        x_hat.
    """
    return np.array([1.0 - x_hat, x_hat], np.float64)


def cell_load(x_k: float, x_kp1: float) -> npt.NDArray[np.float64]:
    """Calculate the local load vector for a cell with vertices a and b
    using a quadrature rule."""
    f_cell = np.zeros(2, np.float64)

    for point, weight in zip(quadrature_points, quadrature_weights):
        pass

    return f_cell


# %% [markdown]
# So for the second cell we can assemble the local load vector contribution
# %%
print(cell_load(*mesh.geometry[mesh.topology[1]]))

# %% [markdown]
# ### Exercise 7
#
# Complete the function `assemble_load`.
# %%


def assemble_load(fs: FunctionSpace, cell_load_fn: Callable) -> npt.NDArray[np.float64]:
    f = np.zeros(fs.size)

    # Loop over the cells of the mesh
    for cell in range(0, fs.mesh.topology.shape[0]):
        pass
        # Step 1: Calculate the load vector on this cell

        # Step 2: Extract the local to global degree of freedom mapping for
        # the cell

        # Step 3: Assemble into the vector

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
# 1$) we shall modify the stiffness matrix $\mathbf{A}$ by
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
    dofs: npt.NDArray[np.int32], A: scipy.sparse.lil_matrix, f: npt.NDArray[np.float64]
):
    """Apply boundary conditions on dofs to the linear system (A, f).

    Note: A and f are modified in-place.

    Args:
        dofs: the degrees of freedom to apply boundary conditions to.
        A: the stiffness matrix.
        f: the force vector.
    """
    num_dofs = A.shape[0]

    for dof in dofs:
        # Zero the row
        A.rows[dof] = []
        A.data[dof] = []

        # Loop over all the rows
        for row in range(num_dofs):
            # Does this row have an entry on the column associated with dof?
            if dof in A.rows[row]:
                idx = A.rows[row].index(dof)
                A.rows[row].pop(idx)
                A.data[row].pop(idx)

        A[dof, dof] = 1.0
        f[dof] = 0.0


boundary_dofs = np.array([0, mesh.geometry.shape[0] - 1], dtype=np.int32)
apply_boundary_conditions(boundary_dofs, A, f)

if num_cells == 4:
    print(A)
    print(f)

# %% [markdown]
# ## Solving
# We can now solve the system $\mathbf{A} \mathbf{u}_h = \mathbf{f}$ to find
# the vector of previously unknown coefficients $\mathbf{u}_h$ of the finite
# element solution $u_h$.
# %%
A_csr = A.tocsr()
u = scipy.sparse.linalg.spsolve(A_csr, f)

plt.plot(mesh.geometry, u, "o-")
plt.plot(mesh.geometry, np.sin(c * mesh.geometry), "-")
plt.xlabel(r"$x$")
plt.ylabel(r"$u$")
plt.show()


# %% [markdown]
# ## Further exercises
#
# ### Exercise 8
#
# Make a new function `cell_stiffness_quadrature` to compute the stiffness
# matrix using a quadrature approach. Pass this up to your assembler and
# re-run, making sure you get the same result.
#
# %%
def cell_stiffness_quadrature(x_k: float, x_kp1: float) -> npt.NDArray[np.float64]:
    """Calculate the local stiffness matrix for a cell with vertices a and b
    using a quadrature rule."""
    raise NotImplementedError


A_analytical = assemble_stiffness(fs, cell_stiffness)
A_quadrature = assemble_stiffness(fs, cell_stiffness_quadrature)
assert np.all(np.isclose(A_analytical.todense(), A_quadrature.todense()))


# %% [markdown]
# ### Exercise 9
#
# Write a function `solve` which takes the `num_cells` (number of cells) as an
# argument. Return the solution from the function. Plot a sequence of solutions
# on increasingly fine meshes.
#
# %%
def solve(num_cells: int) -> tuple[Mesh, npt.NDArray[np.float64]]:
    """Solve the Poisson problem on a uniform unit interval mesh.

    Args:
        num_cells: Number of cells in the mesh.

    Returns:
        The mesh and the vector of finite element solution coefficients.
    """
    raise NotImplementedError

for n in [4, 8, 16, 32, 64]:
    mesh_n, u_n = solve(n)
    plt.plot(mesh_n.geometry, u_n, "o-", label=f"$N = {n}$")

x_fine = np.linspace(0.0, 1.0, 200)
plt.plot(x_fine, np.sin(c * x_fine), "k--", label="exact")
plt.xlabel(r"$x$")
plt.ylabel(r"$u$")
plt.legend()
plt.show()


# %% [markdown]
# ### Exercise 10
#
# Write a function `solve_with_error` which takes `num_cells` as an argument
# and returns the squared $H^1_0$ error between the exact solution $u$ and
# the finite element solution $u_h$,
#
# $$
# e_h = \lVert u - u_h \rVert^2_{H^1_0} = \int_0^1 \left( \frac{\mathrm{d}u(x)}{\mathrm{d}x} - \frac{\mathrm{d}u_h(x)}{\mathrm{d}x}\right)^2 \, \mathrm{d}x.
# $$
#
# Evaluate the integral cell-by-cell using the same 2-point Gauss rule used to
# assemble the load vector. On each cell $K_k = [a, b]$ the finite element
# solution can be expanded in the basis as $u_h(x) = \sum_{i = 0}^{N} u_h^i
# \phi_i(x)$, so its derivative is
#
# $$
# \frac{\mathrm{d}}{\mathrm{d}x} u_h(x) = \sum_{i = 0}^{N} u_h^i \, \frac{\mathrm{d}}{\mathrm{d}x} \phi_i(x)
# $$
#
# which can be transferred to reference cell $\hat{K}$ using the map
# $F_{k}(\hat{x}) = x = a + h\hat{x}$.
#
# On a sequence of increasingly refined meshes (e.g. $N = 16, 32, 64, \ldots$)
# compute the associated error $e_h$. Algorithmically, calculating $e_h$ is
# another 'assembly' loop across the cells, the calculation of a local error
# contribution, and assembly into a single scalar - rather than a matrix
# ($\mathbf{K}$) or a vector ($\mathbf{f}$).


# Plot the error $e_h$ against $h$ on a log-log plot, and calculate the slope
# (in log-log space). What do you observe? Does this agree with the finite
# element error estimate you derived in class?
# %%
def solve_with_error(
    num_cells: int,
) -> float:
    """Solve the Poisson problem and return the squared H^1_0 error between
    the exact solution u and the finite element solution u_h, computed by
    cell-wise quadrature of int (u' - u_h')^2 dx using the reference basis
    derivatives."""
    raise NotImplementedError

num_cells_list = [16, 32, 64, 128, 256, 512]
hs = np.array([1.0 / n for n in num_cells_list])
errors = np.array([solve_with_error(n) for n in num_cells_list])

slope, intercept = np.polyfit(np.log(hs), np.log(errors), 1)
print(slope)
print(intercept)
print(f"Slope of log(e_h) vs log(h): {slope[0]:.3f}")

plt.figure()
plt.loglog(hs, errors, "o-", label=r"$e_h$")
plt.loglog(
    hs,
    np.exp(intercept + 1.0) * hs**slope,
    "--",
    label=f"slope $\\approx$ {slope[0]:.3f}",
)
plt.xlabel(r"$h$")
plt.ylabel(r"$e_h$")
plt.legend()
plt.show()
