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
# # The Galerkin finite element method
#
# In this notebook we show the creation of a simple one-dimensional Galerkin
# finite element code for the following reaction-diffusion type partial
# differential equation.
#
# *The notebook is not complete - your task will be to complete the missing
# parts and submit it as the first part of the coursework.*
#
# ## Preliminaries
#
# Find $u : [0, 1] \to \mathbb{R}$ such that
#
# $$
# - \nabla u + k^2 u = f,
# $$
#
# with Dirichlet-type boundary conditions
#
# $$
# u(0) = 0, \quad u(1) = 0.
# $$
#
# ### Exercise 1
#
# Comment on the following aspects of this PDE:
#
# 1. Linear or non-linear, and why?
# 2. For $k^2 > 0$, elliptic, parabolic or hyperbolic?
#
# %% [markdown]
# *Answer*
#
# Write your answer using Markdown here.

# %% [markdown]
# ### Exercise 2
#
# Once we have finished, we would like to *verify* that our finite element code
# can correctly solve a PDE. One straightforward way to do this is via the
# method of manufactured solutions. We will invent a solution
# $u_{\mathrm{exact}}$ that satisfies the Dirichlet boundary conditions, then
# derive $f$ via substitution into the strong form of the PDE. On a sequence of
# finer and finer meshes, we will solve our PDE and check that our numerical
# solution converges at the correct rate. If it does, then it is likely we have
# a correct implementation.
#
# Let $u_{\mathrm{exact}} = \sin(cx)$ with $c$ an unknown constant.
#
# %% [markdown]
# *Answer*
#
# Write your answer using Markdown here.

# %%
import numpy as np


def generate_mesh(n):
    cell_vertex_connectivity = np.zeros()

    return cell_vertex_connectivity
