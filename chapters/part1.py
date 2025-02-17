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
# The notebook is not complete - your task will be to complete the missing
# parts and submit the completed notebook as the first part of the coursework. 
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
# ## Exercise 1
# 
# Comment on the following aspects of this PDE:
#
# 1. Linear or non-linear, and why?
# 2. For $k^2 > 0$, elliptic, parabolic or hyperbolic? 
# 
# %% [markdown]
# Write your answer here.
#
# %% [markdown]
# ## Exercise 2
# 
# We would like to *verify* that our finite element method is working as we
# would expect (more on this later). One straightforward way to do this is via
# the method of manufactured solutions - we will invent a solution which satisfies
# the Dirichlet boundary conditions
#
# %% [markdown]
# Write your answer here.
