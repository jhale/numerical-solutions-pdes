---
title: Numerical Solutions of PDEs and Applications
file_format: mystnb
author:
- Jack S. Hale
- Franck Sueur
bibliography:
- references.bib
date: Summer Semester 2024/2025
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: myst
---

# Numerical Solutions of PDEs and Applications

This course is an introduction to the numerical solution of partial
differential equations (PDEs). It contains a theoretical part setting
the mathematical foundations necessary for some important numerical
methods used to obtain solutions to some classical PDEs, in particular
the finite element method and the finite difference method. The
theoretical part of the course is supported by the development of a
one-dimensional Galerkin finite element code for the Poisson problem and
a one-dimensional finite difference code for a scalar hyperbolic
transport problem.

We largely follow the reference [@Q] which is available via the
[National Library's Website](https://a-z.lu) as well as on some less
official websites. In particular, we cover the contents of [@Q Chapters
1, 2, 3, 4, and 14].

# Syllabus

## A brief survey of partial differential equations

This chapter briefly introduces the notion of linear PDEs and their
classification into elliptic, parabolic and hyperbolic equations. We
mention some classical examples, mainly issued from physics and
engineering, such as the transport equation, the Laplace equation, the
heat equation and the wave equation.

## Elements of functional analysis

We introduce the main notions and theoretical results of functional
analysis that are extensively used in the numerical analysis of partial
differential equations. We consider the Riesz representation theorem
regarding representation of continuous linear forms on a Hilbert space,
and we survey the notions of bilinear form, of continuous injection of a
Hilbert space into another, the notion of derivative in the sense of
Fréchet, some elements of the theory of distributions, the basic
properties of the Lebesgue and Sobolev spaces, and the notion of adjoint
operator.

## Elliptic equations

We illustrate boundary-value problems for elliptic equations (in one and
several dimensions), present their variational reformulations, treat the
boundary conditions and analyze their well-posedness. Several examples
of physical interest are introduced, in particular the Poisson equation,
starting with the one-dimensional case, for various boundary conditions.
We consider some variational formulations of these problems, and then
turn to the boundary-value problems associated to the Poisson equation
in the two-dimensional case. We establish that under some regularity
condition the weak formulation is equivalent to the strong one. For
general elliptic problems, the Lax-Milgram theorem ensures that the weak
formulation is well-posed.

## The Galerkin finite element method for elliptic problems

We formulate Galerkin's method for the numerical discretization of
elliptic boundary-value problems and analyze its existence, uniqueness,
stability and convergence features in an abstract functional setting. We
then introduce the Galerkin finite elements method, first in one
dimension, and then in several dimensions.

## The Galerkin finite element method -- a numerical example

This section introduces students to the development of a one-dimensional
Galerkin finite element solver for the Poisson problem using Python. We
focus on building the solver within a provided Jupyter notebook,
providing hands-on experience with the computational and algorithmic
aspects of the finite element method.

## Finite differences for hyperbolic equations

The aim of this final chapter is to study classical finite differences
methods for approximating solutions of first-order hyperbolic equations.
We start with exposing some hyperbolic equations starting with the
scalar transport problem in one dimension which we analyze by the method
of characteristics. We also establish an a priori estimate by the energy
method. We then turn to systems of linear hyperbolic equations in one
dimension and give the example of the wave equation. Then we introduce
the finite difference method, together with its variants: the
forward/centered Euler scheme, the Lax-Friedrichs scheme, the
Lax-Wendroff scheme, in the case of the scalar transport problem, before
to move to more general cases, for which we analyze the consistency,
stability, convergence, dissipation and dispersion properties of the
finite difference methods

## Finite differences -- a numerical example

This section introduces students to the development of a one-dimensional
finite difference solver for a hyperbolic transport problem using
Python. We focus on building the solver within a provided Jupyter
notebook, providing hands-on experience with the computational and
algorithmic aspects of the finite difference method.

# Practical matters

## Assumed knowledge

It is assumed that students have taken *Functional Analysis* and
*Numerical Analysis* courses in MAMATH, or have equivalent knowledge.

## Dropping the course

Students cannot drop this course after the two week trial period at the
start of the semester.

## Teaching

For *practical session* must bring a laptop with Python version 3.10 or
later installed (recommended), or with access to [Google
Colaboratory](https://colab.research.google.com).

## Assessment

Assessment is via coursework (30%) and final examination (70%). The due
date for the coursework will be set at a later time.

## Attendance policy

Attendance at the *lectures* is strongly advised.

Attendance at the *practical sessions* is mandatory as they relate to
the coursework - one absence will be excused. Further absences must be
supported by documentation according to the study regulations and a
meeting with the study programme director and/or instructors.

## Retake policy

Failure on the coursework can be compensated through the examination,
and vice versa, as long as the final mark for the course is greater than
or equal to 10.

In the event that the final mark is less than 10:

-   The student may request a retake the examination once.

-   The student may not resubmit the coursework.

If the final mark remains less than 10 the student has failed, and can
take the entire course again in the next semester that it is offered.

## Communication

Electronic communication to students as a group will be via the
University's Moodle. Private communications from students to the
instructors must be made via the University email.

## Technical notes

### Building LaTeX source

To continuously build and view the TeX source:

    latexmk -lualatex -pvc -output-directory=build/ main.tex

### Converting to Markdown

To convert the tex file to Markdown (WIP):

    pandoc -s -C -t markdown-citations --from latex --to markdown --bibliography=ref.bib main.tex
