# Hybrid Matrix-Free Solver

This project implements a hybrid MPI + multithreading matrix-free finite element
solver for the convection–diffusion–reaction problem

$$
\begin{cases}
-\nabla \cdot (\mu \nabla u) + \boldsymbol{\beta} \cdot \nabla u + \gamma u = f,
  & \text{in } \Omega, \\
u = g, & \text{on } \Gamma_D \subset \partial\Omega, \\
\nabla u \cdot \mathbf{n} = h, & \text{on } \Gamma_N = \partial\Omega \setminus \Gamma_D.
\end{cases}
$$

## Building

```bash
$ mkdir -p build
$ cd build
$ cmake ..
$ make
```

## Repository layout

```
src/
build/
```
