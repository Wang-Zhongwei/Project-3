# Project-3
Diffusion Model Using Finite Difference Methods. Class Project for Numerical Methods in Scientific Computing

## bvp.py
This Python script defines a simulation model for the diffusion equation in 2D domains.

The script includes two main classes: Diffusion and Diffusion2D.

The Diffusion class:

Initializes the simulation domain, grid spacing, boundary conditions, initial conditions, and heat source.
Supports various time evolution methods like Euler's method, the Trapezoidal rule, and the Method of Lines.
Computes the next time step based on the selected method.
Loads the evolution operator, which defines the change in the system over time.
Provides a method to calculate the error between the simulation result and the expected solution.
The Diffusion2D class, a subclass of Diffusion, creates the 2D second-order finite difference matrix used for the 2D diffusion equation. This class includes an additional method to create a sparse matrix representation of the finite difference operator.

Please note that this script is designed to handle Dirichlet boundary conditions. Other types of boundary conditions such as Neumann, Robin, and Periodic are not implemented and will raise an exception.






