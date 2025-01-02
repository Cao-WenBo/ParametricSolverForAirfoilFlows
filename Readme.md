# Solving all laminar flows around airfoils all-at-once using a parametric neural network solver

## Abstract

Recent years have witnessed increasing research interests of physics-informed neural networks (PINNs) in solving forward, inverse, and parametric problems governed by partial differential equations (PDEs). Despite their promise, PINNs still face significant challenges in many scenarios due to ill-conditioning. Time-stepping-oriented neural network (TSONN) addresses this by reformulating the ill-conditioned optimization problem into a series of well-conditioned sub-problems, greatly improving its ability to handle complex scenarios. This paper presents a new solver for laminar flow around airfoils based on TSONN and mesh transformation, validated across various test cases. Specifically, the solver achieves mean relative errors of approximately 3.6% for lift coefficients and 1.4% for drag coefficients. Furthermore, this paper extends the solver to parametric problems involving flow conditions and airfoil shapes, covering nearly all laminar flow scenarios in engineering, i.e., the shape parameter space is defined as the union of 30% perturbations of each airfoil in the UIUC airfoil database, the Reynolds number ranges from 100 to 5000, and the angle of attack spans from -5° to 15°. The parametric solver solves all laminar flows within the parameter space in just 4.6 day, at approximately 40 times the computational cost of solving a single flow. The model training involves hundreds of millions of flow conditions and airfoil shapes, ultimately yielding a surrogate model with strong generalization capability that does not require labeled data. Concretely, the surrogate model achieves average errors of 4.6% for lift coefficients and 1.1% for drag coefficients, demonstrating its potential high generalizability, low-cost and effectiveness in tackling high-dimensional parametric problem. 

----------

* **Airfoil2CST/cst_geom.py:** Conversion between airfoil coordinates and CST parameters.

- **ParametricSolver.py:** The main class that predicts the  flow around an airfoil.

- **result.py:** The main program  for plotting the results in the paper.

