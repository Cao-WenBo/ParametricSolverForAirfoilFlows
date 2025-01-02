# Solving all laminar flows around airfoils all-at-once using a parametric neural network solver

## Abstract

Physics-informed neural networks (PINNs) have recently become a new popular method for solving forward and inverse problems governed by partial differential equations (PDEs). However, in the flow around airfoils, the fluid is greatly accelerated near the leading edge, resulting in a local sharper transition, which is difficult to capture by PINNs. Therefore, PINNs are still rarely used to solve the flow around airfoils. In this study, we combine physical-informed neural networks with mesh transformation, using neural network to learn the flow in the uniform computational space instead of physical space. Mesh transformation avoids the network from capturing the local sharper transition and learning flow with internal boundary (wall boundary). We successfully solve inviscid flow and provide an open-source subsonic flow solver for arbitrary airfoils. Our results show that the solver exhibits higher-order attributes, achieving nearly an order of magnitude error reduction over second-order finite volume methods (FVM) on very sparse meshes. Limited by the learning ability and optimization difficulties of neural network, the accuracy of this solver will not improve significantly with mesh refinement. Nevertheless, it achieves comparable accuracy and efficiency to second-order FVM on fine meshes. Finally, we highlight the significant advantage of the solver in solving parametric problems, as it can efficiently obtain solutions in the continuous parameter space about the angle of attack. 

----------

* **Airfoil2CST/cst_geom.py:** Conversion between airfoil coordinates and CST parameters.

- **ParametricSolver.py:** The main class that predicts the  flow around an airfoil.

- **result.py:** The main program  for plotting the results in the paper.

