# Python + CUDA FFT Demagnetising Field Solver
Python + CUDA FFT solver for the demagnetising field arising from a magnetisation distribution on a regular Cartesian grid.

## Implementation
The demagnetising field is computed as a convolution in Fourier space of the geometric (pre-computed) demagnetising tensor $N_{xy}$ and the magnetisation:

$\mathcal{F}(H_x) = \mathcal{F}(N_{xx}) * \mathcal{F}(M_x) + \mathcal{F}(N_{xy}) * \mathcal{F}(M_y) + \mathcal{F}(N_{xz}) * \mathcal{F}(M_z)$

$\mathcal{F}(H_y) = \mathcal{F}(N_{xy}) * \mathcal{F}(M_x) + \mathcal{F}(N_{yy}) * \mathcal{F}(M_y) + \mathcal{F}(N_{yz}) * \mathcal{F}(M_z)$

$\mathcal{F}(H_z) = \mathcal{F}(N_{xz}) * \mathcal{F}(M_x) + \mathcal{F}(N_{yz}) * \mathcal{F}(M_y) + \mathcal{F}(N_{zz}) * \mathcal{F}(M_z)$

All computation on the GPU is performed in single precision.

The demagnetising tensor is computed once via a C implementation using openMP parallelisation on the host, and transferred to the GPU.

Due to odd-even symmetries and diagonal symmetry of the demagnetising tensor, only 6 of 9 components are stored 
and approximately 1/8 of all interaction distances computed. 



## Accuracy 
The calculated field is accurate down to the noise floor of single precision.


<img width="560" height="431" alt="FieldSolveVersusDipole" src="https://github.com/user-attachments/assets/77838a98-80ce-4d57-b649-06c209314bd4" />



## Example : Field due to a Permalloy disk in the vortex state
The flux-closure domain state minimises the in-plane demagnetising field at the expense of the out-of-plane component.


<img width="1225" height="772" alt="FieldSolve_DiskVortex" src="https://github.com/user-attachments/assets/8d3ba354-d780-46e3-973b-fc684f69fe67" />



## Required packacges:
CUDA >= 11.0

Python >= 3.10

cuPy

NumPy
