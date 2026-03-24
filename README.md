Python + CUDA FFT solver for the demagnetising field arising from a magnetisation distribution on a regular Cartesian grid.

The demagnetising field is computed as a convolution in Fourier space of the geometric (pre-computed) demagnetising tensor and the magnetisation.
All computation on the GPU is performed in single precision.

The demagnetising tensor is computed once via a C implementation using openMP parallelisation on the host, and transferred to the GPU.


Required packacges:
cuPy, NumPy, Matplotlib

Results from examples:

-- Accuracy versus a dipole field --

<img width="640" height="480" alt="FieldSolveVersusDipole" src="https://github.com/user-attachments/assets/b7053028-1d04-4c60-bf6f-9c066e02c01f" />


-- Field distribution for a Permalloy disk in the vortex state --


<img width="1186" height="662" alt="FieldSolve_DiskVortex" src="https://github.com/user-attachments/assets/dbd0e9f4-8ad5-4fc4-8eca-07c633b6d312" />
