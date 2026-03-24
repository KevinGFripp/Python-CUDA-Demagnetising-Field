from numpy import zeros,zeros_like,pi,arange,log10,abs
from numpy import ndarray
from DLL_libraries.DemagnetisationTensor.GPU_Demag_Tensor import gpu_demag_tensor
from CUDA.Kernels.DemagField.ComputeDemagField import compute_demag_field
from cupy import asarray, asnumpy, float32
import matplotlib.pyplot as plt

def example_compute_dipole_field():
    """
    Compute the stray field from a single cell to test the accuracy versus the analytical result,
    a dipole.

    In single precision, the field is accurate till the noise floor around 1e-8,
    where round-off errors dominate.
    The cell-to-cell distances corresponding to the noise floor is strictly far-field.
    :return:
    """

    #Compute field for a dipole
    #Grid size
    Nx = 512
    Ny = 512
    Nz = 1

    #Cell sizes (nm)
    Dx = 1.0
    Dy = 1.0
    Dz = 1.0

    # Material parameters
    mu0 = 4*pi/10
    Ms = 1 / mu0  # kA/m

    kxx, kyy, kzz, kxy, kxz, kyz = gpu_demag_tensor(Nx,Ny,Nz,Dx,Dy,Dz)

    # define magnetisation
    Mx = zeros(shape=(Nx, Ny, Nz), dtype=float32,order='C')
    My, Mz = (zeros_like(Mx),zeros_like(Mx))
    Mx[Nx // 2 - 1, Ny // 2 - 1, 0] = Ms

    #move to gpu
    Mx = asarray(Mx,order='C')
    My = asarray(My,order='C')
    Mz = asarray(Mz,order='C')

    #solve
    Hx, Hy, Hz = compute_demag_field(kxx,kyy,kzz,kxy,kxz,kyz,
                                     Mx,My,Mz,
                                     Nx,Ny,Nz)

    #return to cpu
    Hx_cpu, Hy_cpu, Hz_cpu = asnumpy(Hx), asnumpy(Hy), asnumpy(Hz)


    # #analytical field Bx = mu0/2 * Ms/r^3 xhat
    xrange = arange(Nx//2)
    Bfield = ndarray(shape=Nx//2, dtype=float32,order='C')
    Bfield[0] = -mu0/3. * Ms
    Bfield[1:Nx//2] = -mu0*(Ms/6)/xrange[1:Nx//2]**3


    plt.plot(xrange,log10(abs(Bfield)),
             linestyle='-', color='k', linewidth=3)
    plt.plot(xrange[:Nx//2:3],log10(abs(mu0*Hx_cpu[(Nx//2 -1):(Nx -1):3,Ny//2 -1,0])),
              linestyle='none',color='r',linewidth=3,marker='o',markersize=4,markerfacecolor='r')
    plt.xlabel('x (nm)',fontsize=18)
    plt.ylabel('log(|Bx|)',fontsize=18)
    plt.legend(['Dipole','Bx'],fontsize=16,edgecolor='none')
    plt.rcParams["font.family"] = "Arial"
    plt.show()

    return