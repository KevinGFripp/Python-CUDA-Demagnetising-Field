from numpy import zeros_like,zeros,pi,arange,meshgrid,exp,cos,sqrt
from DLL_libraries.DemagnetisationTensor.GPU_Demag_Tensor import gpu_demag_tensor
from CUDA.Kernels.DemagField.ComputeDemagField import compute_demag_field
from cupy import asarray, asnumpy, float32
import matplotlib.pyplot as plt

def example_compute_demag_field():
    """
    Compute the demagnetising (H) field for a disk in the vortex state and plot.
    # The flux-closure domain state minimises the in-plane magnitude of the field
    # at the cost of the out-of-plane component, and singularity (vortex core).
    :return:
    """
    #Grid size
    Nx = 512
    Ny = 512
    Nz = 1

    #Cell sizes (nm)
    Dx = 5.0
    Dy = 5.0
    Dz = 20.0

    # Material parameters
    mu0 = 4*pi/10
    Ms = 800.0  # kA/m
    diameter = 2250.0

    kxx, kyy, kzz, kxy, kxz, kyz = gpu_demag_tensor(Nx,Ny,Nz,Dx,Dy,Dz)

    # define magnetisation
    Mx = zeros(shape=(Nx, Ny, Nz), dtype=float32,order='C')
    My, Mz = (zeros_like(Mx),zeros_like(Mx))

    # vortex state in xy-plane, core in z
    for i in range(Nx):
        for j in range(Ny):
            radius_sq = (Dx*(Nx/2 - i))**2 + (Dy*(Ny/2 - j))**2

            if radius_sq < (diameter**2)/4:
                mx = float32(cos( (pi/Ny) * float32(j) ))
                my = float32(-cos( (pi/Nx) * float32(i) ))
                mz = -float32(0.05)

                # normalise to unit length
                norm_factor = sqrt(mx**2 + my**2 + mz**2)

                Mx[i,j,  0] = float32(Ms * mx/norm_factor)
                My[i,j,  0] = float32(Ms * my/norm_factor)
                Mz[i, j, 0] = float32(Ms * mz/norm_factor)

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
    Mx, My, Mz = asnumpy(Mx), asnumpy(My), asnumpy(Mz)

    xrange = 5.*(arange(0,Nx,1) - Nx/2)
    yrange = 5.*(arange(0,Ny,1) - Ny/2)
    X,Y = meshgrid(xrange,yrange)

    fig = plt.figure(figsize=(12,7))

    ax1 = fig.add_subplot(2,3,1)
    surf1 = ax1.imshow(mu0*Hx_cpu[0:Nx,0:Ny,0], cmap='turbo')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    ax1.set_title('Hx')
    fig.colorbar(surf1,shrink=0.35, aspect=15)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')
    ax2.set_title('Hy')
    surf2 = ax2.imshow(mu0*Hy_cpu[0:Nx, 0:Ny, 0], cmap='turbo')
    fig.colorbar(surf2,shrink=0.35, aspect=15)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_xlabel('x (nm)')
    ax3.set_ylabel('y (nm)')
    ax3.set_title('Hz')
    surf3 = ax3.imshow(mu0*Hz_cpu[0:Nx, 0:Ny, 0], cmap='turbo')
    fig.colorbar(surf3,shrink=0.35, aspect=15)

    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.set_xlabel('x (nm)')
    ax4.set_ylabel('y (nm)')
    ax4.set_zlabel('Mx (kA/m)')
    ax4.set_title('Mx')
    surf4 = ax4.plot_surface(X, Y, Mx[0:Nx, 0:Ny, 0], cmap='turbo', edgecolor='none')
    fig.colorbar(surf4, shrink=0.35, aspect=15)

    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    ax5.set_xlabel('x (nm)')
    ax5.set_ylabel('y (nm)')
    ax5.set_zlabel('My (kA/m)')
    ax5.set_title('My')
    surf5 = ax5.plot_surface(X, Y, My[0:Nx, 0:Ny, 0], cmap='turbo', edgecolor='none')
    fig.colorbar(surf5, shrink=0.35, aspect=15)

    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    ax6.set_xlabel('x (nm)')
    ax6.set_ylabel('y (nm)')
    ax6.set_zlabel('Mz (kA/m)')
    ax6.set_title('Mz')
    surf6 = ax6.plot_surface(X, Y, Mz[0:Nx, 0:Ny, 0], cmap='turbo', edgecolor='none')
    fig.colorbar(surf6, shrink=0.35, aspect=15)

    plt.show()

    return
