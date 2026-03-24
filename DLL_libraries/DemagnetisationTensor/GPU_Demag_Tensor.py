from DLL_libraries.DemagnetisationTensor.DemagTensor import demag_tensor
from numpy import double,float32,zeros,zeros_like
from cupy import ascontiguousarray,asarray
from numpy.fft import rfftn


def gpu_demag_tensor(Nx: int,Ny: int,Nz: int,Dx: float,Dy: float,Dz: float):

    kxx = zeros(shape=(2 * Nx, 2 * Ny, 2 * Nz), dtype=double, order='C')
    kyy, kzz, kxy, kxz, kyz = (zeros_like(kxx),
                               zeros_like(kxx),
                               zeros_like(kxx),
                               zeros_like(kxx),
                               zeros_like(kxx))

    demag_tensor(kxx, kyy, kzz, kxy, kxz, kyz, Dx, Dy, Dz, Nx, Ny, Nz)

    kxx = -rfftn(kxx).real
    kyy = -rfftn(kyy).real
    kzz = -rfftn(kzz).real
    kxy = -rfftn(kxy).real
    kxz = -rfftn(kxz).real
    kyz = -rfftn(kyz).real

    # store ~first octant
    kxx = kxx[0:Nx+1,0:Ny+1,:]
    kyy = kyy[0:Nx+1,0:Ny+1,:]
    kzz = kzz[0:Nx+1,0:Ny+1,:]
    kxy = kxy[0:Nx+1,0:Ny+1,:]
    kxz = kxz[0:Nx+1,0:Ny+1,:]
    kyz = kyz[0:Nx+1,0:Ny+1,:]

    kxx_gpu = ascontiguousarray(asarray(float32(kxx), order='C'))
    kyy_gpu = ascontiguousarray(asarray(float32(kyy), order='C'))
    kzz_gpu = ascontiguousarray(asarray(float32(kzz), order='C'))
    kxy_gpu = ascontiguousarray(asarray(float32(kxy), order='C'))
    kxz_gpu = ascontiguousarray(asarray(float32(kxz), order='C'))
    kyz_gpu = ascontiguousarray(asarray(float32(kyz), order='C'))


    return kxx_gpu, kyy_gpu, kzz_gpu, kxy_gpu, kxz_gpu, kyz_gpu