from cupy.fft import rfftn,irfftn
from CUDA.ThreadsAndBlocks import threads_and_blocks_1d
from CUDA.Kernels.DemagField.DemagFieldConvolution import demag_field_convolution


def compute_demag_field(kxx,kyy,kzz,kxy,kxz,kyz,
                        Mx,My,Mz,Nx,Ny,Nz):


    xFFT = rfftn(Mx, s=(2 * Nx, 2 * Ny, 2 * Nz))
    yFFT = rfftn(My, s=(2 * Nx, 2 * Ny, 2 * Nz))
    zFFT = rfftn(Mz, s=(2 * Nx, 2 * Ny, 2 * Nz))

    threads, blocks = threads_and_blocks_1d(256, 2 * Nx *2* Ny * (Nz +1))

    demag_field_convolution((blocks,), (threads,),
                         (kxx, kyy, kzz,
                          kxy, kxz, kyz,
                          xFFT, yFFT, zFFT,
                          Nx,Ny,Nz))

    Hx = irfftn(xFFT)
    Hy = irfftn(yFFT)
    Hz = irfftn(zFFT)

    return Hx, Hy, Hz