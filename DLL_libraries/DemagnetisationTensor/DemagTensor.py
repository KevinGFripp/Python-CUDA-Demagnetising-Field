import ctypes
from numpy.ctypeslib import ndpointer

lib =ctypes.cdll.LoadLibrary("DLL_libraries/DemagnetisationTensor/DemagnetisingTensor.dll")

demag_tensor = lib.ComputeDemagTensorNewell_GQ
demag_tensor.restype = None
    #           [       tensor         ,cell sizes, grid sizes]
    #arguments [kxx,kyy,kzz,kxy,kxz,kyz,Dx,Dy,Dz,Nx,Ny,Nz]
demag_tensor.argtypes = [ndpointer(ctypes.c_double,ndim =3, flags="C_CONTIGUOUS"),
                         ndpointer(ctypes.c_double,ndim =3, flags="C_CONTIGUOUS"),
                         ndpointer(ctypes.c_double,ndim =3, flags="C_CONTIGUOUS"),
                         ndpointer(ctypes.c_double,ndim =3, flags="C_CONTIGUOUS"),
                         ndpointer(ctypes.c_double,ndim =3, flags="C_CONTIGUOUS"),
                         ndpointer(ctypes.c_double,ndim =3, flags="C_CONTIGUOUS"),
                         ctypes.c_double,
                         ctypes.c_double,
                         ctypes.c_double,
                         ctypes.c_int,
                         ctypes.c_int,
                         ctypes.c_int]
