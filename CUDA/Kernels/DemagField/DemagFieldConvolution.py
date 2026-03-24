from cupy import RawKernel

# kernel code
kernel_code = r'''
    #include <cupy/complex.cuh>
    
    __device__ int Sign(int x);
    __device__ inline int tensor_ind(int i, int j, int k,int Ny,int Nz);
    
    struct indices{
    int i;
    int j;
    int k;
    };
    __device__ struct indices ind2sub(int index,int Nx,int Ny,int Nz);
    
    extern "C" __global__ void demag_field_convolution(float* kxx, 
                                                    float* kyy, 
                                                    float* kzz,
                                                    float* kxy, 
                                                    float* kxz, 
                                                    float* kyz,
                                                    complex<float>* xFFT,
                                                    complex<float>* yFFT,
                                                    complex<float>* zFFT,
                                                    int Nx,int Ny,int Nz)
{
    // expected data size is 2*Nx * 2*Ny * (Nz +1)
    // kernel size is (Nx+1)*(Ny+1)*(Nz+1)
    
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (index < 2*Nx*2*Ny*(Nz+1))
    {
            struct indices inds = ind2sub(index,Nx,Ny,Nz);
            
            float Nxx,Nxy,Nyy,Nzz,Nyz,Nxz;
            complex<float> Mx, My, Mz;

            Mx = xFFT[index];
            My = yFFT[index];
            Mz = zFFT[index];
            
        // offset into reduced size tensor arrays
           int dx = inds.i, dy = inds.j,dz = inds.k;
            
            if (dx -(Nx) > 0)
            {
               dx = 2* Nx -inds.i;
            }        
            if (dy -(Ny) > 0)
            {
               dy = 2* Ny -inds.j;
            }  
            
            if (dz -(Nz) > 0)
            {
               dz = 2* Nz -inds.k;
            }                  
                 
            int tensor_index = tensor_ind(dx,dy,dz,Ny,Nz);
            
            // diagonal elements of tensor are symmetric +-x,y,z
            Nxx = kxx[tensor_index];
            Nyy = kyy[tensor_index];
            Nzz = kzz[tensor_index];
                     
            // recover correct size of tensor values from symmetries
            int G;
            G = Sign((Nx - inds.i)) * Sign(Ny - inds.j);
            Nxy = G*kxy[tensor_index];

            G = Sign((Nx - inds.i)) * Sign(Nz - inds.k);
            Nxz = G*kxz[tensor_index];

            G = Sign((Ny - inds.j)) * Sign(Nz - inds.k);
            Nyz = G*kyz[tensor_index];
            
        xFFT[index] = Nxx * Mx + Nxy * My + Nxz * Mz; 
        yFFT[index] = Nxy * Mx + Nyy * My + Nyz * Mz;
        zFFT[index] = Nxz * Mx + Nyz * My + Nzz * Mz;
        
    }

}

__device__ int Sign(int x)
{
    if (x == 0)
    {
        return -1;
    }
    return ((x > 0) - (x < 0));
}

__device__ inline int tensor_ind(int i, int j, int k,int Ny,int Nz)
{
    return (k + (Nz + 1) * (j + (Ny + 1) * i));
}

__device__ struct indices ind2sub(int indexes,int Nx,int Ny,int Nz)
{
       int i = indexes / (2*Ny * (Nz+1));
       int j = (indexes/ (Nz+1)) % (2*Ny);
       int k = indexes % (Nz+1);
       
       struct indices inds;
       inds.i = i;
       inds.j = j;
       inds.k = k;
       
       return inds;
}
'''
demag_field_convolution = RawKernel(kernel_code, 'demag_field_convolution')
demag_field_convolution.compile()


