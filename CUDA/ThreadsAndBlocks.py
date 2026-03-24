
def threads_and_blocks_3d(threads: int,Nx: int, Ny: int, Nz: int):

     block_z = (Nx + threads - 1) // threads
     block_y = (Ny + threads - 1) // threads
     block_x = (Nz + threads - 1) // threads

     blocks = (block_x, block_y, block_z)

     return threads, blocks


def threads_and_blocks_1d(threads: int,N: int):

     block = (N + threads - 1) // threads

     return threads, block
