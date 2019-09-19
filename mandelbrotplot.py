import numpy as np
import cupy as cp
import matplotlib
from tqdm import tqdm

def mandelbrot_plot_cpu(res, iterations):
    x = cp.linspace(-2, 2, num=res).reshape((1, res))
    y = cp.linspace(-2, 2, num=res).reshape((res, 1))
    C = cp.tile(x, (res, 1)) + 1j * cp.tile(y, (1, res))

    Z = np.zeros((res, res), dtype='complex16')
    M = np.full((res, res), True, dtype=bool)

    for i in tqdm(range(iterations)):
        Z[M] = Z[M] * Z[M] + C[M]
        M[np.abs(Z) > 2] = False

    M = np.uint8(np.flipud(M-1) * 265)

    print(M.shape)
    print(M.dtype)
    print(M)
    matplotlib.image.imsave('mandelbrot.png', M)


def mandelbrot_plot_gpu(res, iterations):
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()

    x = cp.linspace(-2, 2, num=res).reshape((1, res))
    y = cp.linspace(-2, 2, num=res).reshape((res, 1))
    C = cp.tile(x, (res, 1)) + 1j * cp.tile(y, (1, res))

    Z = cp.zeros((res, res), dtype=complex)
    M = cp.full((res, res), True, dtype=bool)


    for i in tqdm(range(iterations)):
        Z[M] = Z[M] * Z[M] + C[M]
        M[cp.abs(Z) > 2] = False

    M_cpu = cp.asnumpy(M)
    M_cpu = M_cpu.astype('int8')
    M_cpu = np.flipud(1-M_cpu) * 265


    matplotlib.image.imsave('mandelbrot.png', M_cpu)

    mempool.free_all_blocks()


mandelbrot_plot_gpu(5000, 10000000)
