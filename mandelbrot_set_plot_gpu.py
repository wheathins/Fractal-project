import numpy as np
import cupy as cp
import matplotlib
from tqdm import tqdm

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
    m_b = np.full((res, res), False, dtype=bool)
    mempool.free_all_blocks()



    m_b = m_b.astype('int32')
    m_b = np.flipud(1-m_b) * 256
    matplotlib.image.imsave('mandelbrotboundary.png', m_b, format='png')
    mempool.free_all_blocks()


#    for i in tqdm(range(res-1)):
#        for j in range(res-1):
#            a = M_cpu[i, j]
#            b = M_cpu[(i + 1), j]
#            c = M_cpu[i, (j + 1)]
#            d = M_cpu[(i + 1), (j + 1)]
#            if (a != b) or (a != c) or (a != d) or (b != c) or (b != d) or (c != d):
#                m_b[i,j] = True

#            else:
#                pass
