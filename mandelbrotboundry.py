#package import
#Numpy and sklearn aren't native to python, so you will need to install them
import numpy as np
import math as m
import cupy as cp
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def mandelbrot_boundary(res_c, res, iter_scale, growth_rate):
    point_c = 0
    big = res_c
    tile_c_x = 0
    tile_c_y = 0
    tile_stop = int(res/5000)
    tile_stop_c = 1

    while res_c <= res:
        if (res_c < 5000):
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            iterations = int(iter_scale*res_c)

            print("Next Resolution: ", res_c, "x", res_c, ";Testing ", res_c ** 2, " points")
            X = cp.linspace(-2, 2, num=res_c).reshape((1, res_c))
            Y = cp.linspace(-2, 2, num=res_c).reshape((res_c, 1))
            C = cp.tile(X, (res_c, 1)) + 1j * cp.tile(Y, (1, res_c))

            Z = cp.zeros((res_c, res_c), dtype=complex)
            M = cp.full((res_c, res_c), True, dtype=bool)

            for i in tqdm(range(iterations)):
                Z[M] = Z[M] * Z[M] + C[M]
                M[cp.abs(Z) > 2] = False

            M_cpu = cp.asnumpy(M)

            for i in tqdm(range(res_c-1)):
                for j in range(res_c-1):
                    a = M_cpu[i, j]
                    b = M_cpu[(i + 1), j]
                    c = M_cpu[i, (j + 1)]
                    d = M_cpu[(i + 1), (j + 1)]
                    if (a != b) or (a != c) or (a != d) or (b != c) or (b != d) or (c != d):
                        point_c = point_c + 1

                    else:
                        pass

            if res_c == big:
                x = np.array([m.log10(res_c)])
                y = np.array([m.log10(point_c)])


            else:
                x = np.append(x, [m.log10(res_c)], axis=0)
                y = np.append(y, [m.log10(point_c)], axis=0)


            res_c = m.floor(res_c * growth_rate) + 1
            point_c = 0

        else:
            res_c = 5000
            point_c = 1


            while tile_stop_c <= tile_stop:
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                print("Now testing: ", tile_stop_c, "out of ", tile_stop)
                iterations = int(iter_scale*res_c)

                tile_c_x = 0
                tile_c_y = 0
                while tile_c_y < tile_stop_c:
                    while tile_c_x < tile_stop_c:
                        print(tile_c_x, tile_c_y)

                        step_tile = 4/tile_stop_c
                        mempool = cp.get_default_memory_pool()
                        mempool.free_all_blocks()

                        X = cp.linspace((-2+(step_tile*tile_c_x)), (-2+(step_tile*(tile_c_x+1))), num=5000).reshape((1, 5000))
                        Y = cp.linspace((-2+(step_tile*tile_c_y)), (-2+(step_tile*(tile_c_y+1))), num=5000).reshape((5000, 1))
                        C = cp.tile(X, (5000, 1)) + 1j * cp.tile(Y, (1, 5000))

                        Z = cp.zeros((5000, 5000), dtype=complex)
                        M = cp.full((5000, 5000), True, dtype=bool)

                        for i in tqdm(range(iterations)):
                            Z[M] = Z[M] * Z[M] + C[M]
                            M[cp.abs(Z) > 2] = False

                        M_cpu = cp.asnumpy(M)
                        print("Now finding the set:")
                        for i in tqdm(range(5000)):
                            for j in range(5000):
                                if M_cpu[i, j] == True:
                                    point_c = point_c + 1

                                else:
                                    pass

                        tile_c_x = tile_c_x + 1

                    tile_c_y = tile_c_y + 1
                    tile_c_x = 0

                x = np.append(x, [m.log10(res_c)], axis=0)
                y = np.append(y, [m.log10(point_c)], axis=0)
                tile_stop_c = tile_stop_c + 1
                res_c = res_c + 5000
                point_c = 1

    x = x.reshape((-1,1))
    y = y.flatten()

    model = LinearRegression()
    model.fit(x, y)
    r_sq = model.score(x, y)

    print("r^2:", r_sq)
    print("The Dimenionsality is:", model.coef_)
    print("The Scaling Factor is:", model.intercept_)

    plt.plot(x, y, 'ro')
    plt.axis([0,10,0,10])
    plt.show()

mandelbrot_boundary(50, 5000, 1, 1.2)
