#Numpy, sklearn, tqdm, cupy, and matplotlib aren't native to python, so you will need to install them
#I recommend using anaconda instead of pip to install these packages. This will resolve many headaches when installing Nvidia drivers and CUDA toolkits
import numpy as np
#general matrix and vector library
import math as m
#needed to calculate log10
import cupy as cp
#CUDA accelerated numpy, functions can be directly subisituted by changing np.* to cp.*
from tqdm import tqdm
#adds progress bars for calculations
from sklearn.linear_model import LinearRegression
#Linear regression for calculating the slope of the loglog plot of the resolution and amount of boxes that overlap the mandelbrot set
import matplotlib.pyplot as plt
#shows loglog plot

def mandelbrot_boundary_gpu(res_c, res, res_tile, iter_scale, growth_rate):
    ########parameters##########
    #res_c - initial resolution tested. I recommend using a value between 20 and 100
    #res - final resolution tested. Values between 1000-5000 show that the set has a fractal dimension of 2
    #res_tile - due to a high memory memory complexity (approx n^2), for testing high resolutions "tiling" is needed (see paper). This parameter controls the tile resolution. Set this parameter as high as memory allows.
    #Using a 1080 ti with 11GB of VRAM, I can calculate the set for resolution up to 10000x10000. For 8GB of VRAM, try res_tile = 7500; 6GB, 6000; 4GB, 5000. If you have less VRAM than 4GB, I recommend using your cpu instead of gpu.
    #iter_scale - Instead of calculating the mandelbrot set with a fixed amount iterations, the amount of iterations is based off of the resolution being calculated. Iter = current_res*iter_scale.
    #I recommend values between .5-5.
    #growth_rate - instead of testing all resolutions between res_c and res. I used an expontially increasing loop counter for the resolutions. growth_rate is the exponent. Must be greater than 1, the larger the value, the less resolutions are tested.
    #I recommend values between 1.05-1.3
    ###########################
    #point counter, var holds total amount of points in the set for a given resolution
    point_c = 0
    #var used to init x and y vars to store calculation results
    init_res = res_c
    #x and y tile counters, used as loop vars for looping tile over complex plane interval
    tile_c_x = 0
    tile_c_y = 0
    #determines final tile resolution
    tile_stop = int(res/res_tile)
    #tile counter, loops from 1 to tile_stop value
    tile_stop_c = 1

    #main resolution loop, res_c works as both the initial resl parameter as well as the loop counter
    while res_c < res:
        #this logic statement controls whether or to tile.
        if res_c < res_tile:
            #calc iters for a given res_c value
            iterations = int(res_c*iter_scale)
            #clear gpu memory
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

            #message to user on current res being calculated
            print("Next Resolution:", res_c, "x", res_c, ";Testing ", res_c ** 2, " points")
            #generate X and Y vectors, these vars will make up the real and imag parts of the complex plane matrix C
            X = cp.linspace(-2, 2, num=res_c).reshape((1, res_c))
            Y = cp.linspace(-2, 2, num=res_c).reshape((res_c, 1))
            C = cp.tile(X, (res_c, 1)) + 1j * cp.tile(Y, (1, res_c))

            #Z matrix of zeros from mandelbrot set definition
            Z = cp.zeros((res_c, res_c), dtype=complex)
            #M is matrix of bools used to store whether or not a given point in c is in the set or not
            #if the abs value of a point of Z is greater than 2, that entry in M is made false, halting multiplication
            M = cp.full((res_c, res_c), True, dtype=bool)

            #perform calculations
            for i in tqdm(range(iterations)):
                Z[M] = Z[M] * Z[M] + C[M]
                M[cp.abs(Z) > 2] = False

            #Move the M array to CPU
            M_cpu = cp.asnumpy(M)
            #update user
            print("Now finding the set")
            #these nested loops loop over the logic set (which is now on the cpu) to find the amount of points in the boundary of the set
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

            #creates arrays x and y if first time through the loop
            #otherwise adds new entries to x and y
            #x and y are used to calc a linear regression, the slope of this regression is the estimate for the fractal dimension
            if res_c == init_res:
                x = np.array([m.log10(res_c)])
                y = np.array([m.log10(point_c)])
            else:
                x = np.append(x, [m.log10(res_c)], axis=0)
                y = np.append(y, [m.log10(point_c)], axis=0)

            #countinue loop
            res_c = m.floor(res_c * growth_rate) + 1
            #reset point counter
            point_c = 0

        else:
            #for any resolution greater than 'res_tile' (user specified parameter), the complex plane is "tiled" res_tile x res_tile sections, then the same calculations are performed on these tiles.
            #if res_tile = 5000, and res = 10000, then tile_stop will be 2.
            res_c = res_tile
            point_c = 0

            #this loop contorls the amount of tiles needed for a given resolution.
            while tile_stop_c <= tile_stop:
                #update user
                print("Next Resolution: ", (res_c*tile_stop_c), "x", (res_c*tile_stop_c), ";Testing ", (res_c*tile_stop_c) ** 2, " points")
                #tot_c records the total amount of tiles that have been calculated, soley for user update
                tot_c = 1
                #tile counters for x and y loop over the tile over the complex plane from -2 to 2 in both the real and imag axes
                tile_c_x = 0
                tile_c_y = 0

                #Since the complex plane has been broken up into tile_stop_c by tile_stop_c tiles, a double loop is needed to calculate the set over the whole complex plane.
                while tile_c_y < tile_stop_c:
                    while tile_c_x < tile_stop_c:
                        #update user
                        print("Calculating", res_tile, "x", res_tile,"tile", tot_c, "out of", tile_stop_c ** 2)
                        #Dynamically calculate iterations
                        iterations = int(res_tile*iter_scale)
                        #step tile calculates the offset of the tile from -2
                        step_tile = 4/tile_stop_c
                        #clear GPU memory
                        mempool = cp.get_default_memory_pool()
                        mempool.free_all_blocks()

                        #generate complex plane from -2 to 2 on both the real and imag axes
                        X = cp.linspace((-2+(step_tile*tile_c_x)), (-2+(step_tile*(tile_c_x+1))), num=res_tile).reshape((1, res_tile))
                        Y = cp.linspace((-2+(step_tile*tile_c_y)), (-2+(step_tile*(tile_c_y+1))), num=res_tile).reshape((res_tile, 1))
                        C = cp.tile(X, (res_tile, 1)) + 1j * cp.tile(Y, (1, res_tile))

                        Z = cp.zeros((res_tile, res_tile), dtype=complex)
                        M = cp.full((res_tile, res_tile), True, dtype=bool)

                        #calculate set
                        for i in tqdm(range(iterations)):
                            Z[M] = Z[M] * Z[M] + C[M]
                            M[cp.abs(Z) > 2] = False

                        #move M to cpu and calculate set size
                        M_cpu = cp.asnumpy(M)
                        #update user
                        print("Now finding the set")
                        #these nested loops loop over the logic set (which is now on the cpu) to find the amount of points in the boundary of the set
                        for i in tqdm(range(res_tile-1)):
                            for j in range(res_tile-1):
                                a = M_cpu[i, j]
                                b = M_cpu[(i + 1), j]
                                c = M_cpu[i, (j + 1)]
                                d = M_cpu[(i + 1), (j + 1)]
                                if (a != b) or (a != c) or (a != d) or (b != c) or (b != d) or (c != d):
                                    point_c = point_c + 1
                                else:
                                    pass

                        #countinue inner loop
                        tile_c_x = tile_c_x + 1
                        tot_c = tot_c + 1
                    #countinue outer loop
                    tile_c_y = tile_c_y + 1
                    tile_c_x = 0

                #outside of the doubly nested loop, record total plot resolution and total amount of points in the set
                x = np.append(x, [m.log10(res_c*tile_stop_c)], axis=0)
                y = np.append(y, [m.log10(point_c)], axis=0)
                #expontially iterate loop
                tile_stop_c = m.floor(tile_stop_c * growth_rate) + 1
                #reset point counter
                point_c = 0
            #exit main loop
            res_c = res

    #prepare x and y for linear regression
    x = x.reshape((-1,1))
    y = y.flatten()

    #calculate linear regression
    model = LinearRegression()
    model.fit(x, y)
    r_sq = model.score(x, y)

    #print results to user
    print('r^2:', r_sq)
    print("The Dimenionsality is:", model.coef_)
    print("The Scaling Factor is:", model.intercept_)

    #show loglog plot to user
    plt.plot(x, y, 'ro')
    plt.axis([0,10,0,10])
    plt.show()
