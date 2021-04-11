import numpy
import math
import quaternion as q

def mandelbrot_quaternion_cpu(res_c, res, res_tile, iter_scale, growth_rate):
    while res_c <= res:
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

        X = np.linspace(-2, 2, num=res_c).reshape((1, res_c))
        Y = np.linspace(-2, 2, num=res_c).reshape((res_c, 1))
        C = np.tile(X, (res_c, 1)) + 1j * np.tile(Y, (1, res_c))

        #Z matrix of zeros from mandelbrot set definition
        Z = np.zeros((res_c, res_c), dtype=complex)
        #M is matrix of bools used to store whether or not a given point in c is in the set or not
        #if the abs value of a point of Z is greater than 2, that entry in M is made false, halting multiplication
        M = np.full((res_c, res_c), True, dtype=bool)

        #perform calculations
        for i in tqdm(range(iterations)):
            Z[M] = Z[M] * Z[M] + C[M]
            M[np.abs(Z) > 2] = False
