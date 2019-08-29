import numpy as np
import quaternion as q
import tensorflow as tf
import cmath as cm

def mandelbrot(res, iterations):
    res_c = 2

    while res_c <= res:
        #create numpy array to store scaling factor and mandelbrot point amount
        x_c = 0
        y_c = 0
        step = 4/res_c
        iter_c = 1
        z = 0

        reg_array = np.array([])
        reg_array = reg_array.astype('int32')
        point_c = 0

        while y_c <= res_c:

            while x_c <= res_c:

                real = (2-(step*x_c))
                imag = (2-(step*y_c))
                con = complex(real, imag)

                while iter_c <= iterations and abs(con) < 2:
                    z = (z ** 2) + con
                    iter_c = iter_c + 1

                    if iter_c == iterations:
                        point_c = point_c + 1

                    else:
                        pass

                x_c = x_c + 1
                z = 0

            x_c = 0
            y_c = y_c + 1

        np.insert(reg_array, reg_array.shape, [res_c, point_c])
        res_c = res_c + 1

    print(point_c)
    print(reg_array)
    print(reg_array.shape)


mandelbrot(20, 10)
