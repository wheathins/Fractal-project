import numpy as np
import quaternion as q
import tensorflow as tf
import cmath as cm

def mandelbrot(res, iterations):
    res_c = 2
    point_c = 0

    while res_c <= res:
        real_c = 0
        imag_c = 0
        step = 4/res_c
        iter_c = 1
        z = 0

        while imag_c <= res_c:
            while real_c <= res_c:
                z = 0
                real = (2-(step*real_c))
                imag = (2-(step*imag_c))
                con = complex(real, imag)

                for n in range(iterations):
                    z = z*z + con
                    if abs(z) >= 2:
                        break

                    elif abs(z) < 2 and n == iterations - 1:
                        point_c = point_c + 1
                        print("Found one!", res_c, abs(z))
                        break

                    else:
                        pass

                real_c = real_c + 1

            real_c = 0
            imag_c = imag_c + 1

        if res_c == 2:
            reg_array = np.array([[(res_c) ** 2, point_c]])
            reg_array = reg_array.astype('int32')

        else:
            reg_array = np.append(reg_array, [[(res_c) ** 2, point_c]], axis = 0)

        res_c = res_c + 1
        point_c = 0


    print(reg_array)
    print(reg_array.shape)


mandelbrot(100, 1000000)
