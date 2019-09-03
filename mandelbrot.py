import numpy as np
import quaternion
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




def quat_mandelbrot(res_q, iterations_q):
    res_c_q = 2
    point_c_q = 0

    while res_c_q <= res_q:
        r_c_q = 0
        i_c_q = 0
        j_c_q = 0
        k_c_q = 0
        step_q = 4/res_c_q
        iter_c_q = 1
        z_q = 0

        while k_c_q <= res_c_q:
            while j_c_q <= res_c_q:
                while i_c_q <= res_c_q:
                    while r_c_q <= res_c_q:
                        z = 0
                        r_q = (2-(step_q*r_c_q))
                        i_q = (2-(step_q*i_c_q))
                        j_q = (2-(step_q*j_c_q))
                        k_q = (2-(step_q*k_c_q))
                        quat = np.quaternion(r_q, i_q, j_q, k_q)

                        for n in range(iterations_q):
                            z_q = z_q*z_q + quat
                            #print(quat, n, np.quaternion.abs(z_q))
                            if np.quaternion.abs(z_q) >= 2:
                                print(np.quaternion.abs(z_q) >= 2)
                                break

                            elif np.quaternion.abs(z_q) < 2 and n == iterations_q - 1:
                                print(np.quaternion.abs(z_q) < 2 and n == iterations_q - 1)
                                point_c_q = point_c_q + 1
                                print("Found one!", res_c_q, np.quaternion.abs(z_q))
                                break

                            else:
                                pass

                        r_c_q = r_c_q + 1

                    r_c_q = 0
                    i_c_q = i_c_q + 1

                i_c_q = 0
                j_c_q = j_c_q + 1

            j_c_q = 0
            k_c_q = k_c_q + 1

        if res_c_q == 2:
            reg_array_q = np.array([[(res_c_q) ** 4, point_c_q]])
            reg_array_q = reg_array_q.astype('int32')

        else:
            reg_array_q = np.append(reg_array_q, [[(res_c_q) ** 4, point_c_q]], axis = 0)

        res_c_q = res_c_q + 1
        point_c_q = 0


    print(reg_array_q)
    print(reg_array_q.shape)

quat_mandelbrot(5, 1)
