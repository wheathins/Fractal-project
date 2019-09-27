#package import
#Numpy and sklearn aren't native to python, so you will need to install them
import numpy as np
import math as m
import cupy as cp
import cmath as cm
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

#the growth_rate var controls the expontial scaling of resolutions tested, use 1 for a standard loop++ experience, otherwise use values between 1 and 2 to speed up time
def loopy_mandelbrot_boundary(res_c, res, iterations, growth_rate):
    #point_c is a var used to track the amount of points in the set for a given plot
    point_c = 0
    #big is a dummy var used to keep track of the initial plot resolution
    big = res_c

    #this first while loop increases the resolution of the mandelbrot set plot by one each iteration
    while res_c <= res:
        real_c = 0
        imag_c = 0
        step = 4/res_c
        z1 = 0
        z2 = 0
        z3 = 0
        z4 = 0

        in_set = 0
        out_set = 0

        #these two while loops over the complex plane to test all the points between -2 and 2 in the real axis and -2 and 2 on the imaginary axis
        while imag_c < res_c:
            while real_c < res_c:
                z1 = 0
                z2 = 0
                z3 = 0
                z4 = 0

                i = 1
                j = 1
                k = 1
                l = 1

                real1 = (2-(step*real_c))
                imag1 = (2-(step*imag_c))

                real2 = (2-((step+1)*real_c))
                imag2 = (2-(step*imag_c))

                real3 = (2-(step*real_c))
                imag3 = (2-((step+1)*imag_c))

                real4 = (2-((step+1)*real_c))
                imag4 = (2-((step+1)*imag_c))

                con1 = complex(real1, imag1)
                con2 = complex(real2, imag2)
                con3 = complex(real3, imag3)
                con4 = complex(real4, imag4)

                while i <= iterations and abs(z1) < 2:
                    z1 = z1*z1 + con1
                    i = i + 1
                    if abs(z1) < 2 and i == iterations:
                        print("in set", abs(z1))
                        in_set = in_set + 1

                    else:
                        pass
                if abs(z1) > 2:
                    out_set = out_set + 1

                while j <= iterations and abs(z2) < 2:
                    z2 = z2*z2 + con2
                    j = j + 1
                    if abs(z2) < 2 and j == iterations:
                        print("in set", abs(z2))
                        in_set = in_set + 1

                    else:
                        pass

                if abs(z2) > 2:
                    out_set = out_set + 1

                while k <= iterations and abs(z3) < 2:
                    z3 = z3*z3 + con3
                    k = k + 1
                    if abs(z3) < 2 and k == iterations:
                        print("in set", abs(z3))
                        in_set = in_set + 1

                    else:
                        pass

                if abs(z3) > 2:
                    out_set = out_set + 1

                while l <= iterations and abs(z4) < 2:
                    z1 = z1*z1 + con4
                    l = l + 1
                    if abs(z4) < 2 and l == iterations:
                        print("in set", abs(z4))
                        in_set = in_set + 1


                    else:
                        pass

                if abs(z4) > 2:
                    out_set = out_set + 1

                if out_set < 1 and in_set < 1:
                    print("Found one!", in_set, out_set)
                    point_c = point_c + 1
                    break

                else:
                    pass


                real_c = real_c + 1

            real_c = 0
            imag_c = imag_c + 1

        #the if condition creates the arrays used to store the scaling factor and number of points in the mandelbrot for each resolution iteration
        #the log is taken of the x var so it become linear
        if res_c == big:
            x = np.array([m.log10(res_c)])
            y = np.array([m.log10(point_c)])

        #the else condition adds the values the arrays created during the first iteration of the resolution loop
        else:
           x = np.append(x, [m.log10(res_c)], axis = 0)
           y = np.append(y, [m.log10(point_c)], axis=0)

        res_c = m.floor(res_c * growth_rate) + 1
        point_c = 0
        in_set = 0
        out_set = 0

    #reshape the arrays so the sklearn library can perform a linear regression
    x = x.reshape((-1,1))
    y = y.flatten()
    print(x.shape, y.shape)
    print(x)
    print(y)

    #call and perform a linear regression on the x and y arrays
    model = LinearRegression()
    model.fit(x, y)
    r_sq = model.score(x, y)
    #print the results
    #the slope of the line is the scaling factor, aka, the hausdorff dimension
    #the intercept is the is the scaling constant, an example for a scaling factor is pi. Pi is the scaling factor a circle.
    print('r^2:', r_sq)
    print("The equation is: y=", model.coef_, "x +", model.intercept_)
    #the scaling factor and scaling constant must be raised the 10 ^ to undo the log taken used to make the relationship linear
    scaling_factor = model.coef_
    scaling_constant = model.intercept_
    print("The Dimenionsality is:", scaling_factor)
    print("The Scaling Factor is:", scaling_constant)


def parallelized_mandelbrot_boundary(res_c, res, iterations, growth_rate):
    point_c = 1
    big = res_c

    while res_c <= res:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()

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

        for i in  tqdm(range(res_c-1)):
            for j in range(res_c-1):
                a = M_cpu[i, j]
                b = M_cpu[(i + 1), j]
                c = M_cpu[i, (j + 1)]
                d = M_cpu[(i + 1), (j + 1)]
                if (a != b != c != d):
                    point_c = point_c + 1

                else:
                    pass

        if res_c == big:
            x = np.array([m.log2(res_c)])
            y = np.array([m.log2(point_c)])


        else:
            x = np.append(x, [m.log2(res_c)], axis=0)
            y = np.append(y, [m.log2(point_c)], axis=0)


        res_c = m.floor(res_c * growth_rate) + 1
        point_c = 0

    x = x.reshape((-1,1))
    y = y.flatten()

    #call and perform a linear regression on the x and y arrays
    model = LinearRegression()
    model.fit(x, y)
    r_sq = model.score(x, y)
    #print the results
    #the slope of the line is the scaling factor, aka, the hausdorff dimension
    #the intercept is the is the scaling constant, an example for a scaling factor is pi. Pi is the scaling factor a circle.
    print("r^2:", r_sq)
    print("The Dimenionsality is:", model.coef_)
    print("The Scaling Factor is:", model.intercept_)

parallelized_mandelbrot_boundary(50, 5000, 10000, 1.05)
