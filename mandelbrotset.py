#package import
#Numpy and sklearn aren't native to python, so you will need to install them
import numpy as np
import math as m
import cmath as cm
from sklearn.linear_model import LinearRegression

#the function loopy mandelbrot uses loops(very slow!) to calculate the hasudorff dimension of the mandelbrot set
#Recommended values for the loopy_mandelbrot_set is res_c = 10, res > 150, iterations = 10000 for larger res values, 100000 for smaller res values
#It works better for larger iteration values and smaller res values
def loopy_mandelbrot_set(res_c, res, iterations):
    #point_c is a var used to track the amount of points in the set for a given plot
    point_c = 0
    #big is a dummy var used to keep track of the initial plot resolution
    big = res_c

    #this first while loop increases the resolution of the mandelbrot set plot by one each iteration
    while res_c <= res:
        print("Next Resolution: ", res_c, "x", res_c, ";Testing ", res_c ** 2, " points")
        real_c = 0
        imag_c = 0
        step = 4/res_c
        z = 0

        #these two while loops over the complex plane to test all the points between -2 and 2 in the real axis and -2 and 2 on the imaginary axis
        while imag_c <= res_c:
            while real_c <= res_c:
                z = 0
                real = (2-(step*real_c))
                imag = (2-(step*imag_c))
                con = complex(real, imag)

                #this for loop tests whether the points created in the loops above are in the mandelbrot set
                #the if, elif, else logic inside the loop adds points in the set to the point_c var
                for n in range(iterations):
                    z = z*z + con
                    if abs(z) >= 2:
                        break

                    elif abs(z) < 2 and n == iterations - 1:
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

        res_c = m.floor(res_c * 1.2)
        point_c = 0

    #reshape the arrays so the sklearn library can perform a linear regression
    x = x.reshape((-1,1))
    y = y.flatten()

    #call and perform a linear regression on the x and y arrays
    model = LinearRegression()
    model.fit(x, y)
    r_sq = model.score(x, y)
    #print the results
    #the slope of the line is the scaling factor, aka, the hausdorff dimension
    #the intercept is the is the scaling constant, an example for a scaling factor is pi. Pi is the scaling factor a circle.
    print('r^2:', r_sq)
    print("The Dimenionsality is:", model.coef_)
    print("The Scaling Factor is:", model.intercept_)



def paralleized_mandelbrot_set(res_c, res, iterations):
    print(res_c, res, iterations)

loopy_mandelbrot_set(10, 10000, 100)
