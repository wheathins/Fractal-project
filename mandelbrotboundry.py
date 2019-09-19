#package import
#Numpy and sklearn aren't native to python, so you will need to install them
import numpy as np
import math as m
import cmath as cm
from sklearn.linear_model import LinearRegression




def loopy_mandelbrot_boundary(res_c, res, iterations):
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
                
                for n in range(iterations):
                    z1 = z1*z1 + con1
                    if abs(z1) >= 2:
                        out_set = out_set + 1
                        break
                    
                    elif abs(z1) < 2 and n == iterations - 1:
                        in_set = in_set + 1
                        break

                    else:
                        pass
                    
                for n in range(iterations):
                    z2 = z2*z2 + con2
                    if abs(z2) >= 2:
                        out_set = out_set + 1
                        break
                    
                    elif abs(z2) < 2 and n == iterations - 1:
                        in_set = in_set + 1
                        break

                    else:
                        pass
                    
                for n in range(iterations):
                    z3 = z3*z3 + con3
                    if abs(z3) >= 2:
                        out_set = out_set + 1
                        break
                    
                    elif abs(z3) < 2 and n == iterations - 1:
                        in_set = in_set + 1
                        break

                    else:
                        pass
                    
                for n in range(iterations):
                    z4 = z4*z4 + con4
                    if abs(z4) >= 2:
                        out_set = out_set + 1
                        break
                    
                    elif abs(z4) < 2 and n == iterations - 1:
                        in_set = in_set + 1
                        break

                    else:
                        pass
                
                
                print(in_set, out_set)    
                if out_set < 1 and in_set < 1:
                    point_c = point_c + 1
                    print("found one", res_c)
                    
                else:
                    pass
                    

                real_c = real_c + 1

            real_c = 0
            imag_c = imag_c + 1
            
        #the if condition creates the arrays used to store the scaling factor and number of points in the mandelbrot for each resolution iteration
        #the log is taken of the x var so it become linear
        if res_c == big:
            x = np.array([m.sqrt((res_c ** 2)/(big ** 2))])
            y = np.array([m.log10(point_c)])

        #the else condition adds the values the arrays created during the first iteration of the resolution loop
        else:
           x = np.append(x, [m.sqrt((res_c ** 2)/(big ** 2))], axis = 0)
           y = np.append(y, [m.log10(point_c)], axis=0)

        res_c = res_c + 1
        point_c = 0
        in_set = 0
        out_set = 0
        
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
    print("The equation is: y=", model.coef_, "x +", model.intercept_)
    #the scaling factor and scaling constant must be raised the 10 ^ to undo the log taken used to make the relationship linear
    scaling_factor = 10 ** model.coef_
    scaling_constant = 10 ** model.intercept_
    print("The Dimenionsality is:", scaling_factor)
    print("The Scaling Factor is:", scaling_constant)
    
