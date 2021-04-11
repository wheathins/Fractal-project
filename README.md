# Fractal-project
This is the code for an extracurricular project in the class 'logic and sets' taught by Dr. Doug LaFountain. The code uses either your CPU or Nvidia GPU to calculate the fractal dimension of either the Mandelbrot Set or the Boundary of the Mandelbrot set. The code can also generate plots of the Set or Boundary. 

To run the code, you need the following packages installed for the CPU version:
-Numpy
-Math
-Tqdm
-sklearn
-matplotlib

For the the GPU versions, you will also need:
-Cupy (CUDA accelerated version of Numpy)
-Necessary Drivers, if you need help configuring a CUDA enviroment see: https://www.tensorflow.org/install/gpu

In the run.py file, you will see access to all the different programs:
-mandelbrot_set_gpu(res_c, res, res_tile, iter_scale, growth_rate) calculates the fractal dimension of the Mandelbrot Set given the parameters:
  res_c- starting resolution, set between 100-500
  res - final resolution, set between 2500-10000 can be bigger but requires lots of time and memory
  res_tile - after this resolution, the complex plane from -2 to 2 on the real line and from -2i and 2i on the imaginary line will be tiled in sqaures of resolution res_tile, this is to to allow for large final resolutions, set between 2500-10000 depending on memory
  iter_scale - the amount of iterations for a given rsolutions, the amount of iterations = iter_scale * current_resolution, set between 1 and 5
  growth_rate - we will test logrithmyically spaced resolutions, so the next resolution tested = current resolution * growth_rate, set between 1.05 and 1.5

mandelbrot_boundary_gpu(res_c, res, res_tile, iter_scale, growth_rate) calculates the fractal dimension of the boundary of the mandelbrot set, parameters are the same as before. 

mandelbrot_set_cpu and mandelbrot_boundary_cpu have the same functions as their GPU counterparts, but only use CPU instead and don't require drivers

mandelbrot_plot_gpu and mandelbrot_plot_cpu both plot the mandelbrot set, but only differ in that they use the GPU and CPU respectively. The arguments are resolution and iterations. 
