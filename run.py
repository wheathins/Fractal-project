from set_dim_calc_gpu import mandelbrot_set_gpu
from boundary_dim_calc_gpu import mandelbrot_boundary_gpu
from set_dim_calc_cpu import mandelbrot_set_cpu
from boundary_dim_calc_cpu import mandelbrot_boundary_cpu
from mandelbrot_set_plot_cpu import mandelbrot_plot_cpu
from mandelbrot_set_plot_gpu import mandelbrot_plot_gpu

mandelbrot_set_gpu(100, 5000, 5000, 5, 1.2)
#mandelbrot_boundary_gpu(50, 5000, 5000, 5, 1.1)
#mandelbrot_set_cpu(50, 1000, 500, 5, 1.1)
#mandelbrot_boundary_cpu(50, 1000, 500, 5, 1.1)

#mandelbrot_plot_gpu(10000, 100)
#mandelbrot_plot_cpu(10000, 64)
