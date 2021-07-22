import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
mod = SourceModule("""  
    __global__ void square(float *d_a)  
    {    
        int idx = threadIdx.x + threadIdx.y*5;    
        d_a[idx] = d_a[idx]*d_a[idx];  
    }
""")
start = drv.Event()
end=drv.Event()
h_a = numpy.random.randint(1,5,(5, 5))
h_a = h_a.astype(numpy.float32)
h_b=h_a.copy()
start.record()
start.synchronize()
square = mod.get_function("square")
# se usa cuando la entrada y la salida es la misma, facilitando el codigo y el rendimiento
square(drv.InOut(h_a), block=(5, 5, 1))
end.record()
end.synchronize()
print("Square with InOut:")
print(h_a)
secs = start.time_till(end)*1e-3
print("Time of Squaring on GPU with inout")
print("%fs" % (secs))
