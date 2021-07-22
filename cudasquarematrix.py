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
d_a = drv.mem_alloc(h_a.size * h_a.dtype.itemsize)
drv.memcpy_htod(d_a, h_a)
square = mod.get_function("square")
square(d_a, block=(5, 5, 1), grid=(1, 1), shared=0)
h_result = numpy.empty_like(h_a)
drv.memcpy_dtoh(h_result, d_a)
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("Time of Squaring on GPU without inout")
print("%fs" % (secs))
print("original array:")
print(h_a)
print("Square with kernel:")
print(h_result)
