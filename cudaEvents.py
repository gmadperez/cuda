import pycuda.autoinit
import pycuda.driver as drv
import numpy
import time
import math
from pycuda.compiler import SourceModule
N = 1000000
# como el numero de elementos es alto, se calcula el indice del thread usando thread and block id
mod = SourceModule("""                   
    __global__ 
    void add_num(float *d_result, float *d_a, float *d_b,int N)
    { 
        int tid = threadIdx.x + blockIdx.x * blockDim.x;   
        while (tid < N)  
        {    
            d_result[tid] = d_a[tid] + d_b[tid];    
            tid = tid + blockDim.x * gridDim.x;  
        }
    }
""")
# se crean los eventos para medir los timings
start = drv.Event()
end=drv.Event()
add_num = mod.get_function("add_num")
h_a = numpy.random.randn(N).astype(numpy.float32)
h_b = numpy.random.randn(N).astype(numpy.float32)
h_result = numpy.zeros_like(h_a)
h_result1 = numpy.zeros_like(h_a)
# cada bloque soporta 1024 threads
n_blocks = math.ceil((N/1024))
# se almacena el tiempo
start.record()
add_num(
    drv.Out(h_result), drv.In(h_a), drv.In(h_b),numpy.uint32(N),
    block=(1024,1,1), grid=(n_blocks,1))
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("Addition of %d element of GPU"%N)
print("%fs" % (secs))
start = time.time()
for i in range(0,N):
    h_result1[i] = h_a[i] +h_b[i]
end = time.time()
print("Addition of %d element of CPU"%N)
print(end-start,"s")
