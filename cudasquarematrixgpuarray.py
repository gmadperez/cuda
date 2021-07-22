import pycuda.gpuarray as gpuarray
import numpy
import pycuda.driver as drv
start = drv.Event()
end=drv.Event()
start.record()
start.synchronize()
h_b = numpy.random.randint(1,5,(5, 5))
d_b = gpuarray.to_gpu(h_b.astype(numpy.float32))
h_result = (d_b**2).get()
end.record()
end.synchronize()
print("original array:")
print(h_b)
print("doubled with gpuarray:")
print(h_result)
secs = start.time_till(end)*1e-3
print("Time of Squaring on GPU with gpuarray")
print("%fs" % (secs))
