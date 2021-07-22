import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
mod = SourceModule("""  
    #include <stdio.h>  
    __global__ void myfirst_kernel()  
    {    
        printf("Hello,PyCUDA!!!");  
        }
""")
function = mod.get_function("myfirst_kernel")
function(block=(1,1,1))
