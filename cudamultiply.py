#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:02:32 2021

@author: nautica
"""

import pycuda.autoinit
import pycuda.driver as drv
import numpy
from PIL import Image, ImageFilter
from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void multiply_constante(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[0] * b[i];
}
""")
image=Image.open('tsukubal.png')
image.show()
multiply_them = mod.get_function("multiply_constante")
valores=numpy.array([range(400)]).astype(numpy.float32)
multiplicando=numpy.array([3]).astype(numpy.float32)
print(multiplicando)
a = numpy.random.randn(1).astype(numpy.float32)
b = numpy.random.randn(400).astype(numpy.float32)

dest = numpy.zeros_like(valores)
multiply_them(
        drv.Out(dest), drv.In(multiplicando), drv.In(valores),
        block=(400,1,1), grid=(1,1))

print(dest)
