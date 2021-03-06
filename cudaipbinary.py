#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:02:32 2021

@author: Madain Perez
binariza una imagen
"""
# maxthreads=1024
import pycuda.autoinit
import pycuda.driver as drv
import numpy
from PIL import Image, ImageFilter
from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void binariza(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  if (b[i]>a[0]) {
    dest[i] = 255;
    }
    else {
    dest[i] = 0;
    }
}
""")
image=Image.open('tsukubal.png').convert('L')
image.show()
width, height=image.size
data=numpy.asarray(image).astype(numpy.float32)
negative = mod.get_function("binariza")
threshold=numpy.array([100]).astype(numpy.float32)
dest = numpy.array([range(width)]).astype(numpy.float32)
data_dest=numpy.zeros_like(data)
data_dest_normal=numpy.zeros_like(data)
for h in range(height):
        row=data[h,:]
        negative(drv.Out(dest), drv.In(threshold), drv.In(row), block=(width,1,1), grid=(1,1))
        data_dest[h,:]=dest
im = Image.fromarray(numpy.uint8(data_dest))
for h in range(height):
        row=data[h,:]
        data_dest_normal[h,:]=255-row
im_normal = Image.fromarray(numpy.uint8(data_dest_normal))
im.show()
#im_normal.show()

