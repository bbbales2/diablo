#%%
import sys
import os
import re
#import skimage.io, skimage.transform
import numpy
import bisect
import json
import argparse
import imageio
import pickle
import scipy
import skimage.filters
os.chdir('/home/bbales2/diablo')

import matplotlib.pyplot as plt

import skimage.io

import pyximport
pyximport.install(reload_support = True)
import bgalign

class File(object):
    def __init__(self, path = None, im = None):
        self.path = path
        self.im = im

files = []

if True:
    vid = imageio.get_reader('loops.mp4', 'ffmpeg')

    W, H = vid.get_meta_data()['size']

    F = vid.get_length()
    for i, frame in enumerate(vid):
        files.append(File('{0}, frame = {1}'.format('loops.mp4', i), frame))

        print "Reading frame {0} / {1}".format(i, F)

#%%

def fftalign(im1, im2, resolution = 1):
    f1 = skimage.color.rgb2gray(im1[::resolution, ::resolution])
    f2 = skimage.color.rgb2gray(im2[::resolution, ::resolution])

    F1 = numpy.fft.fft2(f1)
    F2 = numpy.fft.fft2(f2)

    Ftmp = F1 * F2.conj()
    #Ftmp /= numpy.linalg.norm(Ftmp)

    f = numpy.fft.fftshift(numpy.real(numpy.fft.ifft2(Ftmp)))
    loc = numpy.unravel_index(f.argmax(), f.shape)
    return numpy.linalg.norm(Ftmp), numpy.array([1, 1]) * resolution * (loc - numpy.array((f1.shape[0] / 2, f1.shape[1] / 2)))
#%%
import time

x = 0
y = 0

xs = [x]
ys = [y]

for f1, f2 in zip(files[:-1], files[1:]):
    tmp = time.time()
    #result, (dy, dx) = bgalign.offset(32, f1.im[:384], f2.im[:384])
    S, (dy, dx) = fftalign(f1.im[:384], f2.im[:384], 2)
    print time.time() - tmp

    #plt.imshow(result)
    #plt.show()

    x += dx
    y += dy

    xs.append(x)
    ys.append(y)
#%%
skimage.io.imsave("/home/bbales2/diablo/align.png", files[0].im)
