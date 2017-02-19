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

class File(object):
    def __init__(self, path = None, im = None):
        self.path = path
        self.im = im

files = []

if True:
    vid = imageio.get_reader('follow.mp4', 'ffmpeg')

    W, H = vid.get_meta_data()['size']

    F = vid.get_length()
    for i, frame in enumerate(vid):
        if i > 200:
            break

        files.append(File('{0}, frame = {1}'.format('follow.mp4', i), frame))

        print "Reading frame {0} / {1}".format(i, F)

#%%
Fs = []

def forward(im):
    im = skimage.color.rgb2gray(im)[:384, :]
    im = skimage.filters.gaussian(im, 4.0)[::8, ::8]
    return numpy.fft.fft2(im)

for i, f_ in enumerate(files):
    #f = skimage.color.rgb2gray(f_.im)[:384:2, ::2]
    Fs.append(forward(f_.im))
    print "Processing frame {0} / {1}".format(i, len(files))

#%%

# Taken from http://stackoverflow.com/questions/34968722/softmax-function-python
def softmax(x):
    return numpy.exp(x) / numpy.sum(numpy.exp(x), axis=0)

def whiten(vec):
    vec = (vec - numpy.mean(vec)) / numpy.std(vec)
    return vec

def mle_loc(im):
    F = forward(im)
    dists = []
    offsets = []
    #tmp = time.time()
    for i, F2 in enumerate(Fs[:]):
        Ftmp = F * F2.conj()

        Ftmp /= numpy.linalg.norm(Ftmp)

        f = numpy.fft.fftshift(numpy.real(numpy.fft.ifft2(Ftmp)))

        loc = numpy.unravel_index(f.argmax(), f.shape)

        dists.append(f[loc])
        loc = loc# - numpy.array((F.shape[0] / 2, F.shape[1] / 2))
        offsets.append(loc)

    #print time.time() - tmp
        #print i, loc

    r = numpy.array([numpy.linalg.norm(d) for d in offsets])

    dists2 = softmax(whiten(dists))
    r2 = softmax(-r / 100.0)
    p = dists2 * r2
    p /= sum(p)

    p[numpy.logical_not(p > max(p) * 0.2)] = 0.0
    p /= sum(p)

    l = numpy.round(sum(p * numpy.arange(len(p)))).astype('int')
    
    F2 = Fs[l]
    
    Ftmp = F * F2.conj()

    Ftmp /= numpy.linalg.norm(Ftmp)

    f = numpy.fft.fftshift(numpy.real(numpy.fft.ifft2(Ftmp)))

    #return dists2#p
    offsets = numpy.array(offsets) * 8
    #print p
    #l = (numpy.argmax(dists2) + sum(p * numpy.arange(-len(dists2) / 2, len(dists2) / 2))) * 10.0
    #print p * numpy.arange(len(dists2))
    offset = numpy.array([sum(p * offsets[:, 0]), sum(p * offsets[:, 1])])
    #l = numpy.argmax(dists2 * r2)
    return l, offset, f

#%%
#import time

#tmp = time.time()
#mle_loc(files[10].im)
#print time.time() - tmp


#p = mle_loc(files[12].im)
#print p

#plt.plot(p)

#%%

#import sklearn.mixture


#sums = numpy.cumsum(p.flatten())
#ps = []
#for i in range(50):
#    idx = bisect.bisect_left(sums, numpy.random.random() * sums[-1])
#    ps.append(numpy.unravel_index(idx, p.shape)[0])

#ps = numpy.array(ps).reshape(-1, 1)

#gmm = sklearn.mixture.GMM(2)
#gmm.fit(ps)
#print gmm.means_ * 5
#print gmm.weights_
#%%
#files[200].im

def kf(z, x_, P_, Q, R):
    P = P_ + Q
    y = z - x_
    S = P + R
    K = P / S
    x = x_ + K * y
    P = (1 - K) * P
    return x, P

if False:
    x = 0
    P = 10.0
    pos = 0
    for i in range(200):
        loc, offset = mle_loc(files[i].im)

        x, P = kf(loc, x, P, 2.0, 2.0)

        print i, x, P

    #r3 = softmax(dists * r)
    #plt.plot(dists2, '*')
    #plt.plot(dists2 * r2 / max(dists2 * r2), 'ro')
    #plt.show()

#%%
    tmp = dists2 * r2 / max(dists2 * r2)
#%%
if False:
    import skimage.feature
    import time

    for f1, f2 in [(files[200], files[200])]:#zip(files[:-1], files[1:]):
        f1 = skimage.color.rgb2gray(f1.im)[:380:2, ::2]
        f2 = skimage.color.rgb2gray(f2.im)[:380:2, ::2]

        tmp = time.time()
        F1 = numpy.fft.fft2(f1)
        F2 = numpy.fft.fft2(f2)

        Ftmp = F1 * F2.conj()
        #Ftmp /= numpy.linalg.norm(Ftmp)

        f = numpy.fft.fftshift(numpy.real(numpy.fft.ifft2(Ftmp)))
        print 'm1', time.time() - tmp
        loc = numpy.unravel_index(f.argmax(), f.shape)
        print loc - numpy.array((f1.shape[0] / 2, f1.shape[1] / 2))
        plt.imshow(f)
        plt.colorbar()
        plt.show()
        #plt.imshow(numpy.imag(f))
        #plt.colorbar()
        #plt.show()

        #plt.imshow(f1)
        #plt.show()
        #plt.imshow(f2)
        #plt.show()

        tmp = time.time()
        shifts, error, phase = skimage.feature.register_translation(f1, f2)
        #print 'm2', time.time() - tmp

        print shifts, error
        print ""

        #1/0
