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
import cv2

im = files[0].im
plt.imshow(files[0].im)
plt.show()

#img = (255 * skimage.color.rgb2gray(im)).astype('uint8')[:380]
#%%
import skimage.feature
import time
orb = skimage.feature.ORB(n_keypoints = 200, n_scales = 1)
tmp = time.time()
#features = orb.detect_and_extract(img)
print time.time() - tmp

brief = skimage.feature.BRIEF(patch_size = 9)

#brief = cv2.DescriptorExtractor_create("BRIEF")
def im2g(im):
    img = cv2.cvtColor(im[:380], cv2.COLOR_RGB2GRAY)
    img = cv2.equalizeHist(img)

    return img

def get_harris(img, N):
    corners = cv2.cornerHarris(img, 2, 3, 0.05)
    peaks = skimage.feature.corner_peaks(corners)
    values = [corners[i, j] for i, j in peaks]
    peaks = peaks[numpy.argsort(values)[-N:]]

    return peaks

import sklearn.neighbors
nn = sklearn.neighbors.NearestNeighbors(2)

tmp = time.time()
im1 = im2g(files[10].im)
im2 = im2g(files[15].im)
peaks1 = get_harris(im1, 200)
peaks2 = get_harris(im2, 200)
nn.fit(peaks1)
stuff = brief.extract(im1, peaks1)
desc1 = brief.descriptors
stuff3 = brief.extract(im2, peaks2)
desc2 = brief.descriptors
print time.time() - tmp
#%%
ys = []
xs = []
for (y0, x0), (y1, x1) in zip(*stuff):#
    xs.append(x0)
    xs.append(x1)
    ys.append(y0)
    ys.append(y1)
    #print x0, x1, y0, y1
    plt.plot((x0, x1), (y0, y1))
plt.show()

plt.hist(xs, alpha = 0.5)
plt.hist(ys, alpha = 0.5)
plt.show()
#%%
reload(skimage.feature.brief)
#%%
pairs = set()
dxs = []
dys = []
for i, others in enumerate(nn.radius_neighbors(peaks2, 64.0, return_distance = False)):
    dists = [numpy.logical_xor(desc1[o], desc2[i]).sum() for o in others]

    if i == 0 or i == 30:
        print i, others, dists

    idxs = numpy.argsort(dists)

    if float(dists[idxs[0]]) / (dists[idxs[1]] + 0.001) < 0.6:
        #dxs.append(peaks2[i] - peaks1[o])
        #print peaks1[o] - peaks2[i]
        pairs.add((others[idxs[0]], i))
        #pairs.add((i, others[idxs[1]]))
#%%
import pyximport
pyximport.install(reload_support = True)

import brief
reload(brief)

class BRIEF(object):
    def __init__(self, N, k = 32, g = 1.0):
        self.N = N
        self.g = g

        pts = []

        while len(pts) < 2 * N:
            i = int(numpy.random.randn() * k / 5.0)
            j = int(numpy.random.randn() * k / 5.0)

            if i > -k // 2 and i < k // 2 and j > -k // 2 and j < k // 2:
                pts.append((i, j))

        self.pts1 = numpy.array(pts[:N], dtype = 'int')
        self.pts2 = numpy.array(pts[len(self.pts1):], dtype = 'int')

    def process(self, im, kp):
        im = cv2.GaussianBlur(im, (0, 0), self.g)

        return brief.process_(self.N, self.pts1, self.pts2, im, kp)

    def process_reference(self, im, kp):
        im = cv2.GaussianBlur(im, (0, 0), self.g)

        bits = numpy.zeros((len(kp), self.N), dtype = 'uint8')

        for i, c in enumerate(kp):
            for j in range(self.N):
                y1 = c[0] + self.pts1[j][0]
                x1 = c[1] + self.pts1[j][1]

                y2 = c[0] + self.pts2[j][0]
                x2 = c[1] + self.pts2[j][1]

                if y1 > 0 and y1 < im.shape[0] and y2 > 0 and y2 < im.shape[0] and x1 > 0 and x1 < im.shape[1] and x2 > 0 and x2 < im.shape[1]:
                    #print 'hi', im[y1, x1] < im[y2, x2]
                    bits[i, j] = int(im[y1, x1] > im[y2, x2])

                    #print c[0], c[1], y1, x1, y2, x2, self.pts1[j], self.pts2[j]

        return bits

brf = BRIEF(256, 16, g = 1.0)

tmp = time.time()
bits1 = brf.process(im1, peaks1)
print time.time() - tmp

tmp = time.time()
bits2 = brf.process_reference(im1, peaks1)
print time.time() - tmp

ys = []
xs = []
#for (y0, x0), (y1, x1) in zip(*stuff):#
for (y0, x0), (y1, x1) in zip(brf.pts1, brf.pts2):#*stuff):#
    xs.append(x0)
    xs.append(x1)
    ys.append(y0)
    ys.append(y1)
    #print x0, x1, y0, y1
    plt.plot((x0, x1), (y0, y1))
plt.show()

plt.hist(xs, alpha = 0.5)
plt.hist(ys, alpha = 0.5)
plt.show()
#%%
desc1 = brf.process(im1, peaks1)
desc2 = brf.process(im2, peaks2)
#%%
N = desc1.shape[0]

tmp = time.time()
distances = numpy.zeros((N, N))
hamming = numpy.zeros((N, N))

for i in range(N):
    for j in range(N):
        distances[i, j] = numpy.linalg.norm(peaks1[i] - peaks2[j])
        hamming[i, j] = (desc1[i] == desc2[j]).sum()

pairs = []

for i in range(N):
    j1 = -1
    j2 = -1
    max1 = 0
    max2 = 0
    for j in range(N):
        if distances[i, j] < 64.0:
            if hamming[i, j] > max1:
                max2 = max1
                max1 = hamming[i, j]

                j2 = j1
                j1 = j

    if max1 > 0 and max2 > 0:
        if float(max2) / max1 > 0.80:
            pairs.append((i, j1))

    #print i, j1, j2, max1, max2
print time.time() - tmp

#matches = set()
#for pair in pairs:
#    if (pair[1], pair[0]) in pairs:
#        matches.add(pair)
#dxs = numpy.array(dxs)
dxs = []
dys = []
#matches = skimage.feature.match_descriptors(desc1, desc2, metric = 'hamming', cross_check = True)
print "Open CV", time.time() - tmp
plt.imshow(im1, cmap = plt.cm.gray)
for i, j in list(pairs):
    x0, y0 = peaks1[i]
    x1, y1 = peaks2[j]

    dxs.append(x1 - x0)
    dys.append(y1 - y0)

    #if numpy.sqrt((x0 - x1)**2 + (y0 - y1)**2) > 100.0:
    #    continue

    plt.plot([y0, y1], [x0, x1], 'r-')
#plt.show()
plt.plot(peaks1[:, 1], peaks1[:, 0], '*w')
plt.gcf().set_size_inches((12, 12))
plt.show()

print numpy.median([dxs, dys], axis = 1)

#%%
plt.imshow(im1, cmap = plt.cm.gray)
plt.show()

plt.imshow(, cmap = plt.cm.gray)
plt.show()
#%%
tmp = time.time()
skimage.filters.gaussian(img, 1.0)
print time.time() - tmp
#%%
tmp = time.time()
x = 1.0
for i in range(40000):
    x = numpy.sqrt(x)
print time.time() - tmp
#%%

plt.plot(dxs, dys, '*')
#plt.imshow(numpy.log(-corners))
#plt.colorbar()
#plt.show()
#%%
import seaborn as sns
sns.jointplot(x=dxs, y=dys, kind="kde")
#%%
img = (255 * skimage.color.rgb2gray(im)).astype('uint8')[:380]
img = cv2.equalizeHist(img)
tmp = time.time()
feats = skimage.feature.corner_harris(img)
print time.time() - tmp
tmp = time.time()
peaks = skimage.feature.corner_peaks(feats)
plt.imshow(img, cmap = plt.cm.gray)
plt.plot(peaks[:, 1], peaks[:, 0], 'w*')
plt.gcf().set_size_inches((12, 12))
plt.show()
print time.time() - tmp
tmp = time.time()
briefs = brief.extract(img, peaks)
print time.time() - tmp
