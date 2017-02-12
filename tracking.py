#%%
import sys, pygame
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
os.chdir('/home/bbales2/diablo')

import googlenet
import matplotlib.pyplot as plt

pygame.init()

import skimage.io
import tensorflow as tf

class File(object):
    def __init__(self, path = None, im = None):
        self.path = path
        self.im = im

files = []

if True:
    vid = imageio.get_reader('monsters.mp4', 'ffmpeg')

    W, H = vid.get_meta_data()['size']

    F = vid.get_length()
    for i, frame in enumerate(vid):
        #if i > 100:
        #    break

        files.append(File('{0}, frame = {1}'.format('monsters.mp4', i), frame))

        print "Reading frame {0} / {1}".format(i, F)

# Load up the neural network
#%%
print "Loading GoogleNet neural network"

sess = tf.Session()

tens = tf.placeholder(tf.float32, shape = [1, H, W, 3])

net = googlenet.GoogleNet({'data' : tens})

net.load('googlenet.tf', sess, ignore_missing = True)

target = [net.layers[name] for name in net.layers if name == 'pool5_7x7_s1'][0]

test = sess.run(target, feed_dict = { tens : numpy.zeros((1, H, W, 3)) })[0]

print "Neural network loaded"
#%%

with open('/home/bbales2/annotater/classifiers.pkl') as f:
    classifiers = pickle.load(f)

#%%
msg = ""

labels = classifiers.keys()

signals = []

for i, f in enumerate(files):
    frame = f.im
    hist = sess.run(target, feed_dict = { tens : frame.reshape(1, frame.shape[0], frame.shape[1], frame.shape[2]) })[0]
    hist = classifiers['fallen'].predict_proba(hist.reshape((-1, 1024)))[:, 1].reshape((H / 16, W / 16))
    #plt.imshow(f.im)
    #plt.title(i)
    #plt.show()

    signals.append(hist)
    print i

    #if i > 50:
    #    break
#%%
ims = [f.im for f in files]
#%%
with open('nn_output', 'w') as f:
    pickle.dump((signals, ims), f)
#%%
frame = files[10].im
hist = sess.run(target, feed_dict = { tens : frame.reshape(1, frame.shape[0], frame.shape[1], frame.shape[2]) })[0]
hist = classifiers['fallen'].predict_proba(hist.reshape((-1, 1024)))[:, 1].reshape((H / 16, W / 16))
#%%
skimage.transform.rescale
#%%
plt.imshow(hist[:, :], interpolation = 'NONE')
plt.colorbar()
plt.show()
#%%
import bisect
import time
import skimage.transform

samples_list = []
s = 4.0
for i in range(len(files)):
    hist = skimage.transform.rescale(signals[i], s) / s**2
    l = hist.sum()
    m = numpy.random.poisson(l)
    sums = numpy.cumsum(hist.flatten())
    xs = []
    for i in range(m):
        idx = bisect.bisect_left(sums, numpy.random.random() * l)
        xs.append(numpy.unravel_index(idx, hist.shape))

    xs = numpy.array(xs)

    #plt.imshow(hist, interpolation = 'NONE', cmap = plt.cm.gray)
    #plt.colorbar()
    #plt.plot(xs[:, 1], xs[:, 0], 'ro')
    #plt.show()

    samples_list.append(xs / s)
#%%
wk = []
mk = []
Pk = []

for hist, samples in zip(signals, samples_list[:20]):
    pSk = 0.99
    pDk = 0.99
    kk = 0.01
    Q = 1.0
    R = 1.0

    wkm1 = wk
    mkm1 = mk
    Pkm1 = Pk
    wkgkm1 = []
    mkgkm1 = []
    Pkgkm1 = []
    Sk = []
    Kk = []
    Pkgk = []

    J = len(wkm1)

    i = 0
    # Skip prediction for birth targets
    # Prediction for existing targets
    for j in range(J):
        i += 1

        wkgkm1.append(wkm1[j] * pSk)
        mkgkm1.append(mkm1[j])
        Pkgkm1.append(Pkm1[j] + Q)

    print wkgkm1

    # Add the generic birth process
    wkgkm1.append(0.1)
    mkgkm1.append(numpy.array([8.0, 8.0]))
    Pkgkm1.append(100.0)

    Jkgkm1 = len(wkgkm1)

    nkgkm1 = []
    # Construction of PHD update components
    for j in range(Jkgkm1):
        nkgkm1.append(mkgkm1[j])
        Sk.append(R + Pkgkm1[j])

        Kk.append(Pkgkm1[j] / Sk[j])
        Pkgk.append((1.0 - Kk[j]) * Pkgkm1[j])

    # Update
    wk = numpy.zeros(Jkgkm1)
    mk = numpy.zeros((Jkgkm1, 2))
    Pk = numpy.zeros(Jkgkm1)
    for j in range(Jkgkm1):
        wk[j] = (1 - pDk) * wkgkm1[j]
        mk[j] = mkgkm1[j]
        Pk[j] = Pkgkm1[j]

    l = 0

    def normal(z, mean, std):
        p = scipy.stats.norm.pdf(z, mean, std)
        return p[0] * p[1]

    wtt = [wk]
    mtt = [mk]
    Ptt = [Pk]
    for zl, z in enumerate(samples):
        #print z
        wt = numpy.zeros(Jkgkm1)
        mt = numpy.zeros((Jkgkm1, 2))
        Pt = numpy.zeros(Jkgkm1)
        for j in range(Jkgkm1):
            wt[j] = pDk * wkgkm1[j] * normal(z, nkgkm1[j], Sk[j])
            mt[j] = mkgkm1[j] + Kk[j] * (z - nkgkm1[j])
            Pt[j] = Pkgk[j]
        ws = wt.sum()

        wt = wt / (kk + ws)

        wtt.append(wt)
        mtt.append(mt)
        Ptt.append(Pt)

    #print wk
    #print mk

    wk = numpy.vstack(wtt).flatten()
    mk = numpy.vstack(mtt)
    Pk = numpy.vstack(Ptt).flatten()

    wt = []
    mt = []
    Pt = []
    U = 4.0
    I = set()
    for l in range(len(wk)):
        idxs = numpy.argsort(wk)

        j = None
        for i in idxs[::-1]:
            if i in I:
                continue
            else:
                j = i

        if j == None:
            break

        L = list()
        for i in range(len(mk)):
            if i not in I:
                if (mk[i] - mk[j]).dot(mk[i] - mk[j]) / Pk[j] < U:
                    L.append(i)

        #L = numpy.where((mk - mk[j]).dot((mk - mk[j]).transpose()) / Pk[j] < 2.0)

        wtt = sum(wk[L])
        mtt = numpy.zeros(2)
        for l in L:
            mtt += wk[l] * mk[l] / wtt

        Ptt = 0.0
        for l in L:
            Ptt += wk[l] * (Pk[l] + (mk[l] - mtt).dot(mk[l] - mtt)) / wtt

        wt.append(wtt)
        mt.append(mtt)
        Pt.append(Ptt)

        for l in L:
            I.add(l)

    idxs = numpy.argsort(wt)[-10:]

    wk = []
    mk = []
    Pk = []

    for i in idxs:
        wk.append(wt[i])
        mk.append(mt[i])
        Pk.append(Pt[i])

    for w, m, p in zip(wk, mk, Pk):
        print w, m, p
    print ""

    plt.imshow(hist, interpolation = 'NONE', cmap = plt.cm.gray)

    if len(mk) > 0:
        xs = []
        ys = []
        for i in range(len(wk)):
            if wk[i] > 0.2 * max(wk) and Pk[i] < 10:
                xs.append(mk[i][1])
                ys.append(mk[i][0])

        plt.colorbar()
        plt.plot(xs, ys, 'ro')
        plt.show()

    #wk = numpy.array(wk)
    #mk = numpy.array(mk)
    #Pk = numpy.array(Pk)

#%%

#%%
tmp1 = time.time()
u = numpy.random.random((L, 2)) * hist.shape
sig = 1.5
I = numpy.random.random(L)

def norm(x, u, sig):
    return numpy.exp(-0.5 * (x - u).dot(x - u) / (sig**2)) / (2 * numpy.pi * sig)

print u, I

for r in range(10):
    T = []
    for j in range(m):
        total = 0.0

        for l in range(L):
            total += norm(xs[j], u[l], sig)

        T.append(total)

    un = numpy.zeros(u.shape)
    for l in range(L):
        total = 0.0

        unum = 0.0
        signum = 0.0

        denom = 0.0

        for j in range(m):
            tmp = norm(xs[j], u[l], sig) / T[j]

            total += tmp
            unum += xs[j] * tmp
            signum += sum((xs[j] - u[l]) * (xs[j] - u[l])) * tmp

        un[l] = unum / total
        I[l] = total

#    total = 0.0
#    denom = 0.0
#
#    for j in range(m):
#        tmp = norm(xs[j], u[l], sig) / T[j]
#
#        total += tmp
#        signum += sum((xs[j] - un[l]) * (xs[j] - un[l])) * tmp
#
#    sig = numpy.sqrt(signum / total)

    u = un

    #print "Means: ", u
    #print "Intensities: ", I
    #print "sig: ", sig
    #print "---"
print "Time: ", time.time() - tmp1

plt.imshow(hist, interpolation = 'NONE')
plt.colorbar()
idxs = numpy.where(I > 0.1)
plt.plot(u[idxs, 1], u[idxs, 0], 'wo')
plt.show()