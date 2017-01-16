#%%
import imageio
import os
import numpy
import time
import json
import matplotlib.pyplot as plt
import skimage.color
import scipy.signal
import skimage.transform

import time
import skimage.io

import pyximport
pyximport.install(reload_support = True)#

os.chdir('/home/bbales2/diablo')

import bgalign
import label

vid = imageio.get_reader('monsters.mp4', 'ffmpeg')

frames = []
for i, im in enumerate(vid):
    frames.append(im)
    #skimage.io.imsave('monstersFrames/{0}.png'.format(i), im)
    print len(frames)
    #plt.imshow(frames[-1])
    #plt.show()
#%%
reload(bgalign)

dxs = []

idxs = range(len(frames))

M = 360
N = frames[0].shape[1]

Mal = M - 48

pts = []
for i in range(48, M - 48, 48):
    for j in range(48, N - 48, 48):
        pts.append((i, j))

pts = numpy.array(pts)

plt.imshow(frames[0], interpolation = 'NONE')
plt.plot(pts[:, 1], pts[:, 0], 'r+')
plt.show()
#%%

for i in range(1, len(frames)):
    #tmp = time.time()
    f1 = frames[i][:360].copy()#skimage.color.rgb2hsv(frames[i1])
    f0 = frames[i - 1][:360].copy()#skimage.color.rgb2hsv(frames[i0])

    f1[250:, 550:] = 0
    f0[250:, 550:] = 0

    res, o = bgalign.offset(24, f1, f0)

    #print time.time() - tmp

    #plt.subplot(2, 1, 1)
    #plt.imshow(f1)
    #plt.subplot(2, 1, 2)
    #plt.imshow(f0)
    #plt.gcf().set_size_inches((20, 10))
    #plt.show()

    print o

    #plt.imshow(res, interpolation = 'NONE')
    #plt.show()

    dxs.append(o)
    #intensities = label.image(8, f1, refs)

#%%
xs = numpy.cumsum(dxs, axis = 0)

plt.plot(xs[:, 1], -xs[:, 0], 'x')
plt.show()
#%%

import glob

reload(label)

refs = []

lnames = ['grass', 'goblin', 'quillbeast', 'zombie', 'sorc', 'rock', 'dirt']
lidxs = dict([(name, i) for i, name in enumerate(lnames)])

for l in lnames:
    hist = numpy.zeros(4096)
    for fname in glob.glob('sprites/{0}/*.png'.format(l)):
        sample = skimage.io.imread('{0}'.format(fname))

        hist += label.buildHist(sample)

    refs.append(hist / numpy.linalg.norm(hist))

refs = numpy.array(refs).T
mTm = refs.T.dot(refs)
#%%
import mahotas

reload(bgalign)
reload(label)

xs = numpy.arange(80, dtype = 'float')
ys = numpy.arange(45, dtype = 'float')

Ys, Xs = numpy.meshgrid(ys, xs, indexing = 'ij')

shapes = [numpy.ones((5, 5)),
          numpy.ones((3, 3)),
          numpy.ones((2, 4)),
          numpy.ones((4, 2)),
          numpy.ones((4, 2)),
          numpy.ones((2, 2)),
          numpy.ones((5, 5))]

#%%

dxs = []
signals = []

try:
    writer = imageio.get_writer('movie.mp4', fps = 24.0)

    cmap = plt.cm.jet

    for i in range(1, len(frames)):
        tmp = time.time()
        f1 = frames[i][:360]
        f0 = frames[i - 1][:360]

        res, o = bgalign.offset(24, f1, f0)
        it = label.image(8, f1, refs)
        print 'labeling', time.time() - tmp

        dxs.append(o)

        tmp = time.time()
        #for r in range(len(lnames)):
        #    it[:, :, r] = mahotas.convolve(it[:, :, r], shapes[r]) / shapes[r].sum()
        #tmp = time.time()

        labels = numpy.argmax(it, axis = 2)

        goblins = (labels == lidxs['goblin']) * 1.0

        #gweight = mahotas.convolve(goblins, numpy.ones((4, 2)))

        labeled, number = mahotas.label(goblins)

        #seeds = []
        #for n in range(number):
        #    seeds.append((numpy.where(labeled == n)[0][0], numpy.where(labeled == n)[1][0]))

        #ws = mahotas.cwatershed((gweight > 0.0) * 1, labeled > 0.0)

        #plt.imshow(ws, interpolation = 'NONE')
        #plt.show()

        #sizes = mahotas.labeled.labeled_size(labeled) * 1.0
        #too_small = numpy.where(sizes <= 10)
        #labeled = mahotas.labeled.remove_regions(labeled, too_small)

        rxs = mahotas.labeled.labeled_sum(Xs, labeled)
        rys = mahotas.labeled.labeled_sum(Ys, labeled)
        sizes = mahotas.labeled.labeled_size(labeled) * 1.0

        coms = numpy.array([rys / sizes, rxs / sizes]).T
        signals.append(coms[1:] * 8)
        print coms
        print 'mh', time.time() - tmp

        #mh.labeled_sum(array, labeled)
        plt.imshow(f1, interpolation = 'NONE')
        #plt.plot(coms[:, 1] * 8, coms[:, 0] * 8, 'o')
        #plt.show()
        #plt.imshow(labeled, interpolation = 'NONE')
        #plt.show()

        #labels = labeled / 1.0
        labels = labels / float(refs.shape[1])
        plt.imshow(labels * refs.shape[1], interpolation = 'NONE', vmin = 0, vmax = refs.shape[1], extent = (0, f1.shape[1], f1.shape[0], 0), alpha = 0.5)
        plt.colorbar()
        plt.show()
        #labels = intensities[:, :, 2] / intensities[:, :, 2].max()
        labels = skimage.transform.resize(labels, (f1.shape[0], f1.shape[1]), order = 0)

        towrite = f1 * 0.75 / 255.0 + cmap(labels)[:, :, :3] * 0.25
        #plt.imshow(towrite)
        #plt.show()
        writer.append_data(towrite)
        print i
finally:
    writer.close()
#%%
goblins = {}

xs = numpy.cumsum(dxs, axis = 0)

gid = 0

xts = {}
yts = {}

for t in range(len(signals)):
    for sp in signals[t]:
        if numpy.linalg.norm(sp - [296.0, 580.0]) < 10:
            continue

        s = sp + xs[t] - numpy.array([M / 2.0, N / 2.0])

        gkeys = []
        distances = []
        for g in goblins:
            gkeys.append(g)
            distances.append(numpy.linalg.norm(goblins[g]['pos'] - s))

        if len(distances) == 0:
            distance = 1.0e9
        else:
            i = numpy.argmin(distances)
            distance = distances[i]
            g = gkeys[i]

            goblins[g]['life'] *= 1.3

        if distance > 40.0:
            goblins[gid] = { 'pos' : s, 'vel' : 0.0, 'life' : 1.0 }
            g = gid

            gid += 1

            print "making new goblin at {0}".format(s)

        goblins[g]['vel'] = distance * 0.25 + goblins[g]['vel'] * 0.75
        goblins[g]['pos'] = s * 0.5 + goblins[g]['pos'] * 0.5

        if g not in xts:
            xts[g] = []
            yts[g] = []

        xts[g].append(goblins[g]['pos'][1])
        yts[g].append(goblins[g]['pos'][0])

        print "updating goblin {0}, {1}".format(goblins[g]['pos'], s, distance)

    toDel = []
    for g in goblins:
        goblins[g]['life'] *= 0.8

        if goblins[g]['life'] < 0.2:
            print "killing goblin at {0}".format(goblins[g]['pos'])
            toDel.append(g)

    for g in toDel:
        del goblins[g]

plt.plot(xs[:, 1], -xs[:, 0])
for k in xts.keys():
    plt.plot(xts[k], -numpy.array(yts[k]))
#%%
    #plt.show()

with writer.saving(fig, "writer_test.mp4", 10):
    for i in range(10):
        fig.set_data(frames[i])
        writer.grab_frame()
    #skimage.io.imsave('/home/bbales2/lineage_process/dec9/{0}.png'.format(i1), f1)

    #plt.subplot(2, 1, 1)
    #plt.imshow(f1)
    #plt.plot(pts[:, 1], pts[:, 0], 'r+')
    #plt.subplot(2, 1, 2)
    #plt.imshow(f0)
    #plt.gcf().set_size_inches((15, 10))
    #plt.show()
    #plt.imshow(res, interpolation = 'NONE')
    #plt.colorbar()
    #plt.show()

    #print o

    #out = scipy.signal.fftconvolve(f1[:, :, 0] * G, f0[:, :, 0], mode = 'same')

    #c = numpy.unravel_index(numpy.argmax(out), out.shape)

    #dxs.append([c - numpy.array([hy, hx])])

    #print time.time() - tmp
    #print dxs[-1], i1, i0

    #plt.imshow(out)
    #plt.colorbar()
    #plt.show()

#%%

#%%
#%%

reload(label)
b = 8

f1 = frames[300]

M = f1.shape[0]
N = f1.shape[1]

mb = M / b
nb = N / b

hists = numpy.zeros((mb, nb, refs.shape[0]))

tmp = time.time()
for bi in range(mb):
    for bj in range(nb):
        hists[bi, bj] = label.buildHist(f1[bi * b : (bi + 1) * b, bj * b : (bj + 1) * b])
print time.time() - tmp

tmp = time.time()
labels = label.image(8, f1, refs)
print time.time() - tmp

print bi

#%%
labels = numpy.zeros((mb, nb, len(refs)))
for bi in range(mb):
    for bj in range(nb):
        for i in range(len(refs)):
            labels[bi, bj, i] = hists[bi, bj].dot(refs[i])
#%%
tmp = time.time()
labels = label.image(8, f1, refs)
print time.time() - tmp
#%%
plt.imshow(f1, interpolation = 'NONE')
plt.imshow(numpy.argmax(labels, axis = 2), interpolation = 'NONE', extent = (0, f1.shape[1], f1.shape[0], 0), alpha = 0.5)
plt.show()

#%%
for i, fname in enumerate(fnames):
    plt.imshow(f1, interpolation = 'NONE')
    plt.imshow(labels[:, :, i], interpolation = 'NONE', extent = (0, f1.shape[1], f1.shape[0], 0), alpha = 0.5)
    plt.title(fname)
    plt.show()
#%%

pts = []

M = frames[0].shape[0]
N = frames[0].shape[1]
R = 12

for i in range(R, M - R, 48):
    for j in range(R, N - R, 48):
        pts.append((i, j))

pts = numpy.array(pts)

result = numpy.zeros((2 * R, 2 * R))
for dx in range(-R, R):
    for dy in range(-R, R):
        loss = 0.0
        #f0s = []
        #f1s = []
        for pt in pts:
            loss += numpy.linalg.norm(f0[pt[0], pt[1]] - f1[pt[0] + dy, pt[1] + dx])
            #f0s.append(f0[pt[0], pt[1]])
            #f1s.append(f1[pt[0] + dy, pt[1] + dx])

        result[dy + R, dx + R] = loss

        #f1[pts + numpy.array([dx, dy])]

    print dx

#%%

print numpy.unravel_index(numpy.argmin(result), result.shape) - numpy.array([R, R])

plt.imshow(result, interpolation = 'NONE')
plt.colorbar()
plt.show()

#%%

#%%

reload(bgalign)

tmp = time.time()
res = bgalign.loffset(12, f0, f1)
print time.time() - tmp

plt.imshow(res[1], interpolation = 'NONE')

print res

#print numpy.unravel_index(numpy.argmin(res[0]), res[0].shape) - numpy.array([R, R])

#plt.imshow(res[0], interpolation = 'NONE')
#plt.colorbar()
#plt.show()

