#%%

import os
import matplotlib.pyplot as plt
import numpy
import cv2
import sklearn.decomposition
import imageio
import pickle
import scipy
import skimage.filters
import sys
import mahotas

os.chdir('/home/bbales2/diablo')

#%%

ims = []

if True:
    vid = imageio.get_reader('/home/bbales2/Videos/shop_vid.mp4', 'ffmpeg')

    W, H = vid.get_meta_data()['size']

    F = vid.get_length()
    for i, frame in enumerate(vid):
        #if i > 200:
        #    break

        ims.append(frame)

        print "Reading frame {0} / {1}".format(i, F)
        
#%%
import mahotas
import time

seged = []

samples = []
for i, im in enumerate(ims[::2]):
    seg = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
                                                          
    seg = (numpy.logical_and(seg[:, :, 2] > 75, im[:, :, 0] > 10)).astype('float')

    #plt.imshow(seg, cmap = plt.cm.viridis, interpolation = 'NONE')
    #plt.gcf().set_size_inches((12, 12))
    #plt.show()

    tmp = time.time()
    labeled, n = mahotas.labeled.label(seg)
    
    #plt.imshow(labeled, cmap = plt.cm.viridis, interpolation = 'NONE')
    #plt.gcf().set_size_inches((12, 12))
    #plt.show()
    
    boxes = mahotas.labeled.bbox(labeled)
    
    hs = boxes[:, 1] - boxes[:, 0]
    ws = boxes[:, 3] - boxes[:, 2]
    
    ratio = hs / ws.astype('float')
    
    sizes = mahotas.labeled.labeled_size(labeled)
    toremove = numpy.logical_or.reduce([ratio < 0.5, ratio > 4.0,
                                        sizes < 5, sizes > 100,
                                        hs < 7, hs > 10,
                                        ws <= 1, ws > 15])
                                             
    labeled2 = mahotas.labeled.remove_regions(labeled, numpy.where(toremove)[0])
    
    boxes = boxes[numpy.logical_not(toremove)]
    
    for i0, i1, j0, j1 in boxes:
        samp = seg[i0 : i1, j0 : j1]

        nx = 15 - samp.shape[1]
        ny = 10 - samp.shape[0]

        samp = numpy.pad(samp, ((0, ny), (0, nx)), 'constant', constant_values = 0)

        if i0 + 10 >= im.shape[1]:
            continue
        
        samples.append((samp, (i0, i1, j0, j1), im[i0 : i0 + 10, max(0, j0) : min(im.shape[1], j1 + 50)]))
        
        #plt.imshow(samples[-1][0])
        #plt.show()
        #plt.imshow(samples[-1][1])
        #plt.show()
    
    print time.time() - tmp
    
    print "Processed image {0} and have collected {1} samples".format(i, len(samples))
    
    #plt.imshow(labeled2 > 0, cmap = plt.cm.viridis, interpolation = 'NONE')
    #plt.gcf().set_size_inches((12, 12))
    #plt.show()


import pickle

with open('text_samples.pkl', 'w') as f:
    pickle.dump(samples, f)