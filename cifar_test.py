#%%

import os
os.chdir('/home/bbales2/diablo')

import cifar
import skimage.io

car = skimage.io.imread('align.png')
#%%
cifar.init(480, 636)
#%%
import time
tmp = time.time()
out = cifar.eval(car[:, :636])#sess.run([logits], { inp : [car] })[0]
print out.shape
print time.time() - tmp