#%%

import os
os.chdir('/home/bbales2/diablo')

import cifar
import skimage.io

car = skimage.io.imread('car.png')

car = car[4:-4, 4:-4]

cifar.init(car)
#%%
import time
tmp = time.time()
out = cifar.eval(car)#sess.run([logits], { inp : [car] })[0]
print out.shape
print time.time() - tmp