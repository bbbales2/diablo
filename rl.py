import skimage.color
import numpy
import networkx as nx
import pygame
import scipy
import scipy.spatial
import time
import threading
import sklearn.cluster

import sparse
import interface

class Bot(object):
    def __init__(self):
        self.lscreen = None
        self.lpeaks = None
        self.ldesc = None
        self.lock = threading.Lock()
        self.plock = threading.Lock()
        self.recording = []
        self.x = 0
        self.y = 0
        self.dxs = []
        self.dys = []
        self.brf = sparse.BRIEF(32, 16, g = 1.0)
        self.age = 0

    def __getstate__(self):
        a = dict(self.__dict__)

        del a['lock']

        return a
        
    def __setstate__(self, a):
        for k in a:
            self.__dict__[k] = a[k]
            
        #self.updateNeighbors()
        self.lock = threading.Lock()
        #self.plock = threading.Lock()
        
    def contains(self, p):
        if self.x - 320 < p[0] and self.x + 320 > p[0] and self.y - 220 < p[1] and self.y + 220 > p[1]:
            return True
        else:
            return False

    def handle(self, event, gl): 
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_g:
                pass

                #i = numpy.random.randint(0, len(self.neighbors.data))
                
                
    def g2s(self, g):
        y = -(self.y - g[1]) + 220
        x = -(self.x - g[0]) + 320

        return x, y

    def buildPeaksDesc(self, im):
        im1 = sparse.rgb2g(im[:380])

        peaks1 = sparse.harris(im1, 200)

        desc1 = self.brf.process(im1, peaks1)

        return peaks1, desc1

    def tick(self, gl):
        self.lock.acquire()

        dx = numpy.sqrt(numpy.array(self.dxs)**2 + numpy.array(self.dys)**2)

        self.age += sum(dx)

        self.dxs = []
        self.dys = []
        
        self.lock.release()

        x = numpy.random.randint(0, 640)
        y = numpy.random.randint(35, 395)
        
        self.recording.append([(x, y), (self.x, self.y), gl.getScreen()])

        gl.mk.click((x, y), 1)

    def get_offset(self, lpeaks, peaks, ldesc, desc, return_nidx = False):
        pairs = sparse.match(lpeaks, peaks, ldesc, desc)

        dxs = []
        dys = []

        nidx = set()
        for i, j in list(pairs):
            y0, x0 = lpeaks[i]
            y1, x1 = peaks[j]

            nidx.add(j)
            
            dxs.append(x1 - x0)
            dys.append(y1 - y0)

        ret = -numpy.median([dxs, dys], axis = 1)
            
        if return_nidx == False:
            return ret
        else:
            return ret, nidx
    
    def draw(self, screen, gl):
        if self.lscreen is None:
            self.lscreen = screen
            self.lpeaks, self.ldesc = self.buildPeaksDesc(screen)
            self.lcall = time.time()
            
            return []

        lpeaks, ldesc = self.lpeaks, self.ldesc

        peaks, desc = self.buildPeaksDesc(screen)

        (dx, dy), nidx = self.get_offset(lpeaks, peaks, ldesc, desc, return_nidx = True)

        self.lock.acquire()
        self.x += dx
        self.y += dy

        self.dxs.append(dx)
        self.dys.append(dy)

        self.lcall = time.time()

        self.lscreen = screen
        self.lpeaks = peaks
        self.ldesc = desc
        self.lock.release()

        #surf = pygame.Surface((gl.W, gl.H), pygame.SRCALPHA)
        #surf.fill((255, 255, 255, 0))

        #for j, peak in enumerate(self.lpeaks):
        #    y, x = peak

        #    if j in nidx:
        #        pygame.draw.circle(surf, [255, 0, 0], (x, y), 2)
        #    else:
        #        pygame.draw.circle(surf, [255, 255, 255], (x, y), 2)
                    
        #gl.screen.blit(surf, (0, 0))

        text = ["position: {0} {1}".format(self.x, self.y)]
        text.append("age : {0}".format(self.age))
        
        return text
