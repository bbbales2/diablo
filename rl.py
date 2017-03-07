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

    def __getstate__(self):
        a = dict(self.__dict__)

        #del a['neighbors']
        del a['lock']
        #del a['plock']

        return a
        
    def __setstate__(self, a):
        for k in a:
            self.__dict__[k] = a[k]
            
        #self.updateNeighbors()
        self.lock = threading.Lock()
        #self.plock = threading.Lock()

    def updateNeighbors(self):
        self.neighbors = scipy.spatial.KDTree(self.G.nodes())

        self.kmeans.partial_fit(self.ldesc)

        threading.Thread(target = self.updateFramePredict).start()

    def updateFramePredict(self):
        Xs = []
        ys = []

        if len(self.G.nodes()) < 2:
            return

        self.plock.acquire()
        
        self.lr2n = {}
        
        for i, ((x, y), data) in enumerate(self.G.nodes(data = True)):
            counts = numpy.zeros(self.kmeans.n_clusters)
            
            cls = self.kmeans.predict(data['desc'].astype('float'))
            for c in cls:
                counts[c] += 1

            counts /= sum(counts)
                
            Xs.append(counts)
            ys.append(i)

            self.lr2n[i] = (x, y)

        self.lr = sklearn.linear_model.LogisticRegression()

        self.lr.fit(Xs, ys)

        self.plock.release()
        
    def snapshot(self):
        if not self.G:
            self.G = nx.Graph()

        if self.lscreen is None:
            return

        x = int(self.x)
        y = int(self.y)

        self.G.add_node((x, y), screen = self.lscreen, peaks = self.lpeaks, desc = self.ldesc, age = self.age)

        if self.neighbors:
            for d, i in zip(*self.neighbors.query((x, y), 3)):
                if numpy.isinf(d):
                    continue
                
                self.G.add_edge((x, y), tuple(self.neighbors.data[i].astype('int')), w = d)

        self.updateNeighbors()

    def needSnapshot(self):
        if not self.neighbors:
            return True
        
        d, i = self.neighbors.query((self.x, self.y), 1)

        if d > 200.0:
            return True
        else:
            x, y = tuple(self.neighbors.data[i].astype('int'))

            node = self.G.node[(x, y)]

            if self.age - node['age'] > 1000.0 and (x, y) != self.target:
                self.G.remove_node((x, y))

                self.updateNeighbors()

                return self.needSnapshot()
            else:
                return False
        
    def contains(self, p):
        if self.x - 320 < p[0] and self.x + 320 > p[0] and self.y - 220 < p[1] and self.y + 220 > p[1]:
            return True
        else:
            return False

    def handle(self, event, gl): 
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_g:
                i = numpy.random.randint(0, len(self.neighbors.data))
                
                self.target = tuple(self.neighbors.data[i].astype('int'))
                
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

        if self.needSnapshot():
            self.snapshot()
            print "Snapshot"

        if self.target is not None:
            d, i = self.neighbors.query((self.x, self.y), 1)

            x, y = self.neighbors.data[i].astype('int')

            if (x, y) == self.target:
                self.target = None

                gl.mouse.click(self.g2s((x, y)), 1)

                return

            path = nx.shortest_path(self.G, (x, y), self.target)

            dv = numpy.array(self.target) - numpy.array((x, y))
            dv = dv.astype('float')

            dv /= numpy.linalg.norm(dv)

            dv *= 150.0

            gl.mouse.click(self.g2s((self.x + dv[0], self.y + dv[1])), 1)

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

        surf = pygame.Surface((gl.W, gl.H), pygame.SRCALPHA)
        surf.fill((255, 255, 255, 0))

        for j, peak in enumerate(self.lpeaks):
            y, x = peak

            if j in nidx:
                pygame.draw.circle(surf, [255, 0, 0], (x, y), 2)
            else:
                pygame.draw.circle(surf, [255, 255, 255], (x, y), 2)
                    
        if self.neighbors:
            for d, i in zip(*self.neighbors.query((self.x, self.y), 10)):
                if numpy.isinf(d):
                    continue

                x, y = self.neighbors.data[i].astype('int')
                
                lx, ly = self.g2s((x, y))

                if lx >=0 and lx < gl.W and ly >= 0 and ly < gl.H:
                    if self.target == (x, y):
                        pygame.draw.circle(surf, [255, 255, 0], (int(lx), int(ly)), 5)
                    else:
                        pygame.draw.circle(surf, [0, 255, 0], (int(lx), int(ly)), 3)

        gl.screen.blit(surf, (0, 0))

        text = ["position: {0} {1}".format(self.x, self.y)]
        text.append("age : {0}".format(self.age))
        
        return text
