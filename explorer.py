import skimage.color
import numpy
import networkx as nx
import pygame
import scipy
import scipy.spatial
import time

import sparse

brf = sparse.BRIEF(32, 16, g = 1.0)

def get_offset(im1, im2):
    im1 = sparse.rgb2g(im1[:380])
    im2 = sparse.rgb2g(im2[:380])

    peaks1 = sparse.harris(im1, 200)
    peaks2 = sparse.harris(im2, 200)

    desc1 = brf.process(im1, peaks1)
    desc2 = brf.process(im2, peaks2)

    pairs = sparse.match(peaks1, peaks2, desc1, desc2)

    dxs = []
    dys = []

    for i, j in list(pairs):
        x0, y0 = peaks1[i]
        x1, y1 = peaks2[j]

        dxs.append(x1 - x0)
        dys.append(y1 - y0)

    return -numpy.median([dxs, dys], axis = 1)

class Bot(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.lscreen = None
        self.lpeaks = None
        self.ldesc = None
        self.lcall = 0.0

    def buildTargets(self, r, xmin, xmax, ymin, ymax):
        self.N = ((ymax - ymin) / r) * ((xmax - xmin) / r)
        self.G = nx.Graph()

        G = self.G

        G.add_node(0, { 'loc' : numpy.array([0, 0]) })

        locs = set((0, 0))

        while len(self.G.nodes()) < self.N:
            x = numpy.random.randint(xmin, xmax)
            y = numpy.random.randint(xmin, xmax)

            if (x, y) in locs:
                continue

            locs.add((x, y))
            G.add_node(len(self.G.nodes()), { "loc" : numpy.array((x, y)) })

        nodes, locs = zip(*[(node, data['loc']) for node, data in G.nodes(data = True)])

        tris = scipy.spatial.Delaunay(locs)

        nodesUsed = set()

        for tri in tris.simplices:
            for p0, p1 in [[tri[0], tri[1]],
                           [tri[1], tri[2]],
                           [tri[0], tri[2]]]:
                nodesUsed.add(p0)
                nodesUsed.add(p1)

                d = numpy.linalg.norm(locs[p0] - locs[p1])

                self.G.add_edge(nodes[p0], nodes[p1], weight = d)

        for node in set(G.nodes()) - set([nodes[n] for n in nodesUsed]):
            G.remove_node(node)

    def contains(self, p):
        if self.x - 320 < p[0] and self.x + 320 > p[0] and self.y - 220 < p[1] and self.y + 220 > p[1]:
            return True
        else:
            return False

    def handle(self, event, gl):
        pass

    def g2s(self, g):
        y = -(self.y - g[1]) + 220
        x = -(self.x - g[0]) + 320

        return x, y

    def buildPeaksDesc(self, im):
        im1 = sparse.rgb2g(im[:380])

        peaks1 = sparse.harris(im1, 200)

        desc1 = brf.process(im1, peaks1)

        return peaks1, desc1

    def draw(self, screen, gl):
        if self.lscreen is None:
            self.lscreen = screen
            self.lpeaks, self.ldesc = self.buildPeaksDesc(screen)
            self.lcall = time.time()

            return []

        lpeaks, ldesc = self.lpeaks, self.ldesc

        peaks, desc = self.buildPeaksDesc(screen)
            
        pairs = sparse.match(lpeaks, peaks, ldesc, desc)

        dxs = []
        dys = []

        nidx = set()
        for i, j in list(pairs):
            x0, y0 = lpeaks[i]
            x1, y1 = peaks[j]

            nidx.add(j)
            
            dxs.append(x1 - x0)
            dys.append(y1 - y0)
                
        dy, dx = -numpy.median([dxs, dys], axis = 1)

        self.x += dx
        self.y += dy

        self.lcall = time.time()

        self.lscreen = screen
        self.lpeaks = peaks#[nidx]
        self.ldesc = desc#[nidx]

        surf = pygame.Surface((gl.W, gl.H), pygame.SRCALPHA)
        surf.fill((255, 255, 255, 0))

        for j, peak in enumerate(self.lpeaks):
            y, x = peak

            if j in nidx:
                pygame.draw.circle(surf, [255, 0, 0], (x, y), 2)
            else:
                pygame.draw.circle(surf, [255, 255, 255], (x, y), 2)

        gl.screen.blit(surf, (0, 0))

        return ["position: {0} {1}".format(self.x, self.y)]
