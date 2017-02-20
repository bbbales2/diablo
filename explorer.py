import skimage.color
import numpy
import networkx as nx
import pygame
import scipy
import scipy.spatial
import time

def fftalign(im1, im2, resolution = 1):
    f1 = skimage.color.rgb2gray(im1[::resolution, ::resolution])
    f2 = skimage.color.rgb2gray(im2[::resolution, ::resolution])

    F1 = numpy.fft.fft2(f1)
    F2 = numpy.fft.fft2(f2)

    Ftmp = F1 * F2.conj()
    #Ftmp /= numpy.linalg.norm(Ftmp)

    f = numpy.fft.fftshift(numpy.real(numpy.fft.ifft2(Ftmp)))
    loc = numpy.unravel_index(f.argmax(), f.shape)
    return numpy.linalg.norm(Ftmp), numpy.array([1, 1]) * resolution * (loc - numpy.array((f1.shape[0] / 2, f1.shape[1] / 2)))


class Bot(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.lscreen = None
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

    def draw(self, screen, gl):
        if self.lscreen is None:
            self.lscreen = screen
            self.buildTargets(100, -1000, 1000, -1000, 0)
            self.lcall = time.time()

            return []

        if (time.time() - self.lcall) > 0.075:
            S, (dy, dx) = fftalign(self.lscreen[:384], screen[:384], 2)

            self.x += dx
            self.y += dy

            self.lcall = time.time()

            self.lscreen = screen

        surf = pygame.Surface((gl.W, gl.H), pygame.SRCALPHA)
        surf.fill((255, 255, 255, 0))

        G = self.G

        locs = nx.get_node_attributes(G, 'loc')

        for node in G.nodes():
            if self.contains(locs[node]):
                x, y = self.g2s(locs[node])
                y = int(y)
                x = int(x)
                pygame.draw.circle(surf, [255, 255, 255], (x, y), 4)

                for edge in G.edges(node):
                    onode = edge[1]

                    if self.contains(locs[onode]):
                        x_, y_ = self.g2s(locs[onode])

                        x_ = int(x_)
                        y_ = int(y_)

                        pygame.draw.line(surf, [255, 255, 255], (x, y), (x_, y_), 2)

        gl.screen.blit(surf, (0, 0))

        return ["position: {0} {1}".format(self.x, self.y)]
