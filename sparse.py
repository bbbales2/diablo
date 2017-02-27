import pyximport
pyximport.install(reload_support = False)

import brief_lib
import numpy
import cv2
import skimage.feature

from brief_lib import match

def rgb2g(im):
    img = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    img = cv2.equalizeHist(img)

    return img

def harris(img, N):
    corners = cv2.cornerHarris(img, 2, 3, 0.05)
    peaks = skimage.feature.corner_peaks(corners)
    values = [corners[i, j] for i, j in peaks]
    peaks = peaks[numpy.argsort(values)[-N:]]

    return peaks

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

        return brief_lib.process_(self.N, self.pts1, self.pts2, im, kp)

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
                    bits[i, j] = int(im[y1, x1] > im[y2, x2])

        return bits
