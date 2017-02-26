#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

cimport numpy
import numpy
from libc.math cimport sqrt
#from libc.math cimport exp

cpdef match(numpy.ndarray[numpy.int_t, ndim = 2] peaks1,
            numpy.ndarray[numpy.int_t, ndim = 2] peaks2,
            numpy.ndarray[numpy.uint8_t, ndim = 2] desc1,
            numpy.ndarray[numpy.uint8_t, ndim = 2] desc2,
            float t = 0):

    cdef int i, j, k, N, M

    cdef numpy.ndarray[numpy.float_t, ndim = 2] distances
    cdef numpy.ndarray[numpy.int_t, ndim = 2] hamming

    N = peaks1.shape[0]
    M = desc1.shape[1]

    distances = numpy.zeros((N, N))
    hamming = numpy.zeros((N, N), dtype = 'int')

    for i in range(N):
        for j in range(N):
            distances[i, j] = sqrt((peaks1[i, 0] - peaks2[j, 0]) * (peaks1[i, 0] - peaks2[j, 0]) + (peaks1[i, 1] - peaks2[j, 1]) * (peaks1[i, 1] - peaks2[j, 1]))

            for k in range(M):
                hamming[i, j] += 1 - desc1[i, k] ^ desc2[j, k]

    pairs1 = set(pair(distances, hamming, t))
    pairs2 = set(pair(distances.T, hamming.T, t))

    pairs = []
    for i, j in pairs1:
        if (j, i) in pairs2:
            pairs.append((i, j))

    return pairs

cdef pair(numpy.ndarray[numpy.float_t, ndim = 2] distances,
          numpy.ndarray[numpy.int_t, ndim = 2] hamming,
          float t = 0):
    cdef int i, j, j1, j2, N
    cdef float max1, max2

    N = distances.shape[0]

    pairs = []

    for i in range(N):
        j1 = -1
        j2 = -1
        max1 = 0
        max2 = 0
        for j in range(N):
            if distances[i, j] < 64.0:
                if hamming[i, j] > max1:
                    max2 = max1
                    max1 = hamming[i, j]

                    j2 = j1
                    j1 = j

        if max1 > 0 and max2 > 0:
            if float(max2) / max1 > t:
                pairs.append((i, j1))

    return pairs


cpdef process_(int N,
               numpy.ndarray[numpy.int_t, ndim = 2] pts1,
               numpy.ndarray[numpy.int_t, ndim = 2] pts2,
               numpy.ndarray[numpy.uint8_t, ndim = 2] im, kp):
    cdef unsigned int i, j
    cdef int y1, x1, y2, x2, cx, cy, L, M

    cdef numpy.ndarray[numpy.uint8_t, ndim = 2] bits

    L = im.shape[0]
    M = im.shape[1]

    bits = numpy.zeros((len(kp), N), dtype = 'uint8')

    for i in range(len(kp)):
        cy, cx = kp[i]
        for j in range(N):
            x1 = cx + pts1[j, 1]
            y1 = cy + pts1[j, 0]

            x2 = cx + pts2[j, 1]
            y2 = cy + pts2[j, 0]

            if y1 > 0 and y1 < L and y2 > 0 and y2 < L and x1 > 0 and x1 < M and x2 > 0 and x2 < M:
                if im[y1, x1] < im[y2, x2]:
                    bits[i, j] = 1
                else:
                    bits[i, j] = 0

    return bits
