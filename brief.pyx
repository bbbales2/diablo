#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

cimport numpy
import numpy
#from libc.math cimport exp


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
