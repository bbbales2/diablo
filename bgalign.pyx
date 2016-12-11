#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

cimport numpy
import numpy

#from libc.math cimport exp

cpdef inline float polyint(int n, int m, int l):
    cdef float xtmp, ytmp

    if n < 0 or m < 0 or l < 0 or n % 2 > 0 or m % 2 > 0 or l % 2 > 0:
        return 0.0

    xtmp = 2 * 0.5**(n + 1)
    ytmp = 2 * 0.5**(m + 1) * xtmp
    return 2 * 0.5**(l + 1) * ytmp / ((n + 1) * (m + 1) * (l + 1))

cpdef tuple offset(int R, numpy.ndarray[numpy.uint8_t, ndim = 3] im1, numpy.ndarray[numpy.uint8_t, ndim = 3] im2):
    cdef int M, N, dx, dy, i, j, mini, minj
    cdef double dr, dg, db, minV

    cdef numpy.ndarray[numpy.float_t, ndim = 2] result

    M = im1.shape[0]
    N = im1.shape[1]

    result = numpy.zeros((2 * R, 2 * R))

    for dx in range(-R, R):
        for dy in range(-R, R):
            for i in range(R, M - R, 48):
                for j in range(R, N - R, 48):
                    dr = float(im2[i, j, 0]) - float(im1[i + dy, j + dx, 0])
                    dg = float(im2[i, j, 1]) - float(im1[i + dy, j + dx, 1])
                    db = float(im2[i, j, 2]) - float(im1[i + dy, j + dx, 2])

                    result[dy + R, dx + R] += dr * dr + dg * dg + db * db

    minV = result[0, 0]
    mini = 0
    minj = 0

    for i in range(2 * R):
        for j in range(2 * R):
            if result[i, j] < minV:
                mini = i
                minj = j
                minV = result[i, j]

    return result, (-mini + R, -minj + R)

cpdef loffset(int R, numpy.ndarray[numpy.uint8_t, ndim = 3] im1, numpy.ndarray[numpy.uint8_t, ndim = 3] im2):
    cdef int M, N, dx, dy, i, j, ii, jj, mini, minj
    cdef double dr, dg, db, minv

    cdef numpy.ndarray[numpy.int_t, ndim = 3] result

    M = im1.shape[0]
    N = im1.shape[1]

    ni = (M - 2 * R) / 48
    nj = (N - 2 * R) / 48

    result = numpy.zeros((ni, nj, 2), dtype = 'int')

    #tmp = numpy.zeros((2 * R, 2 * R))

    for i in range(ni):
        for j in range(nj):
            minv = 3 * 255.0 * 255.0
            mini = 0
            minj = 0

            ii = R + i * 48
            jj = R + j * 48

            #print ii, jj

            for dx in range(-R, R):
                for dy in range(-R, R):
                    dr = float(im2[ii, jj, 0]) - float(im1[ii + dy, jj + dx, 0])
                    dg = float(im2[ii, jj, 1]) - float(im1[ii + dy, jj + dx, 1])
                    db = float(im2[ii, jj, 2]) - float(im1[ii + dy, jj + dx, 2])

                    v = dr * dr + dg * dg + db * db

                    v += dx * dx / 2.0 + dy * dy / 2.0

                    if v < minv:
                        mini = dy
                        minj = dx
                        minv = v

                        #print dy, dx, v

            result[i, j, 0] = mini
            result[i, j, 1] = minj

    return result#, tmp
